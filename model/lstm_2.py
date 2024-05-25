import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch.nn.functional as F


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using CUDA

def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def read_data(pkl_path, label_encoder=None):
    features_list = []
    labels_list = []

    with open(pkl_path, 'rb') as file:
        raw_data = pickle.load(file)

    for label, _, coordinates in raw_data:
        flat_coordinates = [coord for pair in coordinates for coord in pair]
        reshaped_coordinates = np.array(flat_coordinates).reshape(29, 80)
        feature_tensor = torch.tensor(reshaped_coordinates, dtype=torch.float32)
        features_list.append(feature_tensor)
        labels_list.append(label)

    # Initialize label encoder if not provided
    if label_encoder is None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_list)
    else:
        labels_encoded = label_encoder.transform(labels_list)

    return features_list, labels_encoded, label_encoder


def collate_fn(batch):
    features, labels = zip(*batch)
    features = [f.clone().detach() for f in features]
    lengths = [len(f) for f in features]
    padded_features = pad_sequence(features, batch_first=True)
    return padded_features, torch.tensor(labels), torch.tensor(lengths)


# Custom Dataset class
class LipPointsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SequentialModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(SequentialModel, self).__init__()
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        # Third LSTM layer
        self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dims[2], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Apply LSTM layers
        x, _ = self.lstm1(x)  # x is the output of all timesteps
        x, _ = self.lstm2(x)  # x is the output of all timesteps
        x, (h_n, c_n) = self.lstm3(x)  # x is the output of all timesteps, h_n is the last hidden state
        # Apply the first fully connected layer
        x = F.relu(self.fc1(h_n[-1]))  # Using the last hidden state
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)



def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch_features, batch_labels, batch_lengths in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()

        # Monitor gradients
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm()
                print(f'Grad norm for {name}: {grad_norm}')  # Or use logging


        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions += (predicted_labels == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for batch_features, batch_labels, batch_lengths in dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            # Calculate accuracy
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

            # Collect all labels and predictions for confusion matrix
            all_predicted_labels.extend(predicted_labels.tolist())
            all_true_labels.extend(batch_labels.tolist())

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    return average_loss, accuracy, conf_matrix


def main():
    train_pkl_path = '../script/train.pkl'
    val_pkl_path = '../script/val.pkl'
    test_pkl_path = '../script/test.pkl'

    # Initialize label encoder and ensure it is only fitted once
    _, _, label_encoder = read_data(train_pkl_path)
    train_features_list, train_labels_encoded, _ = read_data(train_pkl_path, label_encoder)
    val_features_list, val_labels_encoded, _ = read_data(val_pkl_path, label_encoder)
    test_features_list, test_labels_encoded, _ = read_data(test_pkl_path, label_encoder)

    train_dataset = LipPointsDataset(train_features_list, train_labels_encoded)
    val_dataset = LipPointsDataset(val_features_list, val_labels_encoded)
    test_dataset = LipPointsDataset(test_features_list, test_labels_encoded)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Define model
    input_size = 80  # Number of features per time step
    num_classes = len(label_encoder.classes_)
    model = SequentialModel(input_dim=input_size, hidden_dims=[64, 128, 64], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and validation loop
    for epoch in range(150):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, _ = evaluate(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}')

    # Evaluate on test set after fine-tuning
    test_loss, test_accuracy, conf_matrix = evaluate(model, test_loader, criterion)
    print(f'Test Loss after Fine Tuning: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    plot_conf_matrix(conf_matrix)
    torch.save(model.state_dict(), 'lip_reading_model.pth')


# from utils.video import get_lip_points_video
#
# label_classes = np.array(['MONTH', 'MORNING', 'ORDER', 'RIGHT'], dtype='<U7')
# lstm_model = SequentialModel(input_dim=80, hidden_dims=[64, 128, 64], num_classes=len(label_classes))
# lstm_model.load_state_dict(torch.load('lip_reading_model.pth'))


# def predict_video(video_path):
#     lip_points, _ = get_lip_points_video(video_path)
#     flat_coordinates = [coord for pair in lip_points for coord in pair]
#     reshaped_coordinates = np.array(flat_coordinates).reshape(29, 80)
#     feature_tensor = torch.tensor(reshaped_coordinates, dtype=torch.float32)
#     feature_tensor = feature_tensor.unsqueeze(0)  # Adds a batch dimension
#
#     # Create a tensor for lengths; wrap the scalar length in a list to create a 1D tensor
#     lengths = torch.tensor([reshaped_coordinates.shape[0]], dtype=torch.int64)
#
#     lstm_model.eval()
#
#     with torch.no_grad():
#         output = lstm_model(feature_tensor, lengths)
#         predicted_label = output.argmax(dim=1)
#         print (predicted_label)
#         return predicted_label


if __name__ == '__main__':
    set_seed(42)
    main()
    # label = predict_video('../dataset/RIGHT/test/RIGHT_00048.mp4')
    # print (label)