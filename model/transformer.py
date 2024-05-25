import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

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

    if label_encoder is None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_list)
    else:
        labels_encoded = label_encoder.transform(labels_list)

    return features_list, labels_encoded, label_encoder

class LipPointsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def collate_fn(batch):
    features, labels = zip(*batch)
    features = [torch.as_tensor(f, dtype=torch.float32) for f in features]
    lengths = [len(f) for f in features]
    padded_features = pad_sequence(features, batch_first=True)
    return padded_features, torch.tensor(labels)

class LipReadingTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes):
        super(LipReadingTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src)
        transformed = self.transformer_encoder(src)
        output = self.fc_out(transformed[:, -1, :])
        return output

def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy


def main():
    train_pkl_path = '../script/train.pkl'
    val_pkl_path = '../script/val.pkl'

    _, _, label_encoder = read_data(train_pkl_path)
    train_features_list, train_labels_encoded, _ = read_data(train_pkl_path, label_encoder)
    val_features_list, val_labels_encoded, _ = read_data(val_pkl_path, label_encoder)

    train_dataset = LipPointsDataset(train_features_list, train_labels_encoded)
    val_dataset = LipPointsDataset(val_features_list, val_labels_encoded)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    input_dim = 80  # Flattened frame features
    d_model = 512  # Size of the Transformer embeddings
    nhead = 4  # Number of heads in multi-head attention models
    num_encoder_layers = 3  # Number of sub-encoder-layers in the encoder
    num_classes = len(label_encoder.classes_)  # Based on unique labels

    model = LipReadingTransformer(input_dim, d_model, nhead, num_encoder_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}')


if __name__ == '__main__':
    main()
