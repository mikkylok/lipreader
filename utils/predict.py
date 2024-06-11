import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.video import get_lip_points_video


class LipReadingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LipReadingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        out = self.fc(output[:, -1, :])
        return out


def predict_video(video_path, average_sampling):
    # Prepare data
    start = time.time()
    lip_points, _ = get_lip_points_video(video_path, average_sampling=average_sampling)
    end_frame = time.time()
    flat_coordinates = [coord for pair in lip_points for coord in pair]
    reshaped_coordinates = np.array(flat_coordinates).reshape(29, 80)
    feature_tensor = torch.tensor(reshaped_coordinates, dtype=torch.float32)
    feature_tensor = feature_tensor.unsqueeze(0)

    # Model inference
    lstm_model.eval()
    with torch.no_grad():
        logits = lstm_model(feature_tensor)
        probabilities = F.softmax(logits, dim=1)
        end_inference = time.time()
        total_time = round(end_inference - start, 2)
        preprocess_time = round(end_frame - start, 2)
        inference_time = round(end_inference-end_frame, 2)
        print (f'Total Time: {total_time}s, Preprocess: {preprocess_time}s, Inference: {inference_time}s')
        predicted_index = probabilities.argmax(dim=1)
        prob = probabilities[0, predicted_index].item()
        predicted_label = label_classes[predicted_index]
        return predicted_label, prob, total_time


label_classes = ['ABOUT', 'ANSWER', 'FAMILY', 'FRIDAY', 'MIDDLE', 'PRICE', 'RIGHT',
                 'SEVEN', 'SOMETHING', 'THEIR']
lstm_model = LipReadingLSTM(input_size=80, hidden_size=128, num_classes=len(label_classes), num_layers=1)
lstm_model.load_state_dict(
    torch.load('model/lip_reading_model_lstm_fixed_len_128_1.pth', map_location=torch.device('cpu')))