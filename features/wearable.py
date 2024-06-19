import torch
import torch.nn as nn

class WearableFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WearableFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn[-1]

    def extract_features(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        return self.forward(data_tensor).detach()
