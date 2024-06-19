import torch
import torch.nn as nn

class AcademicFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(AcademicFeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

    def extract_features(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        return self.forward(data_tensor).detach()
