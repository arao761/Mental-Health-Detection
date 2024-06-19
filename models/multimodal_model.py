import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, input_sizes, hidden_size):
        super(MultimodalFusion, self).__init__()
        self.fc1 = nn.Linear(sum(input_sizes), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, social, wearable, academic, peer):
        x = torch.cat((social, wearable, academic, peer), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
