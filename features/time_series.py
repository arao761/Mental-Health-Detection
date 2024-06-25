import torch.nn as nn

class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNFeatureExtractor, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        h, _ = self.rnn(x)
        return h[:, -1, :]
