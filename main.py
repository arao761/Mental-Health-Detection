import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import networkx as nx
import matplotlib.pyplot as plt

from data_handle.load_data import load_social_media_data, load_wearable_data, load_academic_data, load_peer_interaction_data
from data_handle.preprocess_data import preprocess_social_media_data, preprocess_wearable_data, preprocess_academic_data, preprocess_peer_interaction_data
from models.multimodal_model import MultimodalFusion
from models.train_eval import train_model, evaluate_model
from features.gnn_features import GNNFeatureExtractor
from features.nlp_features import extract_nlp_features
from features.time_series import RNNFeatureExtractor

# Loads and preprocesses the data
social_media_data = preprocess_social_media_data(load_social_media_data())
wearable_data = preprocess_wearable_data(load_wearable_data())
academic_data = preprocess_academic_data(load_academic_data())
peer_interaction_data = preprocess_peer_interaction_data(load_peer_interaction_data())

# Feature extraction
nlp_features = extract_nlp_features(social_media_data['text'])
rnn_feature_extractor = RNNFeatureExtractor(input_size=10, hidden_size=50, num_layers=2)
time_series_features = extract_features(wearable_data, rnn_feature_extractor)
gnn_feature_extractor = GNNFeatureExtractor(input_size=10, hidden_size=50, output_size=128)
graph_features = extract_features(peer_interaction_data, gnn_feature_extractor)

# Combines features
combined_features = torch.cat((nlp_features, time_series_features, graph_features), dim=1)

# Creates labels
labels = torch.tensor(np.random.randint(0, 2, size=(combined_features.shape[0],)), dtype=torch.float32)

# Dataset preparation
dataset = TensorDataset(combined_features, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}

# Model initialization
model = MultimodalFusion(input_sizes=[nlp_features.shape[1], time_series_features.shape[1], graph_features.shape[1]], hidden_size=128)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation
train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()
