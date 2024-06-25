import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import networkx as nx
import matplotlib.pyplot as plt

from data_handle.load_data import load_social_media_data, load_wearable_data, load_academic_data, load_peer_interaction_data
from data_handle.preprocess_data import preprocess_social_media_data, preprocess_wearable_data, preprocess_academic_data, preprocess_peer_interaction_data
from features.social_media import SocialMediaFeatureExtractor
from features.wearable import WearableFeatureExtractor
from features.academic import AcademicFeatureExtractor
from features.peer_interaction import PeerInteractionFeatureExtractor
from models.multimodal_model import MultimodalFusion
from models.train_eval import train_model, evaluate_model

def visualize_graph(graph, title):
    nx.draw(graph, with_labels=True)
    plt.title(title)
    plt.show()

def main():
    # Loads the data
    social_media_data = load_social_media_data()
    wearable_data = load_wearable_data()
    academic_data = load_academic_data()
    peer_interaction_data = load_peer_interaction_data()

    # Preprocess the data
    social_media_data_processed = preprocess_social_media_data(social_media_data)
    wearable_data_processed = preprocess_wearable_data(wearable_data)
    academic_data_processed = preprocess_academic_data(academic_data)
    peer_interaction_data_processed = preprocess_peer_interaction_data(peer_interaction_data)

    # Feature extraction
    social_extractor = SocialMediaFeatureExtractor()
    wearable_extractor = WearableFeatureExtractor(input_size=wearable_data_processed.shape[1], hidden_size=50, num_layers=2)
    academic_extractor = AcademicFeatureExtractor(input_size=academic_data_processed.shape[1], output_size=50)
    peer_extractor = PeerInteractionFeatureExtractor(in_feats=peer_interaction_data_processed.shape[1], out_feats=128)

    social_features = social_extractor.extract_features(social_media_data_processed)
    wearable_features = wearable_extractor.extract_features(wearable_data_processed)
    academic_features = academic_extractor.extract_features(academic_data_processed)
    peer_features = peer_extractor.extract_features(peer_interaction_data_processed)

    # Creates a graph for peer interactions
    peer_graph = nx.Graph()
    peer_graph.add_edges_from(zip(peer_interaction_data['user_id'], peer_interaction_data['num_friends']))
    visualize_graph(peer_graph, "Peer Interaction Graph")

    # Prepares the dataset
    labels = torch.tensor(np.random.randint(0, 2, size=(social_features.shape[0], 1)), dtype=torch.float32)
    features = torch.cat((social_features, wearable_features, academic_features, peer_features), dim=1)
    
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initializes model, criterion, and optimizer
    model = MultimodalFusion(input_sizes=[768, 50, 50, 128], hidden_size=128)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trains and evaluate the model
    train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()