import torch
from torch.utils.data import DataLoader, TensorDataset
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

# Function to visualize the graph
def visualize_graph(graph, title):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

# Loads data
social_media_data = load_social_media_data('data/social_media_data_large.csv')
wearable_data = load_wearable_data('data/wearable_data_large.csv')
academic_data = load_academic_data('data/academic_data_large.csv')
peer_interaction_data = load_peer_interaction_data('data/peer_interaction_data_large.csv')

# Preprocesses data
social_media_texts = preprocess_social_media_data(social_media_data)
wearable_data_processed = preprocess_wearable_data(wearable_data)
academic_data_processed = preprocess_academic_data(academic_data)
peer_interaction_data_processed = preprocess_peer_interaction_data(peer_interaction_data)

# Features extraction
social_extractor = SocialMediaFeatureExtractor()
wearable_extractor = WearableFeatureExtractor(input_size=wearable_data_processed.shape[1], hidden_size=50, num_layers=2)
academic_extractor = AcademicFeatureExtractor(input_size=academic_data_processed.shape[1], output_size=50)

# Creates a graph for peer interactions using NetworkX
peer_graph = nx.Graph()
peer_graph.add_edges_from(zip(peer_interaction_data['source'], peer_interaction_data['target']))
peer_features = preprocess_peer_interaction_data(peer_interaction_data)  # Example of how to preprocess graph data

# Visualize the peer interaction graph
visualize_graph(peer_graph, "Peer Interaction Graph")

# Extract features using the preprocessed data (dummy extractor, to be replaced with actual processing)
peer_extractor = PeerInteractionFeatureExtractor(in_feats=peer_features.shape[1], out_feats=128)

social_features = social_extractor.extract_features(social_media_texts)
wearable_features = wearable_extractor.extract_features(wearable_data_processed)
academic_features = academic_extractor.extract_features(academic_data_processed)
peer_features = peer_extractor.extract_features(peer_graph, peer_interaction_data_processed)

# Creates datasets and dataloaders
X_social = torch.tensor(social_features, dtype=torch.float32)
X_wearable = torch.tensor(wearable_features, dtype=torch.float32)
X_academic = torch.tensor(academic_features, dtype=torch.float32)
X_peer = torch.tensor(peer_features, dtype=torch.float32)

# Dummy labels for illustration purposes (replace with your actual labels)
labels = torch.tensor(np.random.randint(0, 2, size=(X_social.shape[0],)), dtype=torch.float32)

dataset = TensorDataset(X_social, X_wearable, X_academic, X_peer, labels)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}

# Initializes model, criterion, optimizer
model = MultimodalFusion(input_sizes=[768, 50, 50, 128], hidden_size=128)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Trains the model
train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

# Evaluates the model
evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()
