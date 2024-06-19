import torch
import dgl
from dgl.nn import GraphConv

class PeerInteractionFeatureExtractor(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(PeerInteractionFeatureExtractor, self).__init__()
        self.gcn = GraphConv(in_feats, out_feats)

    def forward(self, g, features):
        g.ndata['h'] = features
        h = self.gcn(g, g.ndata['h'])
        return h.mean(1)

    def extract_features(self, graph, features):
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return self.forward(graph, features_tensor).detach()
