import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("WARNING: torch_geometric not installed. GNNDetector will run in dummy mode.")


class SwarmSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SwarmSAGE, self).__init__()
        if not PYG_AVAILABLE:
            return
        # GraphSAGE computes node representations by aggregating features from local neighborhood
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if not PYG_AVAILABLE:
            return torch.zeros((x.size(0), 1))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class GNNDetector:
    def __init__(self):
        self.enabled = PYG_AVAILABLE
        if self.enabled:
            # Features: [account_age_normalized, post_frequency, text_toxicity_variance]
            self.model = SwarmSAGE(in_channels=3, hidden_channels=16, out_channels=1)
            self.model.eval()

            # For MVP, we apply a deterministic weight initialization so that
            # highly connected clusters with high post_frequency/toxicity trigger higher probability
            with torch.no_grad():
                self.model.conv1.lin_l.weight.fill_(0.5)
                self.model.conv1.lin_r.weight.fill_(0.5)
                self.model.conv2.lin_l.weight.fill_(0.5)
                self.model.conv2.lin_r.weight.fill_(0.5)

    def predict(self, social_context: dict) -> float:
        """
        social_context expected format:
        {
            "nodes": [
                {"id": "user_0", "features": [0.1, 0.8, 0.9]}, # Target user at index 0
                {"id": "user_1", "features": [0.2, 0.7, 0.8]}
            ],
            "edges": [
                [0, 1], # edge from user 0 to user 1
                [1, 0]
            ]
        }
        Returns the swarm_probability (0.0 to 1.0) for the target user (index 0).
        """
        if not self.enabled or not social_context or "nodes" not in social_context:
            return 0.1  # Default low bot score

        try:
            nodes = social_context.get("nodes", [])
            edges = social_context.get("edges", [])

            if not nodes:
                return 0.1

            x = torch.tensor([n["features"] for n in nodes], dtype=torch.float)

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            with torch.no_grad():
                out = self.model(x, edge_index)

            # Return the swarm probability for the target user (index 0)
            prob = float(out[0][0].item())

            # Scale it to a more observable range for the MVP demo (0.1 to 0.95)
            return min(0.95, max(0.1, prob * 2.0))
        except Exception as e:
            print(f"GNN Error: {e}")
            return 0.1
