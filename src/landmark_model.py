import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkClassifier(nn.Module):
    """
    2-layer MLP classifier for 63-dim MediaPipe hand landmark features.
    Architecture: 63 → 128 → 64 → num_classes
    BatchNorm always applied — uses running stats at inference (eval mode).
    """

    def __init__(self, num_classes: int = 3, input_dim: int = 63):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim   = input_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, num_classes)   # output layer

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # BN runs in all modes — in eval it uses stored running mean/var
        h = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        h = self.dropout2(F.relu(self.bn2(self.fc2(h))))
        return self.fc3(h)

    def update_num_classes(self, new_num_classes: int) -> None:
        """Resize output layer, preserving weights for existing classes."""
        if new_num_classes == self.num_classes:
            return
        old_fc3 = self.fc3
        new_fc3 = nn.Linear(64, new_num_classes)
        nn.init.kaiming_normal_(new_fc3.weight, nonlinearity="relu")
        nn.init.zeros_(new_fc3.bias)
        with torch.no_grad():
            n_copy = min(self.num_classes, new_num_classes)
            new_fc3.weight.data[:n_copy] = old_fc3.weight.data[:n_copy].clone()
            new_fc3.bias.data[:n_copy]   = old_fc3.bias.data[:n_copy].clone()
        self.fc3 = new_fc3
        self.num_classes = new_num_classes

    def predict_probs(self, x: torch.Tensor):
        """Returns (class_idx, confidence, all_probs_list)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            confidence, class_idx = probs.max(-1)
            return int(class_idx.item()), float(confidence.item()), probs.squeeze(0).tolist()


if __name__ == "__main__":
    model = LandmarkClassifier(num_classes=40)
    dummy = torch.randn(4, 63)
    print("Output shape:", model(dummy).shape)
    idx, conf, _ = model.predict_probs(dummy[:1])
    print(f"Predict: class={idx}, conf={conf:.3f}")
