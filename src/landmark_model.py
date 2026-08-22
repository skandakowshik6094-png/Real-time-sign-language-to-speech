import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkClassifier(nn.Module):
    """
    High-Precision PyTorch Neural Network Classifier for MediaPipe 3D Hand Landmarks.
    Classifies 63-dimensional normalized finger joint vectors (21 joints x 3D coords).
    """
    def __init__(self, num_classes=3, input_dim=63):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Batch Normalization requires batch size > 1 during training
        if x.shape[0] == 1 or not self.training:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            logits = self.fc3(h)
        else:
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.dropout1(h)
            h = F.relu(self.bn2(self.fc2(h)))
            h = self.dropout2(h)
            logits = self.fc3(h)

        return logits

    def update_num_classes(self, new_num_classes):
        """
        Dynamically adjusts the final classification layer if new classes are added.
        """
        if new_num_classes == self.num_classes:
            return

        old_num_classes = self.num_classes
        new_fc3 = nn.Linear(64, new_num_classes)
        with torch.no_grad():
            min_cls = min(old_num_classes, new_num_classes)
            new_fc3.weight[:min_cls] = self.fc3.weight[:min_cls]
            new_fc3.bias[:min_cls] = self.fc3.bias[:min_cls]

        self.fc3 = new_fc3
        self.num_classes = new_num_classes

    def predict_probs(self, x):
        """
        Returns class probabilities and predicted class index for landmark feature input.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, class_idx = probs.max(-1)
            return int(class_idx.item()), float(confidence.item()), probs.squeeze(0).tolist()

if __name__ == "__main__":
    model = LandmarkClassifier(num_classes=3)
    dummy_x = torch.randn(4, 63)
    out = model(dummy_x)
    print("LandmarkClassifier initialized! Output shape:", out.shape)
