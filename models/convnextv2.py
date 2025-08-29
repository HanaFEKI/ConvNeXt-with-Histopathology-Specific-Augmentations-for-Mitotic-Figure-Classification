import torch
import torch.nn as nn
import timm

class ConvNextV2Classifier(nn.Module):
    def __init__(self, model_name: str = "convnextv2_base",
                 weights: str = "DEFAULT",
                 num_classes: int = 1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.model(x)
        if self.num_classes == 1:  # Binary classification
            logits = logits.squeeze(1)
            y_prob = torch.sigmoid(logits)
            y_hat = (y_prob > 0.5).float()
        else:  # Multi-class
            y_prob = torch.softmax(logits, dim=1)
            y_hat = torch.argmax(y_prob, dim=1)
        return logits, y_prob, y_hat
