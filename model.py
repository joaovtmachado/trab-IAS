import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(ResNet50Transfer, self).__init__()
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        
        return self.resnet(x)