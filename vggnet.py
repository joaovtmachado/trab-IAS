import torch.nn as nn
import torchvision.models as models

class VGGNetTransfer(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(VGGNetTransfer, self).__init__()
        
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
                
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)