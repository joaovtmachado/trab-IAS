import torch.nn as nn
import torchvision.models as models

class VGGNetTransfer(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(VGGNetTransfer, self).__init__()
        
        # Carrega a arquitetura VGG16 pré-treinada no ImageNet
        # Usamos weights=models.VGG16_Weights.DEFAULT nas versões novas do PyTorch
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Congela as camadas convolucionais (extratores de características)
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
                
        # Substitui a última camada de classificação para o nosso problema binário (Benigno/Maligno)
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)