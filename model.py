import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Pode manter a sua classe SimpleCNN aqui em cima...

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(ResNet50Transfer, self).__init__()
        
        # 1. Carrega a ResNet50 com os pesos oficiais pré-treinados no ImageNet
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. Congelamento de Camadas (Conceito usado no seu projeto de referência)
        # Se freeze_features for True, nós "trancamos" os pesos das convoluções. 
        # Isso faz o modelo treinar MUITO mais rápido, pois a GPU só vai calibrar a última camada.
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # 3. Adaptar o Classificador Final
        # A ResNet50 original termina em uma camada chamada 'fc' que tem 2048 neurônios de entrada
        num_features = self.resnet.fc.in_features
        
        # Substituímos a camada final por uma nova camada Linear adaptada para o seu número de classes (2)
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Passa a imagem diretamente pela ResNet modificada
        return self.resnet(x)