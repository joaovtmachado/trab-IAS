import torch.nn as nn
import torchvision.models as models

class MobileNetTransfer(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(MobileNetTransfer, self).__init__()
        
        # Carrega a MobileNet V2 pré-treinada
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Congela a parte de extração de características
        if freeze_features:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
                
        # Modifica o classificador final
        # A MobileNetV2 tem a camada final no índice 1 do bloco 'classifier'
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)