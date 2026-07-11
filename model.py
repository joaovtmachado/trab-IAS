import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Imagem de entrada: 3 canais (RGB), 224x224 pixels
        # Camada 1: Extrai 16 mapas de características
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada 2: Extrai 32 mapas de características
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Camada 3: Extrai 64 mapas de características
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Como passamos pelo MaxPooling 3 vezes (reduz pela metade cada vez):
        # 224 -> 112 -> 56 -> 28. O tamanho final da imagem achatada será 64 canais * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        
        # Camada de saída final: 2 classes (0: Benigno, 1: Maligno)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # Passa pela convolução, aplica a função de ativação ReLU e diminui o tamanho com Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # "Achata" os mapas de características para passar para os neurônios tradicionais (Linear)
        x = x.view(-1, 64 * 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Sem ativação final aqui porque usaremos a CrossEntropyLoss depois
        return x