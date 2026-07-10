import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class BreaKHisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.magnifications = [] # A Dica de Ouro!
        
        # O BreaKHis geralmente é dividido em pastas 'benign' e 'malignant'
        # Usamos o Pathlib (ou os.walk) para varrer todas as imagens .png
        for path in Path(root_dir).rglob('*.png'):
            caminho_str = str(path)
            
            # 1. Pegando o rótulo (0 para benigno, 1 para maligno)
            # Verifica se a palavra 'benign' ou 'malignant' está no caminho
            if 'benign' in caminho_str.lower():
                label = 0
            elif 'malignant' in caminho_str.lower():
                label = 1
            else:
                continue # Pula se não achar a classe
                
            # 2. Pegando o fator de ampliação (Zoom)
            # No BreaKHis, o zoom (40X, 100X, 200X, 400X) geralmente está no nome da pasta ou do arquivo.
            # Exemplo de pasta: .../40X/...
            if '40X' in caminho_str:
                mag = 40
            elif '100X' in caminho_str:
                mag = 100
            elif '200X' in caminho_str:
                mag = 200
            elif '400X' in caminho_str:
                mag = 400
                
            self.image_paths.append(caminho_str)
            self.labels.append(label)
            self.magnifications.append(mag)

    def __len__(self):
        
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mag = self.magnifications[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label, mag