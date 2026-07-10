#só pra testar os arquivos menores por enquanto


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import dataset

transformacoes_teste = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

caminho_do_dataset = './BreaKHis_v1' 

try:
    meu_dataset = dataset.BreaKHisDataset(root_dir=caminho_do_dataset, transform=transformacoes_teste)
    print(f"Dataset carregado com sucesso! Total de imagens: {len(meu_dataset)}")
    
    meu_dataloader = DataLoader(meu_dataset, batch_size=4, shuffle=True)

    # 4. Pegar num lote (batch) de imagens para testar
    imagens, rotulos, zooms = next(iter(meu_dataloader))

    print("-" * 40)
    print(f"Formato do batch de imagens: {imagens.shape}") # Esperado: [4, 3, 224, 224] (Batch, Canais, Altura, Largura)
    print(f"Rótulos do batch: {rotulos}") # Esperado: 0 para Benigno, 1 para Maligno
    print(f"Zooms do batch: {zooms}X")
    print("-" * 40)

    # 5. Visualizar a primeira imagem do batch
    img_para_plot = imagens[0] # Pega a primeira imagem
    rotulo_para_plot = rotulos[0].item()
    zoom_para_plot = zooms[0].item()

    img_plot_format = img_para_plot.permute(1, 2, 0).numpy()

    nome_classe = "Maligno" if rotulo_para_plot == 1 else "Benigno"

    plt.figure(figsize=(6, 6))
    plt.imshow(img_plot_format)
    plt.title(f"Classe: {nome_classe} | Ampliação: {zoom_para_plot}X")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Ocorreu um erro ao carregar o dataset: {e}")
    print("Verifique se o 'caminho_do_dataset' está correto e se contém as imagens .png!")