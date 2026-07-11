import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Importando a classe que você guardou no outro arquivo!
from dataset import BreaKHisDataset

def main():
    print("Iniciando a preparação dos dados...")
    
    # O tamanho 224x224 é o padrão exigido pelas redes pré-treinadas (como a ResNet)
    TIPO_IMAGEM = 224

    # Essa normalização usa a média e o desvio padrão do dataset ImageNet.
    # É OBRIGATÓRIO usar isso se for fazer Transfer Learning depois!
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # ==========================================
    # 1. Pipeline de Treino (Data Augmentation Agressivo)
    # ==========================================
    # Como são células de microscópio, não existe "em cima" ou "em baixo", 
    # então podemos rotacionar e espelhar à vontade!
    transform_treino = transforms.Compose([
        transforms.RandomResizedCrop(TIPO_IMAGEM, scale=(0.8, 1.0)), # Recorte aleatório
        transforms.RandomHorizontalFlip(p=0.5), # 50% de chance de espelhar horizontalmente
        transforms.RandomVerticalFlip(p=0.5),   # 50% de chance de espelhar verticalmente
        transforms.RandomRotation(degrees=180), # Rotaciona a imagem em até 180 graus
        transforms.ToTensor(),                  # Transforma a imagem em Tensor do PyTorch
        normalize                               # Normaliza as cores
    ])

    # ==========================================
    # 2. Pipeline de Validação e Teste (Sem Augmentation!)
    # ==========================================
    transform_val_teste = transforms.Compose([
        transforms.Resize((TIPO_IMAGEM, TIPO_IMAGEM)), # Apenas redimensiona, sem bagunçar
        transforms.ToTensor(),
        normalize
    ])

    # ==========================================
    # 3. Carregando os Datasets com as Transformações
    # ==========================================
    # ATENÇÃO: Coloque o caminho correto da pasta BreaKHis aqui
    caminho_do_dataset = './BreaKHis_v1' 
    
    # Criamos duas versões virtuais do mesmo dataset: uma bagunçada (treino) e uma limpa (val/teste)
    dataset_com_aug = BreaKHisDataset(root_dir=caminho_do_dataset, transform=transform_treino)
    dataset_sem_aug = BreaKHisDataset(root_dir=caminho_do_dataset, transform=transform_val_teste)

    # ==========================================
    # 4. Dividindo em Treino (70%), Validação (15%) e Teste (15%)
    # ==========================================
    indices = list(range(len(dataset_com_aug)))
    
    # Separa 70% para treino e 30% para o resto
    idx_treino, idx_resto = train_test_split(indices, test_size=0.3, random_state=42)
    # Divide os 30% restantes no meio (15% val, 15% teste)
    idx_val, idx_teste = train_test_split(idx_resto, test_size=0.5, random_state=42)

    # Amarramos os índices correspondentes aos datasets com os transforms corretos!
    dataset_treino = Subset(dataset_com_aug, idx_treino)
    dataset_val    = Subset(dataset_sem_aug, idx_val)
    dataset_teste  = Subset(dataset_sem_aug, idx_teste)

    print(f"Imagens de Treino: {len(dataset_treino)}")
    print(f"Imagens de Validação: {len(dataset_val)}")
    print(f"Imagens de Teste: {len(dataset_teste)}")

    # ==========================================
    # 5. Criando os DataLoaders
    # ==========================================
    TAMANHO_LOTE = 32 # Se o seu computador/GPU travar por falta de memória, reduza para 16 ou 8.

    # No treino, o shuffle=True é OBRIGATÓRIO para a rede não decorar a ordem
    train_loader = DataLoader(dataset_treino, batch_size=TAMANHO_LOTE, shuffle=True)
    
    # Na validação e teste, a ordem não importa
    val_loader = DataLoader(dataset_val, batch_size=TAMANHO_LOTE, shuffle=False)
    test_loader = DataLoader(dataset_teste, batch_size=TAMANHO_LOTE, shuffle=False)

    print("DataLoaders criados com sucesso! Prontos para treinar a rede.")
    
    # Retornamos os loaders para usarmos na próxima etapa (O Loop de Treino)
    return train_loader, val_loader, test_loader

# Faz o script rodar se for executado diretamente
if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()