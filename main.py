import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import BreaKHisDataset

def main():
    print("Iniciando a preparação dos dados...")
    
    TIPO_IMAGEM = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_treino = transforms.Compose([
        transforms.RandomResizedCrop(TIPO_IMAGEM, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),   
        transforms.RandomRotation(degrees=180), 
        transforms.ToTensor(),                  
        normalize                               
    ])

    transform_val_teste = transforms.Compose([
        transforms.Resize((TIPO_IMAGEM, TIPO_IMAGEM)), 
        transforms.ToTensor(),
        normalize
    ])

    caminho_do_dataset = './BreaKHis_v1' 
    
    dataset_com_aug = BreaKHisDataset(root_dir=caminho_do_dataset, transform=transform_treino)
    dataset_sem_aug = BreaKHisDataset(root_dir=caminho_do_dataset, transform=transform_val_teste)

    indices = list(range(len(dataset_com_aug)))
    
    idx_treino, idx_resto = train_test_split(indices, test_size=0.3, random_state=42)
    idx_val, idx_teste = train_test_split(idx_resto, test_size=0.5, random_state=42)

    dataset_treino = Subset(dataset_com_aug, idx_treino)
    dataset_val    = Subset(dataset_sem_aug, idx_val)
    dataset_teste  = Subset(dataset_sem_aug, idx_teste)

    print(f"Imagens de Treino: {len(dataset_treino)}")
    print(f"Imagens de Validação: {len(dataset_val)}")
    print(f"Imagens de Teste: {len(dataset_teste)}")

    TAMANHO_LOTE = 32 

    train_loader = DataLoader(dataset_treino, batch_size=TAMANHO_LOTE, shuffle=True)
    
    val_loader = DataLoader(dataset_val, batch_size=TAMANHO_LOTE, shuffle=False)
    test_loader = DataLoader(dataset_teste, batch_size=TAMANHO_LOTE, shuffle=False)

    print("DataLoaders criados com sucesso! Prontos para treinar a rede.")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()