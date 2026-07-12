import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import os
import torchvision.transforms as transforms

# Importações dos seus arquivos locais do projeto
from dataset import BreaKHisDataset
from cross_validation_metrics import get_cross_validation_metrics

# =========================================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# =========================================================================
# Escolha qual modelo treinar neste experimento: 'resnet50', 'vgg16' ou 'mobilenet'
MODELO_SELECIONADO = 'mobilenet'  
DATA_DIR = "./BreaKHis_v1" 
NUM_EPOCHS = 10  # Pode aumentar para 30 ou 50 para os resultados finais do artigo
# =========================================================================

def carregar_modelo(nome_modelo, device):
    """Função auxiliar para instanciar o modelo escolhido dinamicamente."""
    if nome_modelo == 'resnet50':
        from model import ResNet50Transfer
        print("-> Instanciando ResNet50 com Transfer Learning...")
        return ResNet50Transfer(num_classes=2, freeze_features=True).to(device)
        
    elif nome_modelo == 'vgg16':
        from vggnet import VGGNetTransfer
        print("-> Instanciando VGG16 com Transfer Learning...")
        return VGGNetTransfer(num_classes=2, freeze_features=True).to(device)
        
    elif nome_modelo == 'mobilenet':
        from mobilenet import MobileNetTransfer
        print("-> Instanciando MobileNet com Transfer Learning...")
        return MobileNetTransfer(num_classes=2, freeze_features=True).to(device)
    else:
        raise ValueError(f"Modelo {nome_modelo} não reconhecido. Escolha entre 'resnet50', 'vgg16' ou 'mobilenet'.")

def train_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando no dispositivo: {device}")
    print(f"Modelo selecionado para este pipeline: {MODELO_SELECIONADO.upper()}\n")

    # 2. Definição das Transformações de Imagem (Data Augmentation Agressivo no Treino)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_completo_treino = BreaKHisDataset(DATA_DIR, transform=transform_train)
    dataset_completo_eval = BreaKHisDataset(DATA_DIR, transform=transform_eval)

    indices = np.arange(len(dataset_completo_treino))
    labels = np.array(dataset_completo_treino.labels)

    # FASE A: Isolar o Teste Final (15%)
    idx_treino_val, idx_teste, y_treino_val, y_teste = train_test_split(
        indices, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # FASE B: Configurar a Validação Cruzada (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    todas_labels_reais = []
    todas_predicoes = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(idx_treino_val, y_treino_val)):
        print(f"\n" + "="*20 + f" FOLD {fold + 1} / 5 ({MODELO_SELECIONADO.upper()}) " + "="*20)
        
        global_train_idx = idx_treino_val[train_idx]
        global_val_idx = idx_treino_val[val_idx]
        
        subset_treino = Subset(dataset_completo_treino, global_train_idx)
        subset_val = Subset(dataset_completo_eval, global_val_idx)
        
        train_loader = DataLoader(subset_treino, batch_size=32, shuffle=True, pin_memory=True)
        val_loader = DataLoader(subset_val, batch_size=32, shuffle=False, pin_memory=True)
        
        # Chamada dinâmica do modelo configurado
        model = carregar_modelo(MODELO_SELECIONADO, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        best_val_acc = 0.0
        # O nome do checkpoint agora protege os resultados para não sobrescrever outros modelos
        checkpoint_name = f'best_checkpoint_{MODELO_SELECIONADO}_fold_{fold + 1}.pth'
        
        for epoch in range(NUM_EPOCHS):
            # TREINAMENTO
            model.train()
            train_loss = 0.0
            for images, labels_batch, _ in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_loader.dataset)
            
            # VALIDAÇÃO
            model.eval()
            val_loss = 0.0
            corrects = 0
            with torch.no_grad():
                for images, labels_batch, _ in val_loader:
                    images, labels_batch = images.to(device), labels_batch.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item() * images.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    corrects += torch.sum(preds == labels_batch.data)
                    
            val_loss /= len(val_loader.dataset)
            val_acc = corrects.double() / len(val_loader.dataset)
            
            print(f"Época [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f" Nova melhor acurácia no Fold {fold + 1} ({best_val_acc:.4f})! Salvando...")
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'val_acc': best_val_acc.item()
                }, checkpoint_name)
                
        print(f"Acurácia final máxima no Fold {fold + 1}: {best_val_acc:.4f}")
        
        # COMPILAÇÃO DE RESULTADOS DO FOLD
        print(f"Armazenando predições do Fold {fold + 1}...")
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            for images, labels_batch, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                todas_labels_reais.extend(labels_batch.cpu().numpy())
                todas_predicoes.extend(preds.cpu().numpy())

    # FASE C: GERAÇÃO DO RELATÓRIO GLOBAL ISOLADO POR MODELO
    print(f"\nTreinamento de {MODELO_SELECIONADO.upper()} concluído! Gerando métricas...")
    get_cross_validation_metrics(
        reais=todas_labels_reais, 
        predicoes=todas_predicoes, 
        nomes_classes=['Benigno', 'Maligno'],
        pasta_saida=f"resultados_kfold_{MODELO_SELECIONADO}"
    )
    print("Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    train_kfold()