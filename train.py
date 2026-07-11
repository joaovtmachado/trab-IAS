import torch
import torch.nn as nn
import torch.optim as optim

# Importamos o modelo e a função que gera os dados
from model import SimpleCNN
from main import main as carregar_dados # Importando a função main do seu main.py

def train_model():
    # 1. Carregando os DataLoaders
    train_loader, val_loader, test_loader = carregar_dados()
    
    # 2. Configurando o Dispositivo (Usa GPU se disponível, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTreinando no dispositivo: {device}")
    
    # 3. Instanciando o modelo, função de erro e otimizador
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Variáveis para controlar os Checkpoints (do notebook 02)
    NUM_EPOCHS = 10
    best_val_acc = 0.0 # Guarda a melhor acurácia para saber se salva o modelo
    
    print("\nIniciando o treinamento...")
    
    for epoch in range(NUM_EPOCHS):
        # ==========================================
        # FASE 1: TREINAMENTO
        # ==========================================
        model.train() # Avisa ao PyTorch que estamos treinando (ativa dropout, gradientes, etc)
        train_loss = 0.0
        
        for images, labels, zooms in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Zera os gradientes
            optimizer.zero_grad()
            
            # Forward pass (Passa as imagens na rede)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass (Calcula o erro) e otimiza os pesos
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # ==========================================
        # FASE 2: VALIDAÇÃO (Do notebook 04)
        # ==========================================
        model.eval() # Avisa que é avaliação (desliga atualização de pesos)
        val_loss = 0.0
        corrects = 0
        
        # torch.no_grad() economiza muita memória na placa de vídeo, pois não guarda histórico para backward
        with torch.no_grad():
            for images, labels, zooms in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Pegar qual foi a classe prevista (0 ou 1)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        
        print(f"Época [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # ==========================================
        # FASE 3: CHECKPOINTS (Do notebook 02)
        # ==========================================
        # Se a acurácia dessa época for a melhor de todas, salvamos o modelo!
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"⭐ Nova melhor acurácia na Validação ({best_val_acc:.4f})! Salvando checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_checkpoint.pth')

    print("\nTreinamento Finalizado!")
    print(f"Melhor Acurácia de Validação: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()