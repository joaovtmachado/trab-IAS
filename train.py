import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from dataset import BreaKHisDataset
from cross_validation_metrics import calcular_metricas, get_cross_validation_metrics

MODELO_SELECIONADO = "resnet50"
DATA_DIR = "./BreaKHis_v1"
NUM_EPOCHS = 25
N_SPLITS = 5
TEST_SIZE_PACIENTES = 0.15
PATIENCE = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def carregar_modelo(nome_modelo, device):
    if nome_modelo == "resnet50":
        from model import ResNet50Transfer
        print("-> Instanciando ResNet50 com Transfer Learning...")
        return ResNet50Transfer(num_classes=2, freeze_features=True).to(device)
    elif nome_modelo == "vgg16":
        from vggnet import VGGNetTransfer
        print("-> Instanciando VGG16 com Transfer Learning...")
        return VGGNetTransfer(num_classes=2, freeze_features=True).to(device)
    elif nome_modelo == "mobilenet":
        from mobilenet import MobileNetTransfer
        print("-> Instanciando MobileNet com Transfer Learning...")
        return MobileNetTransfer(num_classes=2, freeze_features=True).to(device)
    else:
        raise ValueError(f"Modelo {nome_modelo} nao reconhecido.")


def pesos_de_classe(labels_treino, device):
    contagem = np.bincount(labels_treino, minlength=2).astype(np.float64)
    contagem[contagem == 0] = 1.0
    pesos = len(labels_treino) / (2.0 * contagem)
    return torch.tensor(pesos, dtype=torch.float).to(device)


def split_por_paciente(patients, labels):
    pacientes_unicos = np.array(sorted(set(patients)))
    rotulo_por_paciente = np.array(
        [labels[np.where(patients == p)[0][0]] for p in pacientes_unicos]
    )
    pac_treinoval, pac_teste = train_test_split(
        pacientes_unicos,
        test_size=TEST_SIZE_PACIENTES,
        stratify=rotulo_por_paciente,
        random_state=SEED,
    )
    conjunto_teste = set(pac_teste)
    idx_teste = np.array([i for i, p in enumerate(patients) if p in conjunto_teste])
    idx_treinoval = np.array(
        [i for i, p in enumerate(patients) if p not in conjunto_teste]
    )
    return idx_treinoval, idx_teste


def avaliar(model, loader, device, criterion):
    model.eval()
    perda_total = 0.0
    reais, preds, probs, mags = [], [], [], []
    with torch.no_grad():
        for images, labels_batch, mags_batch in loader:
            images = images.to(device)
            labels_dev = labels_batch.to(device)
            outputs = model(images)
            perda_total += criterion(outputs, labels_dev).item() * images.size(0)
            prob_maligno = torch.softmax(outputs, 1)[:, 1]
            _, predicao = torch.max(outputs, 1)
            reais.extend(labels_batch.numpy())
            preds.extend(predicao.cpu().numpy())
            probs.extend(prob_maligno.cpu().numpy())
            mags.extend(mags_batch.numpy())
    perda_media = perda_total / len(loader.dataset)
    return perda_media, reais, preds, probs, mags


def treinar_um_fold(model, train_loader, val_loader, device, criterion, checkpoint_name):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    melhor_val_loss = float("inf")
    epocas_sem_melhora = 0

    for epoch in range(NUM_EPOCHS):
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

        val_loss, reais, preds, _, _ = avaliar(model, val_loader, device, criterion)
        val_acc = np.mean(np.array(reais) == np.array(preds))
        print(
            f"Epoca [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            epocas_sem_melhora = 0
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, checkpoint_name)
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PATIENCE:
                print(f"Early stopping na epoca {epoch + 1}.")
                break


def train_kfold():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Modelo selecionado: {MODELO_SELECIONADO}\n")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_treino = BreaKHisDataset(DATA_DIR, transform=transform_train)
    dataset_eval = BreaKHisDataset(DATA_DIR, transform=transform_eval)

    labels = np.array(dataset_treino.labels)
    patients = np.array(dataset_treino.patients)

    idx_treinoval, idx_teste = split_por_paciente(patients, labels)
    y_treinoval = labels[idx_treinoval]
    grupos_treinoval = patients[idx_treinoval]

    print(f"Imagens treino+val: {len(idx_treinoval)} | Imagens teste: {len(idx_teste)}")
    print(f"Pacientes treino+val: {len(set(grupos_treinoval))} | "
          f"Pacientes teste: {len(set(patients[idx_teste]))}")
    assert set(grupos_treinoval).isdisjoint(set(patients[idx_teste]))

    gerador = torch.Generator()
    gerador.manual_seed(SEED)

    test_loader = DataLoader(
        Subset(dataset_eval, idx_teste), batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=2, worker_init_fn=seed_worker,
    )

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    reais_cv, preds_cv, probs_cv, mags_cv = [], [], [], []
    metricas_por_fold = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(idx_treinoval, y_treinoval, groups=grupos_treinoval)
    ):
        print("\n" + f" FOLD {fold + 1}/{N_SPLITS} ({MODELO_SELECIONADO}) ")

        global_train_idx = idx_treinoval[train_idx]
        global_val_idx = idx_treinoval[val_idx]

        pac_train = set(patients[global_train_idx])
        pac_val = set(patients[global_val_idx])
        print(f"Pacientes treino: {len(pac_train)} | Pacientes val: {len(pac_val)}")
        assert pac_train.isdisjoint(pac_val)

        train_loader = DataLoader(
            Subset(dataset_treino, global_train_idx), batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=True, num_workers=2, worker_init_fn=seed_worker, generator=gerador,
        )
        val_loader = DataLoader(
            Subset(dataset_eval, global_val_idx), batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=2, worker_init_fn=seed_worker,
        )

        criterion = nn.CrossEntropyLoss(weight=pesos_de_classe(labels[global_train_idx], device))
        model = carregar_modelo(MODELO_SELECIONADO, device)
        checkpoint_name = f"best_checkpoint_{MODELO_SELECIONADO}_fold_{fold + 1}.pth"

        treinar_um_fold(model, train_loader, val_loader, device, criterion, checkpoint_name)

        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, reais, preds, probs, mags = avaliar(model, val_loader, device, criterion)

        metricas_por_fold.append(calcular_metricas(reais, preds, probs))
        reais_cv.extend(reais)
        preds_cv.extend(preds)
        probs_cv.extend(probs)
        mags_cv.extend(mags)

    print(f"\nCV concluida para {MODELO_SELECIONADO}. Gerando metricas...")
    get_cross_validation_metrics(
        reais=reais_cv,
        predicoes=preds_cv,
        probabilidades=probs_cv,
        magnificacoes=mags_cv,
        metricas_por_fold=metricas_por_fold,
        nomes_classes=["Benigno", "Maligno"],
        pasta_saida=f"resultados_kfold_{MODELO_SELECIONADO}",
        titulo=f"Validacao Cruzada - {MODELO_SELECIONADO.upper()}",
    )

    criterion_teste = nn.CrossEntropyLoss()
    prob_acumulada = None
    y_teste, mags_teste = None, None
    for fold in range(N_SPLITS):
        checkpoint_name = f"best_checkpoint_{MODELO_SELECIONADO}_fold_{fold + 1}.pth"
        model = carregar_modelo(MODELO_SELECIONADO, device)
        model.load_state_dict(torch.load(checkpoint_name)["model_state_dict"])
        _, reais, _, probs, mags = avaliar(model, test_loader, device, criterion_teste)
        probs = np.array(probs)
        prob_acumulada = probs if prob_acumulada is None else prob_acumulada + probs
        y_teste, mags_teste = reais, mags

    prob_media = prob_acumulada / N_SPLITS
    preds_teste = (prob_media >= 0.5).astype(int)
    get_cross_validation_metrics(
        reais=y_teste,
        predicoes=preds_teste,
        probabilidades=prob_media,
        magnificacoes=mags_teste,
        nomes_classes=["Benigno", "Maligno"],
        pasta_saida=f"resultados_teste_{MODELO_SELECIONADO}",
        titulo=f"Teste Held-out - {MODELO_SELECIONADO.upper()}",
    )
    


if __name__ == "__main__":
    train_kfold()
