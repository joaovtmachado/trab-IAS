import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calcular_metricas(reais, predicoes, probabilidades=None):
    reais = np.array(reais)
    predicoes = np.array(predicoes)

    acuracia = accuracy_score(reais, predicoes)
    f1 = f1_score(reais, predicoes, average="macro")
    cm = confusion_matrix(reais, predicoes, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = None
    if probabilidades is not None and len(np.unique(reais)) > 1:
        auc = roc_auc_score(reais, np.array(probabilidades))

    return {
        "acuracia": acuracia,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "f1": f1,
        "auc": auc,
        "cm": cm,
    }


def _salvar_matriz_confusao(cm, nomes_classes, caminho_png, titulo):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(nomes_classes)))
    ax.set_yticks(range(len(nomes_classes)))
    ax.set_xticklabels(nomes_classes)
    ax.set_yticklabels(nomes_classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    limite = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > limite else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)


def _salvar_curva_roc(reais, probabilidades, auc, caminho_png, titulo):
    fpr, tpr, _ = roc_curve(np.array(reais), np.array(probabilidades))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)


def _bloco_por_magnificacao(reais, predicoes, magnificacoes):
    reais = np.array(reais)
    predicoes = np.array(predicoes)
    magnificacoes = np.array(magnificacoes)
    linhas = ["ANALISE POR MAGNIFICACAO:"]
    linhas.append(f"{'Mag':>6} {'N':>7} {'Acuracia':>10} {'Sensib':>9} {'Especif':>9}")
    for mag in [40, 100, 200, 400]:
        mascara = magnificacoes == mag
        if mascara.sum() == 0:
            continue
        m = calcular_metricas(reais[mascara], predicoes[mascara])
        linhas.append(
            f"{mag:>6} {int(mascara.sum()):>7} "
            f"{m['acuracia']:>10.4f} {m['sensibilidade']:>9.4f} {m['especificidade']:>9.4f}"
        )
    return "\n".join(linhas)


def _bloco_media_desvio(metricas_por_fold):
    chaves = ["acuracia", "sensibilidade", "especificidade", "f1", "auc"]
    linhas = ["METRICAS POR FOLD (media +/- desvio):"]
    for chave in chaves:
        valores = [f[chave] for f in metricas_por_fold if f.get(chave) is not None]
        if not valores:
            continue
        media = float(np.mean(valores))
        desvio = float(np.std(valores))
        linhas.append(f"{chave.capitalize():>16}: {media:.4f} +/- {desvio:.4f}")
    return "\n".join(linhas)


def get_cross_validation_metrics(
    reais,
    predicoes,
    probabilidades=None,
    magnificacoes=None,
    metricas_por_fold=None,
    nomes_classes=["Benigno", "Maligno"],
    pasta_saida="resultados_kfold",
    titulo="Validacao Cruzada",
):
    os.makedirs(pasta_saida, exist_ok=True)

    m = calcular_metricas(reais, predicoes, probabilidades)
    report_texto = classification_report(
        np.array(reais), np.array(predicoes), target_names=nomes_classes
    )

    
    print(f"Acuracia Geral:   {m['acuracia']:.4f}")
    print(f"Sensibilidade:    {m['sensibilidade']:.4f} (Recall de Malignos)")
    print(f"Especificidade:   {m['especificidade']:.4f} (Recall de Benignos)")
    print(f"F1-Score (Macro): {m['f1']:.4f}")
    if m["auc"] is not None:
        print(f"AUC-ROC:          {m['auc']:.4f}")
    print("\nMatriz de Confusao:")
    print(m["cm"])
    print("\nRelatorio Detalhado por Classe:")
    print(report_texto)
    print("#" * 40)

    _salvar_matriz_confusao(
        m["cm"], nomes_classes,
        os.path.join(pasta_saida, "matriz_confusao.png"),
        f"Matriz de Confusao - {titulo}",
    )
    if probabilidades is not None and m["auc"] is not None:
        _salvar_curva_roc(
            reais, probabilidades, m["auc"],
            os.path.join(pasta_saida, "curva_roc.png"),
            f"Curva ROC - {titulo}",
        )

    caminho_txt = os.path.join(pasta_saida, "relatorio_final_metricas.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(f" RELATORIO - {titulo} - PASTA: {pasta_saida}\n")
        f.write(f"Acuracia Geral:   {m['acuracia']:.4f}\n")
        f.write(f"Sensibilidade (Recall Maligno): {m['sensibilidade']:.4f}\n")
        f.write(f"Especificidade (Recall Benigno): {m['especificidade']:.4f}\n")
        f.write(f"F1-Score (Macro):       {m['f1']:.4f}\n")
        if m["auc"] is not None:
            f.write(f"AUC-ROC:                {m['auc']:.4f}\n")
        f.write("\n")
        if metricas_por_fold:
            f.write(_bloco_media_desvio(metricas_por_fold) + "\n\n")
        f.write("MATRIZ DE CONFUSAO:\n")
        f.write(np.array2string(m["cm"]) + "\n\n")
        f.write(report_texto + "\n\n")
        if magnificacoes is not None:
            f.write(_bloco_por_magnificacao(reais, predicoes, magnificacoes) + "\n")

    print(f"\n[Sucesso] Relatorio tecnico salvo em: '{caminho_txt}'")
    return m
