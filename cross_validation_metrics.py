import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score

def get_cross_validation_metrics(reais, predicoes, nomes_classes=['Benigno', 'Maligno'], pasta_saida="resultados_kfold"):
    """
    Calcula e exporta as métricas robustas de validação cruzada exigidas pela área de saúde.
    """
    # Garante que a pasta de destino exista
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Converte listas para arrays numpy
    reais = np.array(reais)
    predicoes = np.array(predicoes)
    
    # 1. Cálculos de Métricas Globais
    acuracia = accuracy_score(reais, predicoes)
    f1 = f1_score(reais, predicoes, average='macro')
    
    # 2. Matriz de Confusão e Métricas Clínicas (Sensibilidade e Especificidade)
    cm = confusion_matrix(reais, predicoes)
    
    # Para classificação binária (0: Benigno, 1: Maligno)
    if len(nomes_classes) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall da classe positiva (Maligno)
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Capacidade de identificar corretamente os Benignos
    else:
        sensibilidade = recall_score(reais, predicoes, average='macro')
        especificidade = "N/A (Requer classificação binária)"

    report_texto = classification_report(reais, predicoes, target_names=nomes_classes)

    # 3. Exibição formatada no terminal
    print("\n" + "#"*40)
    print("      RELATÓRIO DE MÉTRICAS GLOBAIS")
    print("#"*40)
    print(f"Acurácia Geral:   {acuracia:.4f}")
    print(f"Sensibilidade:    {sensibilidade:.4f} (Recall de Malignos)")
    print(f"Especificidade:   {especificidade:.4f} (Recall de Benignos)")
    print(f"F1-Score (Macro): {f1:.4f}")
    print("\nMatriz de Confusão:")
    print(cm)
    print("\nRelatório Detalhado por Classe:")
    print(report_texto)
    print("#"*40)

    # 4. Salvando os resultados em um arquivo de texto para uso no Artigo científico
    caminho_txt = os.path.join(pasta_saida, "relatorio_final_metricas.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("#"*50 + "\n")
        f.write(f" RELATÓRIO COMPLETO DE VALIDAÇÃO CRUZADA - PASTA: {pasta_saida}\n")
        f.write("#"*50 + "\n\n")
        f.write(f"Acurácia Geral:   {acuracia:.4f}\n")
        if isinstance(especificidade, float):
            f.write(f"Sensibilidade (Recall): {sensibilidade:.4f}\n")
            f.write(f"Especificidade:         {especificidade:.4f}\n")
        f.write(f"F1-Score (Macro):       {f1:.4f}\n\n")
        f.write("MATRIZ DE CONFUSÃO:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("RELATÓRIO DETALHADO POR CLASSE:\n")
        f.write(report_texto)
        
    print(f"\n[Sucesso] Relatório técnico salvo em: '{caminho_txt}'")