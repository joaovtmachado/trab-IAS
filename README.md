# Classificação de Câncer de Mama em Histopatologia (BreakHis)

Projeto final da disciplina **Inteligência Computacional em Saúde** (UFES, 2026/1).

Classificação binária de imagens histopatológicas de mama em **benigno** vs **maligno**
usando *transfer learning* com redes convolutivas pré-treinadas no ImageNet
(ResNet50, VGG16 e MobileNetV2).

## Problema e motivação

O câncer de mama é um dos tipos mais comuns entre mulheres, e o diagnóstico definitivo
depende da análise histopatológica de lâminas por um patologista, uma tarefa demorada e
sujeita à variabilidade entre observadores. Modelos de visão computacional podem atuar
como ferramenta de apoio à decisão, triando lâminas e destacando casos suspeitos.

O foco deste trabalho **não** é apenas rodar um classificador, mas fazê-lo com uma
metodologia clinicamente honesta: evitar vazamento de dados entre pacientes, tratar o
desbalanceamento de classes e reportar métricas relevantes para a área de saúde
(sensibilidade, especificidade e AUC), além da acurácia.

## Dataset

[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
(Breast Cancer Histopathological Database) 7.909 imagens de 82 pacientes, em quatro
magnificações (40X, 100X, 200X, 400X), rotuladas como benigno ou maligno.

Baixe o dataset e aponte a constante `DATA_DIR` em `train.py` para a raiz descompactada
(padrão: `./BreaKHis_v1`). O carregamento (`dataset.py`) percorre a pasta recursivamente,
então a estrutura interna original do BreaKHis é aceita sem modificações. O identificador
do paciente é extraído do nome do arquivo (ex.: `SOB_M_DC-14-2523-40-001.png` →
paciente `SOB_M_DC-14-2523`).

## Metodologia

- **Split por paciente (anti-vazamento):** como cada paciente tem várias imagens (4
  magnificações × múltiplos campos), um *split* aleatório por imagem colocaria o mesmo
  paciente em treino e teste, inflando as métricas. Separamos primeiro **15% dos
  pacientes** como conjunto de teste isolado e, no restante, aplicamos
  `StratifiedGroupKFold` (5 folds) agrupando por paciente e estratificando por classe.
  Assim, nenhum paciente aparece em mais de uma partição.
- **Transfer learning:** backbones pré-treinados no ImageNet com as camadas convolutivas
  congeladas; apenas o classificador final é treinado para 2 classes.
- **Desbalanceamento:** o dataset tem ~31% benigno / ~69% maligno. Usamos pesos de classe
  (inversamente proporcionais à frequência) na `CrossEntropyLoss`.
- **Regularização:** *data augmentation* no treino (flips e rotação, válidos porque não
  há orientação canônica em lâminas) e *early stopping* pela perda de validação.
- **Avaliação:** métricas por fold com **média ± desvio padrão** (acurácia, sensibilidade,
  especificidade, F1 e AUC-ROC), matriz de confusão e curva ROC, além de uma **análise por
  magnificação**. O modelo final é avaliado uma única vez no conjunto de teste isolado,
  via *ensemble* (média das probabilidades dos 5 folds).

## Estrutura do repositório

```
dataset.py                    Dataset PyTorch do BreaKHis (rótulo, magnificação, paciente)
model.py / vggnet.py / mobilenet.py   Wrappers de transfer learning
train.py                      Pipeline: split por paciente, CV, treino, avaliação
cross_validation_metrics.py   Métricas, figuras (PNG) e relatórios
requirements.txt
resultados_kfold_<modelo>/    Métricas da validação cruzada
resultados_teste_<modelo>/    Métricas no conjunto de teste isolado
```

## Como reproduzir

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# baixe o BreaKHis e ajuste DATA_DIR em train.py, se necessário
# escolha o modelo editando MODELO_SELECIONADO em train.py: 'resnet50', 'vgg16' ou 'mobilenet'
python train.py
```

Os resultados (relatório `.txt`, matriz de confusão e curva ROC em PNG) são gravados em
`resultados_kfold_<modelo>/` e `resultados_teste_<modelo>/`. A seed é fixada
(`SEED = 42`) para reprodutibilidade.

## Resultados

Validação cruzada (5 folds, split por paciente), média ± desvio padrão:

| Modelo      | Acurácia | Sensibilidade | Especificidade | F1 (macro) | AUC |
|-------------|----------|---------------|----------------|------------|-----|
| ResNet50    | _a preencher_ | _ | _ | _ | _ |
| VGG16       | _a preencher_ | _ | _ | _ | _ |
| MobileNetV2 | _a preencher_ | _ | _ | _ | _ |

> Observação metodológica: com o *split* por paciente, espera-se que a acurácia fique
> **abaixo** do resultado obtido anteriormente com *split* por imagem (0,876), que era
> otimista por vazamento de dados. Esse é o comportamento correto e esperado.

## Vídeo de apresentação

_Link a preencher._

## Autores

Yan Castello Simmer e João Vitor Toneto Machado
