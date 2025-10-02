# Breast Ultrasound Classifier

Este repositório contém um classificador de imagens de ultrassonografia de mama utilizando PyTorch. O modelo classifica imagens em três categorias: benigno, maligno e normal.

## Requisitos

Para rodar o código, você precisará do Python 3.x e das seguintes bibliotecas:

- torch
 
- torchvision
 
- matplotlib
 
- seaborn
 
- numpy
 
- pandas
 
- scikit-learn
 
- Pillow
 
Instale as dependências utilizando:
```bash
pip install -r requirements.txt
```

## Estrutura do Repositório

```bash
.
├── main.py                # Código principal
├── dataset/               # Diretório com o dataset (subdiretórios 'benign', 'malignant', 'normal')
├── results/               # Diretório de resultados (gráficos, relatórios, matrizes de confusão)
│   └── graphs/            # Gráficos gerados durante o treinamento
└── requirements.txt       # Lista de dependências do projeto
```

## Funcionalidade

Este código treina dois tipos de modelos para classificação de imagens de ultrassonografia de mama:

- Modelo DNN (Rede Neural Profunda)

- Modelo CNN (Rede Neural Convolucional, utilizando ResNet18 pré-treinada)



Além do treinamento, são realizadas as seguintes etapas:

1. Pré-processamento de dados:

    - As imagens são redimensionadas para o tamanho 224x224 pixels.

    - Aplicação de aumento de dados (Data Augmentation) durante o treinamento.

2. Treinamento:

    - O modelo é treinado usando os datasets de treino e validação.

    - Utilização de Early Stopping para evitar overfitting.

3. Avaliação:

    - O modelo é avaliado usando o conjunto de teste.

    - Cálculo das métricas de desempenho, como acurácia, precisão, recall, F1-score, e geração da matriz de confusão.

4. Análise de Erros:

    - Identificação e análise dos erros cometidos pelo modelo durante a classificação.


## Como Usar

1. Preparar o Dataset:

    - O dataset deve estar organizado em subdiretórios dentro do diretório dataset/, com três subpastas: benign/, malignant/ e normal/. As imagens devem estar no formato .png.

1. Rodar o Código:

    - Para rodar o código, execute o script principal main.py:
```bash
python main.py
```
O código irá carregar o dataset, treinar os dois modelos e avaliar o desempenho.


## Explicação do Código

### Dataset

O dataset é carregado e dividido em três partes: treinamento (70%), validação (15%) e teste (15%). As imagens são carregadas e transformadas utilizando o torchvision.transforms com aumento de dados para o treinamento.

### Modelos
#### Modelo DNN

O modelo DNN é uma rede neural totalmente conectada que toma como entrada uma imagem de tamanho 224x224x3 e a classifica nas três classes.

#### Modelo CNN

O modelo CNN utiliza a ResNet18, que é uma arquitetura de rede convolucional pré-treinada. A última camada é substituída para classificar em três classes.

### Treinamento e Avaliação

O treinamento é feito por um número configurável de épocas. Durante o treinamento, o Early Stopping é utilizado para interromper o treinamento caso o modelo não melhore na validação por um número definido de épocas.

Após o treinamento, o modelo é avaliado no conjunto de teste, e as métricas de desempenho são calculadas, incluindo a matriz de confusão.

### Resultados

Os resultados, como gráficos de loss e acurácia, são salvos no diretório results/graphs/. Relatórios de classificação e análise de erros também são gerados e salvos.

## Exemplo de Saída

Durante a execução, o terminal exibirá o progresso do treinamento:
```bash
Iniciando treinamento do modelo DNN por 20 épocas...
Early Stopping: paciência=5, min_delta=0.001
--------------------------------------------------
Época  1/20: Train Loss: 0.6891, Train Acc: 72.34% | Val Loss: 0.5564, Val Acc: 79.15%
  → Novo melhor modelo! Val Loss: 0.5564
...
Treinamento do modelo DNN concluído!

Avaliação do modelo DNN no conjunto de teste...
Métricas no conjunto de teste para o modelo DNN:
  Acurácia: 0.8123 (81.23%)
  Precisão: 0.8045
  Recall: 0.8132
  F1-Score: 0.8088
```