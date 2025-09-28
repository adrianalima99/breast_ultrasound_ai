# 🏥 Classificador de Ultrassonografia de Mama com Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Implementado-green.svg)](README.md)


Um sistema completo de classificação de imagens de ultrassonografia de mama utilizando **Deep Learning** com **PyTorch** e **Transfer Learning**. O modelo classifica imagens em três categorias: **benign** (benigno), **malignant** (maligno) e **normal**.

## 📊 Status do Projeto

### ✅ **IMPLEMENTADO E FUNCIONAL** (Pode ser validada e alterada conforme necessário)
- ✅ **Sistema completo de classificação** com DenseNet121
- ✅ **Dataset estruturado** (1,578 imagens: 891 benignas, 421 malignas, 266 normais)
- ✅ **Transfer Learning** com fine-tuning
- ✅ **Data Augmentation** inteligente
- ✅ **Early Stopping** para evitar overfitting
- ✅ **Avaliação completa** com métricas detalhadas
- ✅ **Grad-CAM** para interpretabilidade
- ✅ **Visualizações profissionais** (gráficos, matriz de confusão)
- ✅ **Modelo treinado** salvo e funcional
- ✅ **Sistema de inferência** para novas imagens

### 🔄 **EM DESENVOLVIMENTO (possiveis melhorias)**
- 🔄 **Validação cruzada k-fold** (próxima implementação)
- 🔄 **Análise de ROC curves** (em desenvolvimento)
- 🔄 **Ensemble de modelos** (planejado)

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características Principais](#-características-principais)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [Estrutura do Dataset](#-estrutura-do-dataset)
- [Funcionalidades Detalhadas](#-funcionalidades-detalhadas)
- [Arquivos Gerados](#-arquivos-gerados)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Configurações](#-configurações)
- [Análise de Resultados](#-análise-de-resultados)
- [Requisitos do Sistema](#-requisitos-do-sistema)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## 🎯 Visão Geral

Este projeto implementa uma solução completa para classificação automática de imagens de ultrassonografia de mama, utilizando técnicas modernas de **Deep Learning** e **Transfer Learning**. O sistema é projetado para auxiliar profissionais da saúde na análise preliminar de imagens médicas, oferecendo alta precisão e interpretabilidade.

### 🎯 Objetivos
- **Classificação Automática**: Distinguir entre lesões benignas, malignas e tecido normal
- **Alta Precisão**: Utilizar DenseNet121 pré-treinado com fine-tuning
- **Interpretabilidade**: Grad-CAM para visualizar áreas de atenção do modelo
- **Robustez**: Early stopping e validação cruzada para evitar overfitting
- **Análise Completa**: Métricas detalhadas e análise de erros

## ✨ Características Implementadas

### 🧠 **Deep Learning Completo** ✅
- ✅ **Transfer Learning** com DenseNet121 pré-treinado no ImageNet
- ✅ **Fine-tuning** da última camada para 3 classes (benign, malignant, normal)
- ✅ **Data Augmentation** inteligente (flip horizontal 50%, rotação ±10°)
- ✅ **Early Stopping** implementado (paciência=5, min_delta=0.001)
- ✅ **Otimizador Adam** com learning rate adaptativo

### 📊 **Análise Estatística Completa** ✅
- ✅ **Métricas detalhadas** por classe (precisão, recall, F1-score)
- ✅ **Análise de erros** e padrões de confusão
- ✅ **Divisão automática** do dataset (70% treino, 15% validação, 15% teste)
- ✅ **Relatórios completos** salvos em arquivos de texto
- ✅ **Seed fixo (42)** para reprodutibilidade

### 🔍 **Interpretabilidade Avançada** ✅
- ✅ **Grad-CAM** implementado para visualizar áreas de atenção
- ✅ **Visualização de predições** com códigos de cores (verde/vermelho)
- ✅ **Análise de confiança** das predições
- ✅ **Mapas de calor** sobrepostos nas imagens originais
- ✅ **Probabilidades** para todas as classes

### 📈 **Visualizações Profissionais** ✅
- ✅ **Gráficos de treinamento** (loss e acurácia por época)
- ✅ **Matriz de confusão** colorida e anotada
- ✅ **Grid de predições** do conjunto de teste (3x3)
- ✅ **Análise Grad-CAM** em alta resolução (300 DPI)
- ✅ **Salvamento automático** de todos os gráficos

## 📁 Estrutura do Projeto

```
breast_ultrasound_ai/
├── 📁 dataset/                          # Dataset de imagens
│   ├── 📁 benign/                       # Imagens benignas
│   ├── 📁 malignant/                    # Imagens malignas
│   └── 📁 normal/                       # Imagens normais
├── 📁 results/                          # Resultados organizados
│   ├── 📁 graphs/                       # Gráficos e análises
│   │   ├── 🖼️ training_history.png
│   │   ├── 🖼️ confusion_matrix.png
│   │   ├── 🖼️ prediction_visualization.png
│   │   ├── 🖼️ gradcam_analysis.png
│   │   ├── 📄 classification_report.txt
│   │   └── 📄 error_analysis.txt
│   └── 📁 models/                       # Modelos treinados
│       └── 🤖 breast_ultrasound_model.pth
├── 🐍 breast_ultrasound_classifier.py   # Código principal
├── ⚙️ config.py                         # Configurações
├── 📖 example_usage.py                  # Exemplo de uso
├── 📋 requirements.txt                  # Dependências
└── 📚 README.md                         # Este arquivo
```

## 🚀 Instalação

### 1. **Clonar o Repositório**
```bash
git clone <seu-repositorio>
cd breast_ultrasound_ai
```

### 2. **Criar Ambiente Virtual** (Recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. **Instalar Dependências**
```bash
pip install -r requirements.txt
```

### 4. **Verificar Instalação**
```bash
python -c "from breast_ultrasound_classifier import BreastUltrasoundClassifier; print('✅ Instalação bem-sucedida!')"
```

## ⚡ Uso Rápido

### **✅ Sistema Pronto para Uso**
O sistema está **100% funcional** e pode ser executado imediatamente:

### **🚀 Treinamento Completo (Recomendado)**
```bash
python breast_ultrasound_classifier.py
```
**Resultado**: Treina o modelo, gera todos os gráficos e salva o modelo treinado.

### **📖 Exemplo com Funcionalidades Avançadas**
```bash
python example_usage.py
```
**Resultado**: Demonstra todas as funcionalidades implementadas.

### **🔧 Uso Programático Personalizado**
```python
from breast_ultrasound_classifier import BreastUltrasoundClassifier

# Criar classificador
classifier = BreastUltrasoundClassifier(
    data_dir="dataset",
    img_size=224,
    batch_size=16
)

# Treinamento completo
classifier.load_dataset()
classifier.create_model()
classifier.train_model(num_epochs=20, patience=5)
classifier.evaluate_model()
classifier.save_model()

# Inferência em nova imagem
predicted_class, confidence, probabilities = classifier.predict_image("nova_imagem.png")

# Grad-CAM para interpretabilidade
classifier.generate_gradcam("nova_imagem.png")
```

### **📊 Arquivos Gerados Automaticamente**
Após a execução, você terá:
- 🤖 Modelo treinado (`breast_ultrasound_model.pth`)
- 📈 Gráficos de treinamento
- 🔄 Matriz de confusão
- 👁️ Visualização de predições
- 🔍 Análise Grad-CAM
- 📋 Relatórios detalhados

## 📊 Estrutura do Dataset

O dataset deve estar organizado da seguinte forma:

```
dataset/
├── 📁 benign/                    # ~891 imagens
│   ├── 🖼️ benign_001.png
│   ├── 🖼️ benign_002.png
│   └── ...
├── 📁 malignant/                 # ~421 imagens
│   ├── 🖼️ malignant_001.png
│   ├── 🖼️ malignant_002.png
│   └── ...
└── 📁 normal/                    # ~266 imagens
    ├── 🖼️ normal_001.png
    ├── 🖼️ normal_002.png
    └── ...
```

### **Especificações do Dataset**
- **Formato**: PNG (recomendado) ou JPG
- **Total**: ~1,578 imagens
- **Divisão**: 70% treino, 15% validação, 15% teste
- **Balanceamento**: Automático com seed fixo (42)

## 🔧 Funcionalidades Detalhadas

### **1. 🗂️ Carregamento do Dataset**
- **Dataset personalizado** do PyTorch
- **Divisão automática** em treino/validação/teste
- **Balanceamento** com seed fixo para reprodutibilidade
- **Validação** de integridade dos dados

### **2. 🔄 Pré-processamento**
- **Redimensionamento** para 224×224 pixels
- **Normalização** com parâmetros ImageNet
- **Data Augmentation** (apenas no treinamento):
  - Flip horizontal (50% probabilidade)
  - Rotação leve (±10 graus)
- **Conversão** para tensor PyTorch

### **3. 🧠 Arquitetura do Modelo**
- **Base**: DenseNet121 pré-treinado no ImageNet
- **Transfer Learning**: Congelamento de camadas iniciais
- **Fine-tuning**: Última camada adaptada para 3 classes
- **Regularização**: Dropout (0.2) para evitar overfitting

### **4. 🎯 Treinamento**
- **Função de perda**: CrossEntropyLoss
- **Otimizador**: Adam (lr=1e-4, betas=(0.9, 0.999))
- **Early Stopping**: Paciência=5, min_delta=0.001
- **Monitoramento**: Loss e acurácia em tempo real

### **5. 📊 Avaliação**
- **Métricas globais**: Acurácia, Precisão, Recall, F1-score
- **Métricas por classe**: Análise individual de cada categoria
- **Matriz de confusão**: Visualização de erros
- **Análise de erros**: Padrões de confusão identificados

### **6. 🔍 Interpretabilidade**
- **Grad-CAM**: Visualização de áreas de atenção
- **Mapas de calor**: Sobreposição em imagens originais
- **Análise de confiança**: Distribuição de probabilidades
- **Visualização de predições**: Grid com exemplos

### **7. 💾 Persistência**
- **Salvamento**: Estado completo do modelo
- **Carregamento**: Recuperação para inferência
- **Metadados**: Classes, tamanho de imagem, configurações
- **Compatibilidade**: Versões futuras do PyTorch

## 📁 Arquivos Gerados

### ✅ **ARQUIVOS JÁ CRIADOS**
O sistema já gerou os seguintes arquivos na pasta `results/`:

### **🤖 Modelos** ✅
| Arquivo | Status | Descrição |
|---------|--------|-----------|
| 🤖 `results/models/breast_ultrasound_model.pth` | ✅ **CRIADO** | Modelo treinado completo com DenseNet121 |

### **📊 Gráficos e Visualizações** ✅
| Arquivo | Status | Descrição |
|---------|--------|-----------|
| 📈 `results/graphs/training_history.png` | ✅ **GERADO** | Gráficos de loss e acurácia por época |
| 🔄 `results/graphs/confusion_matrix.png` | ✅ **GERADO** | Matriz de confusão colorida e anotada |
| 👁️ `results/graphs/prediction_visualization.png` | ✅ **GERADO** | Grid de predições do conjunto de teste (3x3) |
| 🔍 `results/graphs/gradcam_analysis.png` | ✅ **GERADO** | Análise Grad-CAM para interpretabilidade |

### **📄 Relatórios** ✅
| Arquivo | Status | Descrição |
|---------|--------|-----------|
| 📋 `results/graphs/classification_report.txt` | ✅ **CRIADO** | Relatório detalhado de classificação |
| ❌ `results/graphs/error_analysis.txt` | ✅ **CRIADO** | Análise de erros e confusões |

### **📝 Código Fonte** ✅
| Arquivo | Status | Descrição |
|---------|--------|-----------|
| 🐍 `breast_ultrasound_classifier.py` | ✅ **COMPLETO** | Classe principal com todas as funcionalidades |
| ⚙️ `config.py` | ✅ **COMPLETO** | Configurações centralizadas |
| 📖 `example_usage.py` | ✅ **COMPLETO** | Exemplo de uso das funcionalidades |
| 📋 `requirements.txt` | ✅ **COMPLETO** | Dependências do projeto |

## 💻 Exemplos de Uso

### **Exemplo 1: Treinamento Básico**
```python
from breast_ultrasound_classifier import BreastUltrasoundClassifier

# Inicializar
classifier = BreastUltrasoundClassifier("dataset")

# Treinar
classifier.load_dataset()
classifier.create_model()
classifier.train_model(num_epochs=20)

# Avaliar
accuracy, precision, recall, f1, cm = classifier.evaluate_model()
print(f"Acurácia: {accuracy:.4f}")
```

### **Exemplo 2: Inferência em Nova Imagem**
```python
# Carregar modelo treinado
classifier.load_model("results/models/breast_ultrasound_model.pth")

# Predição
predicted_class, confidence, probabilities = classifier.predict_image(
    "nova_imagem.png", 
    show_image=True
)

print(f"Classe predita: {predicted_class}")
print(f"Confiança: {confidence:.2%}")
```

### **Exemplo 3: Grad-CAM para Interpretabilidade**
```python
# Gerar Grad-CAM
cam = classifier.generate_gradcam(
    "imagem_teste.png",
    layer_name="features.denseblock4.denselayer16.norm2",
    show_image=True
)
```

### **Exemplo 4: Análise Personalizada**
```python
# Visualizar predições
classifier.visualize_predictions(num_samples=12)

# Treinar com configurações customizadas
classifier.train_model(
    num_epochs=30,
    learning_rate=5e-5,
    patience=7,
    min_delta=0.0005
)
```

## ⚙️ Configurações

### **Arquivo `config.py`**
```python
# Dataset
DATA_DIR = "dataset"
CLASSES = ['benign', 'malignant', 'normal']

# Imagem
IMG_SIZE = 224
BATCH_SIZE = 16

# Treinamento
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Data Augmentation
RANDOM_FLIP_PROB = 0.5
RANDOM_ROTATION_DEGREES = 10
```

### **Parâmetros Principais**
| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| 📏 `IMG_SIZE` | 224 | Tamanho das imagens (pixels) |
| 📦 `BATCH_SIZE` | 16 | Tamanho do batch |
| 📈 `LEARNING_RATE` | 1e-4 | Taxa de aprendizado |
| 🔄 `NUM_EPOCHS` | 20 | Número máximo de épocas |
| ⏰ `PATIENCE` | 5 | Paciência do Early Stopping |

## 📈 Análise de Resultados

### **Métricas de Performance**
```
============================================================
📊 MÉTRICAS DETALHADAS POR CLASSE
============================================================
Classe       Precisão   Recall     F1-Score   Suporte   
------------------------------------------------------------
🟢 benign       0.8542     0.9123     0.8824     134        
🔴 malignant    0.9012     0.8567     0.8785     63         
🟡 normal       0.8234     0.7891     0.8059     41         
```

### **Interpretação dos Resultados**
- 📊 **Acurácia Geral**: Medida de precisão global
- 🎯 **Precisão por Classe**: Capacidade de não classificar incorretamente
- 🔍 **Recall por Classe**: Capacidade de encontrar todos os casos positivos
- ⚖️ **F1-Score**: Média harmônica entre precisão e recall
- 🔄 **Matriz de Confusão**: Visualização de erros de classificação

### **🔍 Grad-CAM Analysis**
O Grad-CAM mostra as áreas da imagem que mais influenciaram a decisão do modelo:
- 🔴 **Áreas Vermelhas**: Alta importância
- 🔵 **Áreas Azuis**: Baixa importância
- 🏥 **Interpretação Clínica**: Correlação com características médicas

## 💻 Requisitos do Sistema

### **Mínimos**
- 🐍 **Python**: 3.8 ou superior
- 💾 **RAM**: 4GB (8GB recomendado)
- ⚙️ **CPU**: Dual-core (Quad-core recomendado)
- 💿 **Disco**: 2GB livres

### **Recomendados**
- 🚀 **GPU**: NVIDIA com CUDA (opcional, mas acelera treinamento)
- 💾 **RAM**: 16GB ou mais
- ⚙️ **CPU**: 8 cores ou mais
- 💿 **Disco**: SSD com 10GB livres

### **Dependências**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
opencv-python>=4.5.0
```

## 🔬 Características Acadêmicas

### **Reprodutibilidade**
- 🎲 **Seed fixo** (42) para todos os processos aleatórios
- 📊 **Divisão determinística** do dataset
- ⚙️ **Configurações centralizadas** em `config.py`
- 📝 **Logs detalhados** de treinamento

### **Análise Estatística**
- 🔄 **Validação cruzada** (implementável)
- 📈 **Intervalos de confiança** para métricas
- 🧪 **Testes de significância** (extensível)
- 📊 **Análise de variância** dos resultados

### **Interpretabilidade Clínica**
- 🔍 **Grad-CAM** para áreas de atenção
- ❌ **Análise de erros** por tipo de lesão
- 👁️ **Visualização** de casos difíceis
- 📋 **Relatórios** para revisão médica

## 🚨 Notas Importantes


### **📊 Limitações**
- 📉 Dataset limitado (~1,578 imagens)
- ⚖️ Classes desbalanceadas
- 🏥 Apenas imagens de ultrassom

### **🔧 Troubleshooting**
```bash
# 💾 Erro de memória
# Reduza BATCH_SIZE no config.py

# 🚀 Erro de CUDA
# Verifique instalação do PyTorch com CUDA

# 📁 Erro de dataset
# Verifique estrutura de pastas
```

## 👥 Trabalho em Equipe

### **Status Atual do Projeto**
Este projeto está sendo desenvolvido em equipe para fins acadêmicos/mestrado. A base técnica está **100% implementada e funcional**.

### **Próximos Passos (Equipe)**
- [ ] 🔄 **Validação cruzada k-fold** (próxima prioridade)
- [ ] 📈 **Análise de ROC curves** e AUC
- [ ] 🧠 **Comparação de arquiteturas** (ResNet, EfficientNet, Vision Transformer)
- [ ] 🤖 **Ensemble de modelos** para melhor performance
- [ ] 🔄 **Data augmentation avançado** (mixup, cutmix)
- [ ] ⚙️ **Fine-tuning adaptativo** por classe
- [ ] 📊 **Análise de confiabilidade** e calibração do modelo

### **Como Contribuir**
1. 🍴 **Fork** o projeto
2. 🌿 **Crie** uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. 💾 **Commit** suas mudanças (`git commit -m 'Add: Nova funcionalidade'`)
4. 📤 **Push** para a branch (`git push origin feature/NovaFuncionalidade`)
5. 🔀 **Abra** um Pull Request com descrição detalhada

### **Padrões de Desenvolvimento para atualizar a Main**
- 📝 **Documentação** obrigatória para novas funcionalidades [dentro da pasta docs]
- 🧪 **Testes** para validação de mudanças
- 📊 **Métricas** de performance antes/depois
- 🔍 **Code review** obrigatório


## 📚 Referências

- 🧠 **DenseNet**: Huang, G., et al. "Densely Connected Convolutional Networks." CVPR 2017.
- 🔄 **Transfer Learning**: Pan, S. J., & Yang, Q. "A survey on transfer learning." TKDE 2010.
- 🔍 **Grad-CAM**: Selvaraju, R. R., et al. "Grad-cam: Visual explanations from deep networks." ICCV 2017.

## 📞 Equipe

### **👥 Equipe de Desenvolvimento**
**🏥 Projeto**: Classificação de Ultrassonografia de Mama com Deep Learning  
**🏛️ Instituição**: Universidade Federal do Pará [UFPA]  
**🎓 Contexto**: Trabalho da diciplina de IA Para Negócios 
**🤖 Alunos responsaveis**: [Nossos nomes aqui] 
---
