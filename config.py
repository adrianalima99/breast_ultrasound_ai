"""
Configurações do Projeto de Classificação de Ultrassonografia de Mama
"""

# Configurações do Dataset
DATA_DIR = "dataset"
CLASSES = ['benign', 'malignant', 'normal']
NUM_CLASSES = len(CLASSES)

# Configurações de Imagem
IMG_SIZE = 224
BATCH_SIZE = 16

# Configurações de Treinamento
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Configurações de Data Augmentation
RANDOM_FLIP_PROB = 0.5
RANDOM_ROTATION_DEGREES = 10

# Configurações de Normalização (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Configurações de Divisão do Dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Configurações de Reprodutibilidade
RANDOM_SEED = 42

# Configurações de Output
RESULTS_DIR = "results"
GRAPHS_DIR = "results/graphs"
MODELS_DIR = "results/models"

# Configurações de Visualização
DPI = 300
FIGURE_SIZE = (12, 8)
GRADCAM_LAYER = 'features.denseblock4.denselayer16.norm2'
VISUALIZATION_SAMPLES = 9
