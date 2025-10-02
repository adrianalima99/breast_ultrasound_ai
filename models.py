import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações
torch.manual_seed(42)  # Para reprodutibilidade
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo utilizado: {device}")

# Configurações do dataset
DATA_DIR = "dataset"
CLASSES = ['benign', 'malignant', 'normal']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

class BreastUltrasoundDataset(Dataset):
    """Dataset customizado para imagens de ultrassonografia de mama"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = CLASSES
        self.num_classes = NUM_CLASSES
        
        # Coletar todos os arquivos de imagem
        self.image_files = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.png"):
                    self.image_files.append(str(img_file))
                    self.labels.append(class_idx)
        
        print(f"Total de imagens encontradas: {len(self.image_files)}")
        for i, class_name in enumerate(self.classes):
            count = self.labels.count(i)
            print(f"  {class_name}: {count} imagens")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Carregar imagem
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BreastUltrasoundClassifier:
    def __init__(self, data_dir, img_size=224, batch_size=16):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = CLASSES
        self.num_classes = NUM_CLASSES
        
        # Transformações para treinamento (com data augmentation)
        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal
            transforms.RandomRotation(degrees=10),    # Rotação leve
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformações para validação/teste (sem data augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model_dnn = None
        self.model_cnn = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def load_dataset(self):
        """Carrega o dataset e divide em treino/validação/teste"""
        print("Carregando dataset...")
        
        # Criar dataset completo
        full_dataset = BreastUltrasoundDataset(self.data_dir, transform=self.val_transforms)
        
        # Dividir dataset (70% treino, 15% validação, 15% teste)
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            range(total_size), [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Criar datasets separados
        train_dataset = BreastUltrasoundDataset(self.data_dir, transform=self.train_transforms)
        val_dataset = BreastUltrasoundDataset(self.data_dir, transform=self.val_transforms)
        test_dataset = BreastUltrasoundDataset(self.data_dir, transform=self.val_transforms)
        
        # Aplicar índices
        train_dataset.image_files = [full_dataset.image_files[i] for i in train_indices.indices]
        train_dataset.labels = [full_dataset.labels[i] for i in train_indices.indices]
        
        val_dataset.image_files = [full_dataset.image_files[i] for i in val_indices.indices]
        val_dataset.labels = [full_dataset.labels[i] for i in val_indices.indices]
        
        test_dataset.image_files = [full_dataset.image_files[i] for i in test_indices.indices]
        test_dataset.labels = [full_dataset.labels[i] for i in test_indices.indices]
        
        # Criar DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"Dataset dividido:")
        print(f"  Treino: {len(train_dataset)} imagens")
        print(f"  Validação: {len(val_dataset)} imagens")
        print(f"  Teste: {len(test_dataset)} imagens")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model_dnn(self):
        """Cria o modelo DNN (Deep Neural Network)"""
        print("Criando modelo DNN...")
        
        # Definir uma rede totalmente conectada (DNN)
        self.model_dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        self.model_dnn = self.model_dnn.to(device)
        
        print(f"Modelo DNN criado com {sum(p.numel() for p in self.model_dnn.parameters())} parâmetros")
        return self.model_dnn
    
    def create_model_cnn(self):
        """Cria o modelo CNN (Convolutional Neural Network)"""
        print("Criando modelo CNN...")
        
        import torchvision.models as models
        
        # Usar ResNet18 pré-treinado
        self.model_cnn = models.resnet18(pretrained=True)
        
        # Modificar a última camada para 3 classes
        num_features = self.model_cnn.fc.in_features
        self.model_cnn.fc = nn.Linear(num_features, self.num_classes)
        
        self.model_cnn = self.model_cnn.to(device)
        
        print(f"Modelo CNN criado com {sum(p.numel() for p in self.model_cnn.parameters())} parâmetros")
        return self.model_cnn
    
    def train_model(self, model_type="dnn", num_epochs=20, learning_rate=1e-4, patience=5, min_delta=0.001):
        """Treina o modelo"""
        if model_type == "dnn" and self.model_dnn is None:
            self.create_model_dnn()  # Cria o modelo DNN
            model = self.model_dnn
        elif model_type == "cnn" and self.model_cnn is None:
            self.create_model_cnn()  # Cria o modelo CNN
            model = self.model_cnn
        else:
            raise ValueError("Tipo de modelo inválido!")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nIniciando treinamento do modelo {model_type.upper()} por {num_epochs} épocas...")
        print(f"Early Stopping: paciência={patience}, min_delta={min_delta}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # Treinamento
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validação
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calcular métricas
            train_loss_avg = train_loss / len(self.train_loader)
            val_loss_avg = val_loss / len(self.val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"Época {epoch+1:2d}/{num_epochs}: "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early Stopping
            if val_loss_avg < best_val_loss - min_delta:
                best_val_loss = val_loss_avg
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"  → Novo melhor modelo! Val Loss: {val_loss_avg:.4f}")
            else:
                patience_counter += 1
                print(f"  → Paciência: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"\nEarly Stopping ativado na época {epoch+1}")
                    print(f"Melhor validação loss: {best_val_loss:.4f}")
                    break
        
        # Restaurar melhor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nMelhor modelo restaurado (Val Loss: {best_val_loss:.4f})")
        
        print("-" * 50)
        print(f"Treinamento do modelo {model_type.upper()} concluído!")
        
        # Plotar gráficos de treinamento
        self.plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, model_type="dnn"):
        """Plota o histórico de treinamento e salva os gráficos"""
        
        # Definir o caminho de salvamento dependendo do tipo de modelo
        save_dir = f'results/graphs/{model_type}'
        os.makedirs(save_dir, exist_ok=True)  # Cria o diretório se não existir
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(train_losses, label='Treino', color='blue')
        ax1.plot(val_losses, label='Validação', color='red')
        ax1.set_title('Loss por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Acurácia
        ax2.plot(train_accs, label='Treino', color='blue')
        ax2.plot(val_accs, label='Validação', color='red')
        ax2.set_title('Acurácia por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()

        # Salvar os gráficos no diretório correspondente
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, model_type="dnn"):
        """Avalia o modelo no conjunto de teste"""
        if model_type == "dnn":
            model = self.model_dnn
        elif model_type == "cnn":
            model = self.model_cnn
        else:
            raise ValueError("Tipo de modelo inválido!")

        if model is None:
            raise ValueError(f"Modelo {model_type.upper()} não foi treinado ainda!")
        
        print(f"\nAvaliando o modelo {model_type.upper()} no conjunto de teste...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        print(f"\nMétricas no conjunto de teste para o modelo {model_type.upper()}:")
        print(f"  Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Métricas detalhadas por classe
        self.detailed_class_metrics(all_labels, all_predictions)
        
        # Matriz de confusão
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm)
        
        # Análise de erros
        self.error_analysis(all_labels, all_predictions)
        
        return accuracy, precision, recall, f1, cm

    def detailed_class_metrics(self, all_labels, all_predictions):
        """Calcula métricas detalhadas por classe"""
        print("\n" + "="*60)
        print("MÉTRICAS DETALHADAS POR CLASSE")
        print("="*60)
        
        # Métricas por classe
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        print(f"{'Classe':<12} {'Precisão':<10} {'Recall':<10} {'F1-Score':<10} {'Suporte':<10}")
        print("-" * 60)
        
        for i, class_name in enumerate(self.classes):
            print(f"{class_name:<12} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f} {support[i]:<10}")
        
        # Relatório de classificação completo
        print("\n" + "="*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO COMPLETO")
        print("="*60)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.classes, 
                                     digits=4)
        print(report)
        
        # Salvar relatório em arquivo
        with open('results/graphs/classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
            f.write("="*60 + "\n")
            f.write(report)
        
        print("\nRelatório detalhado salvo em: results/graphs/classification_report.txt")

    def error_analysis(self, all_labels, all_predictions):
        """Analisa os erros do modelo"""
        print("\n" + "="*60)
        print("ANÁLISE DE ERROS")
        print("="*60)
        
        # Calcular taxa de erro por classe
        cm = confusion_matrix(all_labels, all_predictions)
        class_errors = {}
        
        for i, class_name in enumerate(self.classes):
            total_samples = cm[i].sum()
            correct_predictions = cm[i, i]
            errors = total_samples - correct_predictions
            error_rate = errors / total_samples if total_samples > 0 else 0
            
            class_errors[class_name] = {
                'total': total_samples,
                'errors': errors,
                'error_rate': error_rate
            }
            
            print(f"{class_name}: {errors}/{total_samples} erros ({error_rate:.2%})")
        
        # Identificar confusões mais comuns
        print("\nCONFUSÕES MAIS COMUNS:")
        print("-" * 40)
        
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if i != j and cm[i, j] > 0:
                    print(f"{self.classes[i]} → {self.classes[j]}: {cm[i, j]} casos")
        
        # Salvar análise de erros
        with open('results/graphs/error_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("ANÁLISE DE ERROS\n")
            f.write("="*60 + "\n")
            for class_name, stats in class_errors.items():
                f.write(f"{class_name}: {stats['errors']}/{stats['total']} erros ({stats['error_rate']:.2%})\n")
        
        print("\nAnálise de erros salva em: results/graphs/error_analysis.txt")
    
    def plot_confusion_matrix(self, cm, model_type="dnn"):
        """Plota a matriz de confusão e salva a imagem"""
        
        # Definir o caminho de salvamento dependendo do tipo de modelo
        save_dir = f'results/graphs/{model_type}'
        os.makedirs(save_dir, exist_ok=True)  # Cria o diretório se não existir
        
        # Plotando a matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Matriz de Confusão')
        plt.xlabel('Predição')
        plt.ylabel('Verdadeiro')
        
        # Salvar a matriz de confusão no diretório correspondente
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Função principal"""
    print("=== Classificador de Ultrassonografia de Mama ===")
    print("Usando PyTorch")
    print("=" * 50)
    
    # Criar classificador
    classifier = BreastUltrasoundClassifier(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Carregar dataset
    train_dataset, val_dataset, test_dataset = classifier.load_dataset()
    
    # Treinar o modelo DNN
    classifier.train_model(model_type="dnn", num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    
    # Avaliar o modelo DNN
    classifier.evaluate_model(model_type="dnn")
    
    # Treinar o modelo CNN
    classifier.train_model(model_type="cnn", num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    
    # Avaliar o modelo CNN
    classifier.evaluate_model(model_type="cnn")
    
    print("\nTreinamento e avaliação concluídos!")

if __name__ == "__main__":
    main()
