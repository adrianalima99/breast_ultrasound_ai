"""
Exemplo de uso das funcionalidades avançadas do classificador
"""

from breast_ultrasound_classifier import BreastUltrasoundClassifier
from pathlib import Path

def main():
    print("=== Exemplo de Uso das Funcionalidades Avançadas ===")
    
    # Criar classificador
    classifier = BreastUltrasoundClassifier(
        data_dir="dataset",
        img_size=224,
        batch_size=16
    )
    
    # Carregar dataset
    print("\n1. Carregando dataset...")
    classifier.load_dataset()
    
    # Criar modelo
    print("\n2. Criando modelo...")
    classifier.create_model()
    
    # Treinar com Early Stopping
    print("\n3. Treinando modelo com Early Stopping...")
    classifier.train_model(
        num_epochs=20, 
        learning_rate=1e-4, 
        patience=5, 
        min_delta=0.001
    )
    
    # Avaliação completa
    print("\n4. Avaliando modelo...")
    classifier.evaluate_model()
    
    # Visualizar predições
    print("\n5. Visualizando predições...")
    classifier.visualize_predictions(num_samples=9)
    
    # Salvar modelo
    print("\n6. Salvando modelo...")
    classifier.save_model()
    
    # Exemplo de Grad-CAM
    print("\n7. Gerando Grad-CAM...")
    example_image = None
    for class_name in ['benign', 'malignant', 'normal']:
        class_dir = Path("dataset") / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            if images:
                example_image = images[0]
                break
    
    if example_image:
        print(f"Usando imagem: {example_image.name}")
        classifier.generate_gradcam(str(example_image))
    else:
        print("Nenhuma imagem encontrada para Grad-CAM")
    
    print("\n✅ Exemplo concluído!")
    print("Verifique a pasta 'results/' para todos os arquivos gerados.")

if __name__ == "__main__":
    main()
