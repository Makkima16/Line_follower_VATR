from trainer import CNNTrainer

nombre_exp = "experimento_1"

# =========================================================================
# HIPERPARÁMETROS DE ENTRENAMIENTO
# =========================================================================
hiperparametros = {
    "max_epochs": 10,  # Cantidad máxima de épocas permitidas
    "patience": 5,  # Paciencia del Early Stopping (épocas sin mejora en val_loss)
    "batch_size": 32,  # Tamaño del lote (batch)
    "learning_rate": 1e-3,  # Tasa de aprendizaje (Learning Rate)
    "img_size": (224, 224),  # Tamaño de entrada para las CNNs (alto, ancho)
    "val_split": 0.2,  # Proporción de datos para validación (20%)
    "test_split": 0.1,  # Proporción de datos para prueba (10%)
    "seed": 42,  # Semilla para reproducibilidad
}

# Instanciamos el entrenador pasando la carpeta del dataset, el experimento y los hiperparámetros.
# Los resultados se guardarán en: resultados/<experimento>/<nombremodelo>/
trainer = CNNTrainer(
    dataset_dir="dataset",  # Contiene las carpetas de clases: 0, 1, 2, ...
    output_dir="results",  # Carpeta raíz de resultados
    experimento=nombre_exp,  # Nombre del experimento
    **hiperparametros,
)

# Lista de CNNs a entrenar. Opciones disponibles:
# "VGG16", "VGG19", "MobileNet", "ResNet50", "DenseNet121"
# Si dejas la lista vacía [] o None, entrenará todas las disponibles.
modelos_a_entrenar = [
    "VGG16",
    "MobileNet",
    "VGG19",
    "ResNet50",
    "DenseNet121",
]

trainer.entrenar_todos(modelos_a_entrenar)
