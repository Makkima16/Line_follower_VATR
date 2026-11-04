from trainer import CNNTrainer

nombre_exp = "experimento_2"

# =========================================================================
# HIPERPARÁMETROS DE ENTRENAMIENTO
# =========================================================================
hiperparametros = {
    "max_epochs": 15,  # Épocas de la fase 1 (solo la cabeza de clasificación)
    "patience": 5,  # Paciencia del Early Stopping (épocas sin mejora en val_loss)
    "batch_size": 32,  # Tamaño del lote (batch)
    "learning_rate": 1e-3,  # Tasa de aprendizaje (Learning Rate)
    "img_size": (224, 224),  # Tamaño de entrada para las CNNs (alto, ancho)
    "val_split": 0.2,  # Proporción de datos para validación (20%)
    "test_split": 0.1,  # Proporción de datos para prueba (10%)
    "augmentar": True,  # Data augmentation (giro, zoom, brillo, contraste) solo en entrenamiento
    "fine_tune_capas": 4,  # Fase 2: descongela las últimas N capas de VGG16 (bloque 5 = 4 capas)
    "epocas_fine_tune": 20,  # Épocas máximas de la fase 2
    "lr_fine_tune": 1e-5,  # Learning rate bajo para no destruir lo aprendido
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

# Solo VGG16 en esta rama (vgg16_focus).
modelos_a_entrenar = ["VGG16"]

trainer.entrenar_todos(modelos_a_entrenar)
