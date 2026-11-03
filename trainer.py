import csv
import os

import keras
import matplotlib
from keras import applications, layers, models

matplotlib.use("Agg")  # Backend sin interfaz gráfica para guardar figuras en disco
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

MODELOS = {
    "VGG16": (applications.VGG16, applications.vgg16.preprocess_input),
    "VGG19": (applications.VGG19, applications.vgg19.preprocess_input),
    "MobileNet": (applications.MobileNet, applications.mobilenet.preprocess_input),
    "ResNet50": (applications.ResNet50, applications.resnet50.preprocess_input),
    "DenseNet121": (applications.DenseNet121, applications.densenet.preprocess_input),
}


class CNNTrainer:
    """Entrena CNN base de Keras (transfer learning) sobre un dataset en carpetas
    `dataset_dir/0`, `dataset_dir/1`, ... (una carpeta por clase)."""

    def __init__(
        self,
        dataset_dir,
        img_size=(224, 224),
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        patience=5,
        max_epochs=100,
        learning_rate=1e-3,
        seed=42,
        output_dir="resultados",
        experimento="experimento_1",
    ):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.patience = patience
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.output_dir = output_dir
        self.experimento = experimento
        self.exp_dir = os.path.join(self.output_dir, self.experimento)
        self.historiales = {}
        os.makedirs(self.exp_dir, exist_ok=True)
        self._cargar_datos()

    def _cargar_datos(self):
        if self.val_split <= 0 or self.test_split <= 0:
            raise ValueError("val_split y test_split deben ser mayores que 0.")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError(
                "La suma de val_split + test_split debe ser menor que 1.0."
            )

        class_names = sorted(
            [
                d
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
            ]
        )
        if not class_names:
            raise ValueError(
                f"No se encontraron carpetas de clase en '{self.dataset_dir}'."
            )

        extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        file_paths = []
        labels = []

        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.dataset_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in extensiones:
                        file_paths.append(os.path.join(root, file_name))
                        labels.append(idx)

        if not file_paths:
            raise ValueError(
                f"No se encontraron imágenes válidas en '{self.dataset_dir}'."
            )

        self.class_names = class_names
        self.num_classes = len(self.class_names)

        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            file_paths,
            labels,
            test_size=self.test_split,
            stratify=labels,
            random_state=self.seed,
        )

        val_rel = self.val_split / (1.0 - self.test_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_rel,
            stratify=train_val_labels,
            random_state=self.seed,
        )

        self.train_raw = self._crear_dataset(train_paths, train_labels, entrenar=True)
        self.val_raw = self._crear_dataset(val_paths, val_labels)
        self.test_raw = self._crear_dataset(test_paths, test_labels)

    def _crear_dataset(self, paths, labels, entrenar=False):
        path_ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        if entrenar:
            path_ds = path_ds.shuffle(
                len(paths), seed=self.seed, reshuffle_each_iteration=True
            )

        def _cargar(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32)
            return img, label

        ds = path_ds.map(_cargar, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(self.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    def _preparar(self, ds, preprocess, entrenar=False):
        ds = ds.map(
            lambda x, y: (preprocess(tf.cast(x, tf.float32)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if entrenar:
            ds = ds.shuffle(1000, seed=self.seed)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _construir(self, nombre):
        constructor, _ = MODELOS[nombre]
        base = constructor(
            weights="imagenet", include_top=False, input_shape=self.img_size + (3,)
        )
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        salida = layers.Dense(self.num_classes, activation="softmax")(x)
        modelo = models.Model(base.input, salida, name=nombre)
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return modelo

    def entrenar(self, nombre):
        """Entrena un modelo con early stopping sobre val_loss y devuelve un dict de resultados."""
        if nombre not in MODELOS:
            raise ValueError(f"Modelo '{nombre}' no soportado: {list(MODELOS)}")
        prep = MODELOS[nombre][1]
        train = self._preparar(self.train_raw, prep, entrenar=True)
        val = self._preparar(self.val_raw, prep)
        test = self._preparar(self.test_raw, prep)

        modelo = self._construir(nombre)
        # Para si val_loss no mejora (se estanca o sube) durante `patience` épocas
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )
        hist = modelo.fit(
            train,
            validation_data=val,
            epochs=self.max_epochs,
            callbacks=[early],
            verbose=2,
        )

        y_true = np.concatenate([y.numpy() for _, y in test])
        y_pred = np.argmax(modelo.predict(test, verbose=0), axis=1)
        epocas = len(hist.history["loss"])
        mejor = int(np.argmin(hist.history["val_loss"])) + 1

        # Carpeta específica para este modelo: resultados/experimento/nombremodelo
        model_dir = os.path.join(self.exp_dir, nombre)
        os.makedirs(model_dir, exist_ok=True)

        # Generar y guardar gráfica de pérdida y matriz de confusión
        grafica_perdida = self._graficar_perdida(nombre, hist.history, mejor, model_dir)
        grafica_cm = self._graficar_matriz_confusion(nombre, y_true, y_pred, model_dir)
        self.historiales[nombre] = hist.history

        # Generar y guardar classification report de sklearn
        reporte_str, reporte_dict = self._guardar_classification_report(
            nombre, y_true, y_pred, model_dir, epocas, mejor
        )

        # Guardar el modelo entrenado dentro de su propia carpeta
        modelo_path = os.path.join(model_dir, f"{nombre}.keras")
        modelo.save(modelo_path)
        print(f"Modelo guardado: {modelo_path}")

        etiquetas_indices = list(range(self.num_classes))
        return {
            "modelo": nombre,
            "experimento": self.experimento,
            "max_epochs": self.max_epochs,
            "epoca_parada": epocas,
            "mejor_epoca": mejor,
            "train_loss": hist.history["loss"][mejor - 1],
            "val_loss": hist.history["val_loss"][mejor - 1],
            "accuracy": reporte_dict.get("accuracy", 0.0),
            "f1_ponderado": f1_score(
                y_true,
                y_pred,
                labels=etiquetas_indices,
                average="weighted",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                labels=etiquetas_indices,
                average="macro",
                zero_division=0,
            ),
            "precision_macro": reporte_dict.get("macro avg", {}).get("precision", 0.0),
            "recall_macro": reporte_dict.get("macro avg", {}).get("recall", 0.0),
            "grafica_perdida": grafica_perdida,
            "grafica_confusion": grafica_cm,
            "archivo_reporte": f"reporte_clasificacion_{nombre}.txt",
        }

    def _guardar_classification_report(
        self, nombre, y_true, y_pred, model_dir, epocas, mejor_epoca
    ):
        """Calcula el classification_report de scikit-learn, lo muestra en consola y lo guarda en disco."""
        etiquetas_indices = list(range(self.num_classes))
        etiquetas_nombres = [str(c) for c in self.class_names]
        reporte_str = classification_report(
            y_true,
            y_pred,
            labels=etiquetas_indices,
            target_names=etiquetas_nombres,
            digits=4,
            zero_division=0,
        )
        reporte_dict = classification_report(
            y_true,
            y_pred,
            labels=etiquetas_indices,
            target_names=etiquetas_nombres,
            output_dict=True,
            zero_division=0,
        )

        print(f"\n--- Reporte de Clasificación ({nombre}) ---\n{reporte_str}")

        ruta_txt = os.path.join(model_dir, f"reporte_clasificacion_{nombre}.txt")
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"Experimento: {self.experimento}\n")
            f.write(f"Modelo: {nombre}\n")
            f.write(f"Épocas totales entrenadas: {epocas}\n")
            f.write(f"Mejor época seleccionada: {mejor_epoca}\n")
            f.write("=" * 60 + "\n\n")
            f.write(reporte_str)

        print(f"Reporte de clasificación guardado: {ruta_txt}")
        return reporte_str, reporte_dict

    def _graficar_perdida(self, nombre, history, mejor_epoca, model_dir):
        """Genera y guarda una gráfica de pérdida (train vs val) para el modelo."""
        train_loss = history["loss"]
        val_loss = history["val_loss"]
        epocas = list(range(1, len(train_loss) + 1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epocas,
            train_loss,
            label="Pérdida Entrenamiento (Train Loss)",
            color="#1f77b4",
            linewidth=2,
        )
        ax.plot(
            epocas,
            val_loss,
            label="Pérdida Validación (Val Loss)",
            color="#ff7f0e",
            linewidth=2,
        )

        # Resaltar la mejor época seleccionada por EarlyStopping
        ax.axvline(
            x=mejor_epoca,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mejor época ({mejor_epoca})",
        )
        ax.scatter(
            [mejor_epoca], [val_loss[mejor_epoca - 1]], color="red", s=50, zorder=5
        )

        ax.set_title(f"Curva de Pérdida - {nombre}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Épocas", fontsize=11)
        ax.set_ylabel("Pérdida (Cross-Entropy)", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        nombre_archivo = f"perdida_{nombre}.png"
        ruta_grafica = os.path.join(model_dir, nombre_archivo)
        plt.savefig(ruta_grafica, dpi=300)
        plt.close(fig)
        print(f"Gráfica de pérdida guardada: {ruta_grafica}")
        return nombre_archivo

    def _graficar_matriz_confusion(self, nombre, y_true, y_pred, model_dir):
        """Genera y guarda la matriz de confusión de las predicciones sobre el conjunto de test."""
        fig, ax = plt.subplots(figsize=(8, 7))
        etiquetas_indices = list(range(self.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=etiquetas_indices)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

        ax.set_title(f"Matriz de Confusión - {nombre}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Etiqueta Real", fontsize=11)
        plt.tight_layout()

        nombre_archivo = f"matriz_confusion_{nombre}.png"
        ruta_cm = os.path.join(model_dir, nombre_archivo)
        plt.savefig(ruta_cm, dpi=300)
        plt.close(fig)
        print(f"Matriz de confusión guardada: {ruta_cm}")
        return nombre_archivo

    def _graficar_comparativa(self, historiales):
        """Genera una gráfica comparativa con las pérdidas de validación de todos los modelos."""
        fig, ax = plt.subplots(figsize=(9, 5))
        for nombre, hist in historiales.items():
            val_loss = hist["val_loss"]
            epocas = list(range(1, len(val_loss) + 1))
            ax.plot(epocas, val_loss, label=f"{nombre}", linewidth=2)

        ax.set_title(
            f"Comparativa de Validación - Experimento: {self.experimento}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Épocas", fontsize=11)
        ax.set_ylabel("Pérdida de Validación (Val Loss)", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        ruta = os.path.join(self.exp_dir, "comparativa_val_loss.png")
        plt.savefig(ruta, dpi=300)
        plt.close(fig)
        print(f"Gráfica comparativa guardada: {ruta}")

    def entrenar_todos(self, nombres=None, csv_name="resultados.csv"):
        """Entrena los modelos indicados y guarda el CSV resumen en la carpeta del experimento."""
        resultados = []
        ruta = os.path.join(self.exp_dir, csv_name)
        modelos_seleccionados = nombres or list(MODELOS)

        for nombre in modelos_seleccionados:
            print(
                f"\n===== Entrenando {nombre} (Experimento: {self.experimento}) ====="
            )
            res = self.entrenar(nombre)
            resultados.append(res)
            keras.backend.clear_session()
            self._guardar_csv(resultados, ruta)  # se guarda tras cada modelo

        if len(self.historiales) > 1:
            self._graficar_comparativa(self.historiales)

        print(f"\nResumen del experimento guardado en: {ruta}")
        return resultados

    @staticmethod
    def _guardar_csv(resultados, ruta):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resultados[0]))
            w.writeheader()
            w.writerows(resultados)
