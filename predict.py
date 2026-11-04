import json
import os
import sys
from datetime import datetime

import keras
import numpy as np
from keras import applications
from PIL import Image

CLASSES = {
    0: "pare",
    1: "siga",
    2: "velocidad",
    3: "zona escolar",
    4: "fondo",
}

MODELS = {
    "VGG16": applications.vgg16.preprocess_input,
}

IMG_SIZE = (224, 224)
EXPERIMENTO_DIR = os.path.join("results", "experimento_2")
RESULTADOS_DIR = "results"
TEST_IMG_DIR = "test_img"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SignalClassifier:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        for name in MODELS:
            model_path = os.path.join(EXPERIMENTO_DIR, name, f"{name}.keras")
            if not os.path.exists(model_path):
                print(f"[ADVERTENCIA] Modelo no encontrado: {model_path}")
                continue
            self.models[name] = keras.models.load_model(model_path)
            print(f"[INFO] Modelo cargado: {model_path}")

    def preprocess_image(self, image_path, preprocess_fn):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_fn(img_array)
        return img_array

    def predict_with_model(self, image_path, model_name):
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no esta cargado.")

        preprocess_fn = MODELS[model_name]
        model = self.models[model_name]

        img_array = self.preprocess_image(image_path, preprocess_fn)
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]

        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])

        return {
            "model": model_name,
            "class_idx": class_idx,
            "class_name": CLASSES.get(class_idx, f"desconocida_{class_idx}"),
            "confidence": confidence,
            "all_probabilities": {
                CLASSES.get(i, str(i)): float(p) for i, p in enumerate(probabilities)
            },
        }

    def evaluate_all(self, image_path, save_results=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        print(f"\nEvaluando imagen: {image_path}")
        print("-" * 60)
        print(f"{'Modelo':<15} | {'Clase':<18} | {'Confianza'}")
        print("-" * 60)

        results = []
        for model_name in self.models:
            try:
                res = self.predict_with_model(image_path, model_name)
                results.append(res)
                print(
                    f"{res['model']:<15} | {res['class_name']:<18} | {res['confidence']:.4f}"
                )
            except Exception as e:
                print(f"{model_name:<15} | ERROR: {e}")

        print("-" * 60)

        if save_results:
            self._save_results(image_path, results)

        return results

    def _save_results(self, image_path, results):
        os.makedirs(f"{RESULTADOS_DIR}/prediction", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}.json"
        save_path = os.path.join(f"{RESULTADOS_DIR}/prediction", filename)

        data = {
            "image": image_path,
            "timestamp": timestamp,
            "classes": CLASSES,
            "results": results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Resultados guardados en: {save_path}")

    def get_test_images(self):
        images = []
        if not os.path.isdir(TEST_IMG_DIR):
            return images
        for fname in sorted(os.listdir(TEST_IMG_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTENSIONS:
                images.append(os.path.join(TEST_IMG_DIR, fname))
        return images


def main():
    if len(sys.argv) < 2:
        classifier = SignalClassifier()
        images = classifier.get_test_images()
        if not images:
            print(f"No se encontraron imagenes en {TEST_IMG_DIR}/")
            print("Uso: python predict.py <ruta_imagen>")
            print("O coloca imagenes en test_img/")
            sys.exit(1)
        for image_path in images:
            classifier.evaluate_all(image_path)
        return

    image_path = sys.argv[1]
    classifier = SignalClassifier()
    classifier.evaluate_all(image_path)


if __name__ == "__main__":
    main()
