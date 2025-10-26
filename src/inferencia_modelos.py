import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================================================
# FUNCIÓN 1: Cargar modelo y pesos entrenados
# =====================================================
def load_model(model_name, num_classes, weights_path):
    """
    Carga un modelo preentrenado (VGG16 o ResNet50),
    reemplaza la capa final y carga los pesos entrenados.

    Parámetros:
        model_name (str): "vgg16" o "resnet50"
        num_classes (int): número de clases del dataset
        weights_path (str): ruta al archivo .pth con los pesos entrenados

    Retorna:
        model (torch.nn.Module): modelo listo para inferencia
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.lower() == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Modelo no soportado. Usa 'vgg16' o 'resnet50'.")

    # Cargar pesos entrenados
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =====================================================
# FUNCIÓN 2: Predicción sobre una imagen
# =====================================================
def predict_image(image_path, model, class_names=None):
    """
    Realiza una predicción de clase sobre una imagen.

    Parámetros:
        image_path (str): ruta de la imagen a clasificar
        model (torch.nn.Module): modelo cargado con load_model()
        class_names (list[str], opcional): nombres de clases para decodificar la salida

    Retorna:
        pred_class (str o int): clase predicha
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformaciones consistentes con el entrenamiento
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Abrir imagen
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    pred_idx = predicted.item()

    if class_names:
        return class_names[pred_idx]
    else:
        return pred_idx

# =====================================================
# EJEMPLO DE USO
# =====================================================
if __name__ == "__main__":
    # CIFAR-10 tiene 10 clases estándar
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # Ejemplo: cargar el modelo entrenado VGG16
    model = load_model(
        model_name="vgg16",
        num_classes=10,
        weights_path="modelos/vgg16_cifar10.pth"
    )

    # Hacer una predicción sobre una imagen
    result = predict_image(r"C:\Users\prestamour\Downloads\Taller-transfer-learning\notebooks\Golden.jpeg", model, cifar10_classes)
    print("Predicción:", result)

    # Ejemplo: cargar el modelo entrenado RESNET50
    model = load_model(
        model_name="resnet50",
        num_classes=10,
        weights_path="modelos/resnet50_cifar10.pth"
    )

    # Hacer una predicción sobre una imagen
    result = predict_image(r"C:\Users\prestamour\Downloads\Taller-transfer-learning\notebooks\Golden.jpeg", model, cifar10_classes)
    print("Predicción:", result)
