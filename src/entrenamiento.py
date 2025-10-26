import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# =====================================================
# CONFIGURACIÓN
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

os.makedirs("modelos", exist_ok=True)


# =====================================================
# TRANSFORMACIONES
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # requerido por VGG y ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================
# CARGA DE CIFAR-10 (SOLO 1K IMÁGENES)
# =====================================================
train_dataset_full = datasets.CIFAR10(root="./data", train=True,
                                      download=True, transform=transform)
subset_indices = list(range(1000))  # solo 1k imágenes
train_dataset = Subset(train_dataset_full, subset_indices)

test_dataset = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# FUNCIÓN PARA CONSTRUIR MODELO
# =====================================================
def build_model(model_name="vgg16", num_classes=10):
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Congelar capas convolucionales (solo entrenar la parte final)
        for param in model.features.parameters():
            param.requires_grad = False
        # Reemplazar la capa final del clasificador
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Congelar todas las capas excepto la última (fc)
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        # Reemplazar la capa final
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError("Modelo no soportado")

    return model.to(DEVICE)

# =====================================================
# FUNCIÓN DE ENTRENAMIENTO
# =====================================================
def train_model(model, optimizer, train_loader, test_loader, epochs=5, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc, f1 = evaluate_model(model, test_loader)
        print(f"Época [{epoch+1}/{epochs}] | Pérdida: {avg_loss:.4f} | Accuracy: {acc:.2f}% | F1: {f1:.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Modelo guardado en: {save_path}\n")

# =====================================================
# FUNCIÓN DE EVALUACIÓN (Accuracy + F1-score)
# =====================================================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

# =====================================================
# ENTRENAR VGG16
# =====================================================
print("\n===== ENTRENANDO VGG16 =====\n")
vgg16_model = build_model("vgg16", NUM_CLASSES)
optimizer_vgg = optim.Adam(filter(lambda p: p.requires_grad, vgg16_model.parameters()), lr=LR)

train_model(vgg16_model, optimizer_vgg, train_loader, test_loader,
            epochs=EPOCHS, save_path="modelos/vgg16_cifar10.pth")


# =====================================================
# ENTRENAR RESNET50
# =====================================================
print("\n===== ENTRENANDO RESNET50 =====\n")
resnet50_model = build_model("resnet50", NUM_CLASSES)
optimizer_resnet = optim.Adam(filter(lambda p: p.requires_grad, resnet50_model.parameters()), lr=LR)

train_model(resnet50_model, optimizer_resnet, train_loader, test_loader,

            epochs=EPOCHS, save_path="modelos/resnet50_cifar10.pth")
