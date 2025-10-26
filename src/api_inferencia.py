from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from inferencia_modelos import load_model, predict_image

# =====================================================
# CONFIGURACIÓN DE LA API
# =====================================================
app = FastAPI(title="API de Inferencia CIFAR-10",
              description="API para realizar predicciones con modelos preentrenados (VGG16 y ResNet50)",
              version="1.0")

# =====================================================
# CONFIGURACIÓN DEL MODELO (SE CARGA AL INICIAR)
# =====================================================

# Clases de CIFAR-10
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- Cargar modelo al iniciar el servidor ---
print("🔄 Cargando modelo entrenado...")
MODEL = load_model("vgg16", 10, r"C:\Users\prestamour\Downloads\Taller-transfer-learning\modelos\vgg16_cifar10.pth")
print("✅ Modelo cargado correctamente.")

# =====================================================
# ENDPOINT PRINCIPAL
# =====================================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la predicción de clase en formato JSON.
    Ejemplo de uso en Swagger: /docs
    """
    try: 
        # Leer imagen en memoria
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Guardar imagen
        image.save("API.jpg")

        # Inferencia
        prediction = predict_image("API.jpg", MODEL, cifar10_classes)

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# =====================================================
# ENDPOINT DE PRUEBA
# =====================================================
@app.get("/")
def root():
    return {"message": "API de inferencia CIFAR-10 activa. Usa /predict/ para clasificar imágenes."}
