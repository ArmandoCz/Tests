import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Función para limpiar la memoria de la GPU
def limpiar_memoria_gpu():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Memoria de la GPU limpiada.")

limpiar_memoria_gpu()

# Verificar si hay GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")



model_type = "vit_b"
sam_checkpoint = r"SAM_checkpoint/sam_vit_b_01ec64.pth"  # Asegúrate de descargar este checkpoint desde el repositorio oficial de SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Crear un predictor a partir del modelo cargado
predictor = SamPredictor(sam)

image_path = r"images/img2.png"  # Reemplaza con la ruta a tu imagen
image = Image.open(image_path).convert("RGB")
image = image.resize((256, 256))  # Redimensionar la imagen a 256x256 píxeles

try:
    predictor.set_image(np.array(image))
except torch.cuda.OutOfMemoryError:
    print("Memoria de la GPU insuficiente, cambiando a CPU...")
    device = "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))

# Definir un punto de referencia para segmentar (x, y)
input_point = np.array([[215, 56]])  # Coordenadas del punto en la imagen
input_label = np.array([1])  # 1 indica que se quiere segmentar el objeto en ese punto

# Realizar la predicción de la máscara con torch.no_grad() para ahorrar memoria
with torch.no_grad():
    masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

# Mostrar la imagen original y la máscara segmentada
plt.figure(figsize=(10, 5))

# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagen original")
plt.axis("off")

# Máscara segmentada
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(masks[0], cmap="jet", alpha=0.5)  # Superponer la máscara con transparencia
plt.title("Máscara segmentada")
plt.axis("off")

plt.show()
