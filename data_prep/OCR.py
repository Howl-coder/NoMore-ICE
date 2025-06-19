import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
import cv2

# Configuración de Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Rutas
carpeta_principal = r"D:\DefenseGPT\data\raw\esp"
carpeta_salida = r"D:\DefenseGPT\data\raw\esp\texto_extraido"
os.makedirs(carpeta_salida, exist_ok=True)

# Iterar sobre subcarpetas
for subcarpeta in os.listdir(carpeta_principal):
    ruta_subcarpeta = os.path.join(carpeta_principal, subcarpeta)

    if os.path.isdir(ruta_subcarpeta):
        for archivo in os.listdir(ruta_subcarpeta):
            if archivo.lower().endswith(".pdf"):
                ruta_pdf = os.path.join(ruta_subcarpeta, archivo)
                print(f"Procesando: {ruta_pdf}")

                try:
                    # Convertir PDF a imágenes con buena calidad
                    paginas = convert_from_path(ruta_pdf, dpi=300)
                    texto = ""

                    for i, pagina in enumerate(paginas):
                        # PIL → NumPy → OpenCV (grises)
                        img_np = np.array(pagina)
                        gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                        # Suavizar y umbral automático (Otsu)
                        desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)
                        _, binaria = cv2.threshold(desenfocada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # Convertir de vuelta a PIL
                        img_bin_pil = Image.fromarray(binaria)

                        # OCR con idioma inglés
                        texto += f"\n\n--- Pagina {i + 1} ---\n\n"
                        texto += pytesseract.image_to_string(img_bin_pil, lang='spa')

                    # Guardar texto extraído
                    nombre_txt = f"{archivo.replace('.pdf', '')}.txt"
                    ruta_txt = os.path.join(carpeta_salida, nombre_txt)

                    with open(ruta_txt, "w", encoding="utf-8") as f:
                        f.write(texto)

                    print(f"✔ Guardado en: {ruta_txt}")

                except Exception as e:
                    print(f"❌ Error procesando {ruta_pdf}: {e}")
