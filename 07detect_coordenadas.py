import cv2
import pytesseract
import re
import os
from datetime import datetime

# ---------------- CONFIGURACIÓN ---------------- #

video_path = "./videos-m/prueba5.mp4"  # ← CAMBIA AQUÍ TU VIDEO
log_file = "coordenadas_log.txt"

# Si estás en Windows y no detecta tesseract:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------------------------ #

# Abrir video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 segundo

frame_count = 0
ultima_coordenada = None

# Crear encabezado del log si no existe
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("===== LOG DE COORDENADAS =====\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar solo 1 frame por segundo
    if frame_count % frame_interval == 0:

        altura, ancho, _ = frame.shape

        # Recortar zona superior donde está el texto
        roi = frame[0:120, 0:ancho]

        # -------- PREPROCESAMIENTO -------- #
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (3, 3), 0)
        _, thresh = cv2.threshold(gris, 180, 255, cv2.THRESH_BINARY)

        # -------- OCR -------- #
        config = "--psm 6"
        texto = pytesseract.image_to_string(thresh, config=config)

        print("Texto detectado:")
        print(texto)

        # -------- REGEX NUEVO FORMATO -------- #
        patron = r"Lt:\s*([-+]?\d+\.\d+)\s*,\s*Lg:\s*([-+]?\d+\.\d+).*?Sp:\s*([\d\.]+)"

        match = re.search(patron, texto)

        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            velocidad = float(match.group(3))

            coordenada_actual = (lat, lon)

            if coordenada_actual != ultima_coordenada:

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                linea = f"{timestamp} | {lat}, {lon} | {velocidad} km/h\n"

                with open(log_file, "a") as f:
                    f.write(linea)

                print("Guardado:", linea)

                ultima_coordenada = coordenada_actual

    frame_count += 1

cap.release()
print("Proceso terminado correctamente.")