import cv2
import pytesseract
from PIL import Image
import re
import time

VIDEO_PATH = "./videos-m/prueba4.mp4"
LOG_PATH = "coordenadas_log.txt"

# Si estás en Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================================
# DMS → DECIMAL
# ================================
def dms_a_decimal(dms_str):

    patron = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
    match = re.search(patron, dms_str)

    if not match:
        return None

    grados = float(match.group(1))
    minutos = float(match.group(2))
    segundos = float(match.group(3))
    direccion = match.group(4)

    decimal = grados + (minutos / 60) + (segundos / 3600)

    if direccion in ["S", "W"]:
        decimal *= -1

    return round(decimal, 6)

# ================================
# EXTRAER LAT Y LON
# ================================
def extraer_lat_lon(texto):

    patron = r"\d+°\d+'\d+\.\d+\"?[NSEW]"
    coincidencias = re.findall(patron, texto)

    if len(coincidencias) >= 2:
        return coincidencias[0], coincidencias[1]

    return None, None

# ================================
# OCR MEJORADO
# ================================
def extraer_texto(frame):

    region = frame[0:200, 0:900]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    img_pil = Image.fromarray(gray)
    texto = pytesseract.image_to_string(img_pil)

    return texto

# ================================
# PROCESAR VIDEO
# ================================

def procesar_video():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error al abrir video")
        return

    # Crear / limpiar log
    with open(LOG_PATH, "w") as f:
        f.write("===== LOG DE COORDENADAS =====\n")

    print("🔍 Guardando coordenadas cada segundo (modo tiempo real)...\n")

    ultimo_guardado = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tiempo_actual = time.time()

        # 🔥 Guardar cada 1 segundo REAL
        if tiempo_actual - ultimo_guardado >= 1:

            texto = extraer_texto(frame)
            lat_dms, lon_dms = extraer_lat_lon(texto)

            if lat_dms and lon_dms:

                lat_decimal = dms_a_decimal(lat_dms)
                lon_decimal = dms_a_decimal(lon_dms)

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                linea = f"{timestamp} | {lat_decimal}, {lon_decimal}\n"

                with open(LOG_PATH, "a") as f:
                    f.write(linea)

                print("📍", linea.strip())

            ultimo_guardado = tiempo_actual

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Finalizado")
    print(f"📁 Log guardado en: {LOG_PATH}")
if __name__ == "__main__":
    procesar_video()