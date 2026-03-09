import cv2
import pytesseract
from PIL import Image
import re
import json
from ultralytics import YOLO
import numpy as np

# ================================
# CONFIGURACIÓN
# ================================
VIDEO_PATH = "./videos-m/PRUEBA10.mov"
MODEL_PATH = "./best.pt"
REPORT_NAME = "reporte_baches_geolocalizados.json"

# Inicializar YOLO
model = YOLO(MODEL_PATH)
reporte = {}

# ================================
# UTILIDADES DE OCR Y COORDENADAS
# ================================
def dms_a_decimal(dms_str):
    patron = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
    match = re.search(patron, dms_str)
    if not match: return None

    grados, minutos, segundos = float(match.group(1)), float(match.group(2)), float(match.group(3))
    direccion = match.group(4)
    decimal = grados + (minutos / 60) + (segundos / 3600)
    
    if direccion in ["S", "W"]: decimal *= -1
    return round(decimal, 6)

def extraer_coordenadas(frame):
    # Definir región de interés (ROI) para el OCR (ajusta según tu video)
    region = frame[0:150, 0:800] 
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    texto = pytesseract.image_to_string(Image.fromarray(binary))
    
    patron = r"\d+°\d+'\d+\.\d+\"?[NSEW]"
    coincidencias = re.findall(patron, texto)
    
    if len(coincidencias) >= 2:
        lat = dms_a_decimal(coincidencias[0])
        lon = dms_a_decimal(coincidencias[1])
        return lat, lon
    return None, None

# ================================
# PROCESO PRINCIPAL
# ================================
def procesar_sistema_integrado():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video")
        return

    print("🚀 Iniciando detección y geolocalización...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Extraer coordenadas del frame actual
        lat, lon = extraer_coordenadas(frame)

        # 2. Ejecutar detección de baches con Tracking
        results = model.track(frame, persist=True, conf=0.3, verbose=False)

        # 3. Si hay detecciones y tenemos coordenadas válidas
        if results[0].boxes.id is not None and lat is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            clases = results[0].boxes.cls.int().cpu().tolist()

            for b_id, cls in zip(ids, clases):
                # Solo guardamos el bache la primera vez que lo vemos (nuevo ID)
                if b_id not in reporte:
                    reporte[b_id] = {
                        "bache_id": b_id,
                        "tipo": model.names[cls],
                        "lat": lat,
                        "lon": lon,
                        "google_maps": f"https://www.google.com/maps?q={lat},{lon}"
                    }
                    print(f"📍 Bache detectado! ID: {b_id} en ({lat}, {lon})")

        # Visualización
        annotated_frame = results[0].plot()
        # Dibujar coordenadas actuales en pantalla para feedback
        cv2.putText(annotated_frame, f"GPS: {lat}, {lon}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Sistema de Deteccion Vial", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar resultados
    with open(REPORT_NAME, 'w') as f:
        json.dump(list(reporte.values()), f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Proceso finalizado. Se encontraron {len(reporte)} baches únicos.")
    print(f"📁 Reporte guardado en: {REPORT_NAME}")

if __name__ == "__main__":
    procesar_sistema_integrado()