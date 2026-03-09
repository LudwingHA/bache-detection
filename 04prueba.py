from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict

# ================================
# CONFIGURACIONES INICIALES
# ================================
MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/prueba1.MOV"
OUTPUT_PATH = "output_con_metadatos.MOV"

# Set para almacenar IDs únicos de baches detectados
baches_detectados = set()


# ================================
# FUNCIÓN: DMS → DECIMAL
# ================================
def dms_a_decimal(dms_str: str) -> Optional[float]:
    """
    Convierte coordenadas en formato:
    19 deg 20' 48.84" N
    a formato decimal.
    """
    patron = r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])"
    match = re.match(patron, dms_str)

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
# FUNCIÓN PARA EXTRAER METADATOS
# ================================
def extraer_metadatos(video_path: str) -> Optional[Dict]:
    """
    Extrae metadatos usando ExifTool en formato JSON.
    """
    try:
        result = subprocess.run(
            ["exiftool", "-j", video_path],
            capture_output=True,
            text=True,
            check=True
        )

        metadata = json.loads(result.stdout)[0]

        datos_relevantes = {
            "fecha_grabacion": metadata.get("CreateDate"),
            "modelo_dispositivo": metadata.get("Model"),
            "gps_latitud": metadata.get("GPSLatitude"),
            "gps_longitud": metadata.get("GPSLongitude"),
            "gps_altitud": metadata.get("GPSAltitude"),
            "duracion": metadata.get("Duration"),
            "resolucion": f"{metadata.get('ImageWidth')}x{metadata.get('ImageHeight')}"
        }

        return datos_relevantes

    except subprocess.CalledProcessError:
        print("❌ Error ejecutando ExifTool. ¿Está instalado?")
        return None

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None


# ================================
# FUNCIÓN PRINCIPAL
# ================================
def procesar_video():

    # 🔎 1. EXTRAER METADATOS
    metadata = extraer_metadatos(VIDEO_PATH)

    print("\n========== METADATOS DEL VIDEO ==========")

    if metadata:
        for clave, valor in metadata.items():
            print(f"{clave}: {valor}")

        # 🔥 Convertir coordenadas a decimal
        if metadata.get("gps_latitud") and metadata.get("gps_longitud"):

            lat_decimal = dms_a_decimal(metadata["gps_latitud"])
            lon_decimal = dms_a_decimal(metadata["gps_longitud"])

            if lat_decimal is not None and lon_decimal is not None:
                print(f"\n📍 Latitud decimal: {lat_decimal}")
                print(f"📍 Longitud decimal: {lon_decimal}")
                print(f"🌍 Google Maps: https://www.google.com/maps?q={lat_decimal},{lon_decimal}")
            else:
                print("⚠ No se pudieron convertir las coordenadas.")
        else:
            print("No hay información GPS disponible.")

    else:
        print("No se encontraron metadatos.")

    print("=========================================\n")

    # 🔥 2. CARGAR MODELO YOLO
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # 🔥 Tracking con YOLO
    results = model.track(
        source=VIDEO_PATH,
        conf=0.03,
        persist=True,
        stream=True,
        classes=[0, 1]
    )

    for r in results:
        frame = r.orig_img

        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()

            for box, id_bache, cls in zip(boxes, ids, clases):
                baches_detectados.add(id_bache)

                x1, y1, x2, y2 = box
                nombre = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{id_bache} {nombre}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        total_baches = len(baches_detectados)

        cv2.rectangle(frame, (20, 20), (380, 100), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Baches Totales: {total_baches}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        out.write(frame)
        cv2.imshow("Conteo de Baches en Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n========== RESUMEN ==========")
    print(f"Se detectaron un total de {len(baches_detectados)} baches únicos.")
    print("=============================")


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    procesar_video()