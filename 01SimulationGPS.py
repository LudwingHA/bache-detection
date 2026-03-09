from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict

# ================================
# CONFIGURACIONES
# ================================

MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/test1.MOV"
OUTPUT_PATH = "output_con_metadatos.MOV"
JSON_PATH = "detecciones_baches.json"

gps_lat_actual = 19.432600
gps_lon_actual = -99.133200

baches_detectados = set()
detecciones_json = []


# ================================
# DMS → DECIMAL
# ================================
def dms_a_decimal(dms_str: str) -> Optional[float]:

    patron = r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])"
    match = re.match(patron, dms_str)

    if not match:
        return None

    grados = float(match.group(1))
    minutos = float(match.group(2))
    segundos = float(match.group(3))
    direccion = match.group(4)

    decimal = grados + minutos / 60 + segundos / 3600

    if direccion in ["S", "W"]:
        decimal *= -1

    return round(decimal, 6)


# ================================
# SIMULAR MOVIMIENTO DESDE GPS REAL
# ================================
def simular_gps(lat, lon, paso=0.00002):

    lat = lat + paso
    lon = lon + paso * 0.5

    return round(lat, 6), round(lon, 6)


# ================================
# EXTRAER METADATOS
# ================================
def extraer_metadatos(video_path: str) -> Optional[Dict]:

    try:

        result = subprocess.run(
            ["exiftool", "-j", video_path],
            capture_output=True,
            text=True,
            check=True
        )

        metadata = json.loads(result.stdout)[0]

        datos = {
            "fecha": metadata.get("CreateDate"),
            "modelo": metadata.get("Model"),
            "gps_latitud": metadata.get("GPSLatitude"),
            "gps_longitud": metadata.get("GPSLongitude"),
            "duracion": metadata.get("Duration"),
        }

        return datos

    except Exception as e:
        print("Error metadatos:", e)
        return None


# ================================
# PROCESAR VIDEO
# ================================
def procesar_video():

    global gps_lat_actual, gps_lon_actual

    # ========================
    # METADATOS
    # ========================

    metadata = extraer_metadatos(VIDEO_PATH)

    print("\n=== METADATOS ===")

    if metadata:

        for k, v in metadata.items():
            print(k, ":", v)

        if metadata.get("gps_latitud"):

            lat_decimal = dms_a_decimal(metadata["gps_latitud"])
            lon_decimal = dms_a_decimal(metadata["gps_longitud"])

            if lat_decimal is not None and lon_decimal is not None:

                gps_lat_actual = lat_decimal
                gps_lon_actual = lon_decimal

                print("GPS REAL:", gps_lat_actual, gps_lon_actual)

            else:
                print("No se pudo convertir GPS")

        else:
            print("Sin GPS → se usará simulación")

    else:
        print("Sin metadatos")

    print("=================\n")

    # ========================
    # MODELO YOLO
    # ========================

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error video")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_num = 0

    results = model.track(
        source=VIDEO_PATH,
        conf=0.03,
        persist=True,
        stream=True,
        classes=[0, 1]
    )

    # ========================
    # LOOP
    # ========================

    for r in results:

        frame_num += 1

        # mover gps desde origen real
        gps_lat_actual, gps_lon_actual = simular_gps(
            gps_lat_actual,
            gps_lon_actual,
            paso=0.00002
        )

        tiempo_seg = frame_num / fps

        frame = r.orig_img

        if r.boxes.id is not None:

            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()

            for box, id_bache, cls in zip(boxes, ids, clases):

                if id_bache not in baches_detectados:

                    deteccion = {
                        "id": int(id_bache),
                        "clase": model.names[cls],
                        "lat": gps_lat_actual,
                        "lon": gps_lon_actual,
                        "frame": frame_num,
                        "tiempo": round(tiempo_seg, 2)
                    }

                    detecciones_json.append(deteccion)

                baches_detectados.add(id_bache)

                x1, y1, x2, y2 = box

                nombre = model.names[cls]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID:{id_bache} {nombre}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        total = len(baches_detectados)

        cv2.rectangle(frame, (20, 20), (380, 100), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"Baches: {total}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        out.write(frame)

        cv2.imshow("Deteccion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ========================
    # GUARDAR JSON
    # ========================

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(detecciones_json, f, indent=4)

    print("\n==== RESUMEN ====")
    print("Baches:", len(baches_detectados))
    print("JSON:", JSON_PATH)
    print("Video:", OUTPUT_PATH)


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    procesar_video()