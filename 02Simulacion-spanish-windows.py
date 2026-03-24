from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict
import os
from datetime import datetime
import torch
import time

torch.backends.cudnn.benchmark = True

MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/test2.mov"
OUTPUT_PATH = "output_con_metadatos.MOV"
JSON_PATH = "detecciones_baches.json"

CLASES_ESPAÑOL = {
    "longitudinal_crack": "grieta_longitudinal",
    "longitudinal_crack_wide": "grieta_longitudinal_ancha",
    "transverse_crack": "grieta_transversal",
    "transverse_crack_wide": "grieta_transversal_ancha",
    "alligator_crack": "grieta_piel_cocodrilo",
    "alligator_crack_sunken": "grieta_piel_cocodrilo_hundida",
    "pothole": "bache",
    "pothole_deep": "bache_profundo"
}

gps_lat_actual = 19.432600
gps_lon_actual = -99.133200

baches_detectados = {}
detecciones_json = []
UMBRAL_DISTANCIA_GPS = 0.00002


def dms_a_decimal(dms_str: str) -> Optional[float]:
    if not dms_str or not isinstance(dms_str, str):
        return None
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


def simular_gps(lat, lon, paso=0.00002):
    lat = lat + paso
    lon = lon + paso * 0.5
    return round(lat, 6), round(lon, 6)


def generar_enlace_google_maps(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"


def es_mismo_bache(lat1, lon1, lat2, lon2):
    diff_lat = abs(lat1 - lat2)
    diff_lon = abs(lon1 - lon2)
    return diff_lat < UMBRAL_DISTANCIA_GPS and diff_lon < UMBRAL_DISTANCIA_GPS


def extraer_metadatos(video_path: str) -> Optional[Dict]:
    try:
        if not os.path.exists(video_path):
            return None
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
            "nombre_archivo": os.path.basename(video_path)
        }
        return datos
    except:
        return None


def procesar_video():
    global gps_lat_actual, gps_lon_actual

    metadata = extraer_metadatos(VIDEO_PATH)

    if metadata:
        if metadata.get("gps_latitud") and metadata.get("gps_longitud"):
            lat_decimal = dms_a_decimal(metadata["gps_latitud"])
            lon_decimal = dms_a_decimal(metadata["gps_longitud"])
            if lat_decimal is not None and lon_decimal is not None:
                gps_lat_actual = lat_decimal
                gps_lon_actual = lon_decimal

    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        DEVICE = 0
    else:
        DEVICE = "cpu"

    model.to("cuda" if DEVICE == 0 else "cpu")

    half = DEVICE == 0

    cap = cv2.VideoCapture(VIDEO_PATH)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        (fps / 2),
        (width, height)
    )

    frame_num = 0

    results = model.track(
        source=VIDEO_PATH,
        conf=0.02,
        persist=True,
        stream=True,
        device=DEVICE,
        half=half,
        imgsz=640,
        classes=[0,1,2,3,4,5,6,7],
        verbose=False,
        vid_stride=1
    )

    tiempo_inicio = time.time()

    for r in results:

        frame_num += 1

        gps_lat_actual, gps_lon_actual = simular_gps(
            gps_lat_actual,
            gps_lon_actual
        )

        tiempo_seg = frame_num / fps

        frame = r.orig_img

        if r.boxes is not None and r.boxes.id is not None:

            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            for box, id_bache, cls, conf in zip(
                boxes, ids, clases, confs
            ):

                nombre_ing = model.names[cls]
                nombre_esp = CLASES_ESPAÑOL.get(nombre_ing, nombre_ing)

                bache_nuevo = True

                if id_bache in baches_detectados:
                    bache_nuevo = False
                else:
                    for _, info in baches_detectados.items():
                        if es_mismo_bache(
                            gps_lat_actual,
                            gps_lon_actual,
                            info["lat"],
                            info["lon"]
                        ):
                            bache_nuevo = False
                            break

                if bache_nuevo:

                    link = generar_enlace_google_maps(
                        gps_lat_actual,
                        gps_lon_actual
                    )

                    deteccion = {
                        "id": int(id_bache),
                        "clase": nombre_esp,
                        "confianza": round(conf, 3),
                        "lat": gps_lat_actual,
                        "lon": gps_lon_actual,
                        "google_maps": link,
                        "frame": frame_num,
                        "tiempo_segundos": round(tiempo_seg, 2),
                        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    detecciones_json.append(deteccion)

                    baches_detectados[id_bache] = {
                        "lat": gps_lat_actual,
                        "lon": gps_lon_actual
                    }

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

        total = len(baches_detectados)

        cv2.putText(
            frame,
            f"Baches: {total}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        out.write(frame)

        cv2.imshow("Deteccion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tiempo_fin = time.time()

    tiempo_total = tiempo_fin - tiempo_inicio
    fps_procesamiento = frame_num / tiempo_total
    duracion_video = total_frames / fps
    velocidad = duracion_video / tiempo_total
    tiempo_por_frame = tiempo_total / frame_num

    resumen = {
        "metadata_video": metadata if metadata else {},
        "total_baches": len(baches_detectados),
        "tiempo_total": round(tiempo_total,2),
        "fps_procesamiento": round(fps_procesamiento,2),
        "velocidad_vs_real": round(velocidad,2),
        "tiempo_por_frame": round(tiempo_por_frame,4),
        "detecciones": detecciones_json
    }

    with open(JSON_PATH,"w",encoding="utf-8") as f:
        json.dump(resumen,f,indent=4,ensure_ascii=False)

    print("Tiempo total:", tiempo_total)
    print("FPS:", fps_procesamiento)
    print("Velocidad:", velocidad)


if __name__ == "__main__":
    procesar_video()