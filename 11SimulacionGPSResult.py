from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict
import os
from datetime import datetime
import torch

# ================================
# CONFIG
# ================================

MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/test2.MOV"
JSON_PATH = "detecciones_baches_m1.json"

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


# ================================
# GPU M1
# ================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Dispositivo:", DEVICE)


# ================================
# DMS → DECIMAL
# ================================

def dms_a_decimal(dms_str: str):

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
# GPS SIMULADO
# ================================

def simular_gps(lat, lon, paso=0.00002):

    lat += paso
    lon += paso * 0.5

    return round(lat, 6), round(lon, 6)


# ================================
# GOOGLE MAPS
# ================================

def generar_enlace_google_maps(lat, lon):

    return f"https://www.google.com/maps?q={lat},{lon}"


# ================================
# METADATOS
# ================================

def extraer_metadatos(video_path):

    try:

        result = subprocess.run(
            ["exiftool", "-j", video_path],
            capture_output=True,
            text=True,
            check=True
        )

        metadata = json.loads(result.stdout)[0]

        return {
            "fecha": metadata.get("CreateDate"),
            "modelo": metadata.get("Model"),
            "gps_latitud": metadata.get("GPSLatitude"),
            "gps_longitud": metadata.get("GPSLongitude"),
            "duracion": metadata.get("Duration"),
        }

    except:
        return None


# ================================
# PROCESAR VIDEO RAPIDO
# ================================

def procesar_video():

    global gps_lat_actual, gps_lon_actual

    print("\nINICIANDO\n")

    metadata = extraer_metadatos(VIDEO_PATH)

    if metadata:
        if metadata.get("gps_latitud") and metadata.get("gps_longitud"):

            lat = dms_a_decimal(metadata["gps_latitud"])
            lon = dms_a_decimal(metadata["gps_longitud"])

            if lat and lon:
                gps_lat_actual = lat
                gps_lon_actual = lon

    print("Cargando modelo...")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0

    results = model.track(
        source=VIDEO_PATH,
        device=DEVICE,
        conf=0.25,
        persist=True,
        stream=True,
        verbose=False
    )

    for r in results:

        frame_num += 1

        if frame_num % 50 == 0:
            print(
                f"Frame {frame_num}/{total_frames} | Detectados {len(baches_detectados)}"
            )

        gps_lat_actual, gps_lon_actual = simular_gps(
            gps_lat_actual,
            gps_lon_actual
        )

        tiempo_seg = frame_num / fps

        if r.boxes is None or r.boxes.id is None:
            continue

        ids = r.boxes.id.int().cpu().tolist()
        clases = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()

        for id_bache, cls, conf in zip(ids, clases, confs):

            if id_bache in baches_detectados:
                continue

            nombre_en = model.names[cls]
            nombre_es = CLASES_ESPAÑOL.get(nombre_en, nombre_en)

            link = generar_enlace_google_maps(
                gps_lat_actual,
                gps_lon_actual
            )

            deteccion = {

                "id": int(id_bache),
                "clase": nombre_es,
                "confianza": round(conf, 3),
                "lat": gps_lat_actual,
                "lon": gps_lon_actual,
                "google_maps": link,
                "frame": frame_num,
                "tiempo": round(tiempo_seg, 2),
                "fecha": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            detecciones_json.append(deteccion)

            baches_detectados[id_bache] = True

            print("Nuevo:", deteccion)

    cap.release()

    resumen = {

        "total": len(baches_detectados),
        "metadata": metadata,
        "detecciones": detecciones_json

    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:

        json.dump(
            resumen,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nJSON generado:", JSON_PATH)


# ================================
# MAIN
# ================================

if __name__ == "__main__":

    procesar_video()
