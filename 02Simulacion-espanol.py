from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict
import os
from datetime import datetime



MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/prueba1.mov"
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

# simulacion de gps
def simular_gps(lat, lon, paso=0.00002):
    lat = lat + paso
    lon = lon + paso * 0.5
    return round(lat, 6), round(lon, 6)

def generar_enlace_google_maps(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"

def es_mismo_bache(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    diff_lat = abs(lat1 - lat2)
    diff_lon = abs(lon1 - lon2)
    return diff_lat < UMBRAL_DISTANCIA_GPS and diff_lon < UMBRAL_DISTANCIA_GPS
def extraer_metadatos(video_path: str) -> Optional[Dict]:
    try:
        if not os.path.exists(video_path):
            print(f"Error: El archivo {video_path} no existe")
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

    except FileNotFoundError:
        print("Error: exiftool no está instalado")
        return None
    except Exception as e:
        print(f"Error al extraer metadatos: {e}")
        return None


def procesar_video():
    global gps_lat_actual, gps_lon_actual

    print("\n" + "="*50)
    print("INICIANDO PROCESAMIENTO DE VIDEO")
    print("="*50)

    # Metadatos
    metadata = extraer_metadatos(VIDEO_PATH)

    print("\n=== METADATOS DEL VIDEO ===")
    if metadata:
        for k, v in metadata.items():
            if v:
                print(f"{k}: {v}")

        if metadata.get("gps_latitud") and metadata.get("gps_longitud"):
            lat_decimal = dms_a_decimal(metadata["gps_latitud"])
            lon_decimal = dms_a_decimal(metadata["gps_longitud"])

            if lat_decimal is not None and lon_decimal is not None:
                gps_lat_actual = lat_decimal
                gps_lon_actual = lon_decimal
                print(f"\n✅ GPS REAL: {gps_lat_actual}, {gps_lon_actual}")
            else:
                print("\n⚠️ Usando GPS simulado")
        else:
            print("\n⚠️ Usando GPS simulado")
    else:
        print("\n⚠️ Usando GPS simulado")

    print("="*50 + "\n")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encuentra el modelo en {MODEL_PATH}")
        return

    print("Cargando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print("✅ Modelo cargado correctamente")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se puede abrir el video {VIDEO_PATH}")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video: {os.path.basename(VIDEO_PATH)}")
    print(f"📐 Resolución: {width}x{height}")
    print(f"🎞️ FPS: {fps}")
    print(f"📊 Total frames: {total_frames}")
    print(f"📁 Salida: {OUTPUT_PATH}\n")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        (fps/2),
        (width, height)
    )

    frame_num = 0


    results = model.track(
        source=VIDEO_PATH,
        conf=0.02,
        persist=True,
        stream=True,
        classes=[0, 1, 2, 3, 4, 5, 6, 7],
        verbose=False,
        vid_stride=2
        
    )

    print("Procesando video... (Presiona 'q' para detener)")

    for r in results:
        frame_num += 1

        if frame_num % 100 == 0:
            progreso = (frame_num / total_frames) * 100
            print(f"Progreso: {progreso:.1f}% - Detectados: {len(baches_detectados)}")

        gps_lat_actual, gps_lon_actual = simular_gps(
            gps_lat_actual,
            gps_lon_actual,
            paso=0.00002
        )

        tiempo_seg = frame_num / fps
        frame = r.orig_img

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()
            confidences = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [1.0] * len(ids)

            for box, id_bache, cls, conf in zip(boxes, ids, clases, confidences):
          
                nombre_ingles = model.names[cls]
                nombre_espanol = CLASES_ESPAÑOL.get(nombre_ingles, nombre_ingles)
                
                bache_nuevo = True
                
                if id_bache in baches_detectados:
                    bache_nuevo = False
                else:
                    for bache_id, bache_info in baches_detectados.items():
                        if es_mismo_bache(
                            gps_lat_actual, gps_lon_actual,
                            bache_info['lat'], bache_info['lon']
                        ):
                            baches_detectados[id_bache] = bache_info
                            bache_nuevo = False
                            break

                if bache_nuevo:
                    google_maps_link = generar_enlace_google_maps(gps_lat_actual, gps_lon_actual)
                    
                    deteccion = {
                        "id": int(id_bache),
                        "clase_original": nombre_ingles, 
                        "clase": nombre_espanol,  
                        "confianza": round(conf, 3),
                        "lat": gps_lat_actual,
                        "lon": gps_lon_actual,
                        "google_maps": google_maps_link,
                        "frame": frame_num,
                        "tiempo_segundos": round(tiempo_seg, 2),
                        "tiempo_formato": f"{int(tiempo_seg//60):02d}:{int(tiempo_seg%60):02d}",
                        "fecha_deteccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    detecciones_json.append(deteccion)
                    
                    baches_detectados[id_bache] = {
                        'lat': gps_lat_actual,
                        'lon': gps_lon_actual,
                        'clase': nombre_espanol,
                        'frame': frame_num
                    }
                    
                    print(f"✅ Nuevo: ID {id_bache} - {nombre_espanol}")

             
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Detectado ({nombre_espanol})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 255, 0),
                    2
                )


        total = len(baches_detectados)
        cv2.rectangle(frame, (20, 20), (380, 100), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Baches: {total}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"GPS: {gps_lat_actual:.6f}, {gps_lon_actual:.6f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        out.write(frame)
        cv2.imshow("Deteccion de Baches", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹️ Procesamiento detenido")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    print("\n" + "="*50)
    print("GUARDANDO RESULTADOS")
    print("="*50)

    resumen = {
        "metadata_video": metadata if metadata else {},
        "total_baches_unicos": len(baches_detectados),
        "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_procesado": os.path.basename(VIDEO_PATH),
        "detecciones": detecciones_json
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    print(f"\n📊 JSON guardado: {JSON_PATH}")
    print(f"🎥 Video guardado: {OUTPUT_PATH}")
    print("\n=== RESUMEN ===")
    print(f"📍 Total baches únicos: {len(baches_detectados)}")
    
    if detecciones_json:
        print("\n📌 Primeros 5:")
        for i, bache in enumerate(detecciones_json[:5]):
            print(f"   {i+1}. ID {bache['id']}: {bache['clase']}")
    
    print("\n✅ Procesamiento completado!")


if __name__ == "__main__":
    try:
        procesar_video()
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()