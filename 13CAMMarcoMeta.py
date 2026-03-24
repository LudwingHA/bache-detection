from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict, List, Tuple
import os
from datetime import datetime
import numpy as np


MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/test6.mov"
OUTPUT_PATH = "output_con_metadatos.MOV"
JSON_PATH = "detecciones_baches_por_frame.json"


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


UMBRAL_DISTANCIA_GPS = 0.00002

def dms_a_decimal(dms_str: str) -> Optional[float]:
    """Convierte coordenadas DMS a decimal"""
    if not dms_str or not isinstance(dms_str, str):
        return None
        

    patron = r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])"
    match = re.match(patron, dms_str.strip())

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

def extraer_metadatos_por_frame(video_path: str) -> Optional[Dict]:
    """
    Extrae metadatos de ubicación por frame usando exiftool
    """
    try:
        if not os.path.exists(video_path):
            print(f"Error: El archivo {video_path} no existe")
            return None
            

        result = subprocess.run(
            ["exiftool", "-j", "-GPS*", "-Track*", "-XMP*", video_path],
            capture_output=True,
            text=True,
            check=True
        )

        metadata_list = json.loads(result.stdout)
        
        if not metadata_list:
            return None
            
        metadata = metadata_list[0]
        

        gps_data = []
        

        gps_lat_raw = metadata.get("GPSLatitude")
        gps_lon_raw = metadata.get("GPSLongitude")
        
        if gps_lat_raw and gps_lon_raw:
            lat_decimal = dms_a_decimal(gps_lat_raw)
            lon_decimal = dms_a_decimal(gps_lon_raw)
            
            if lat_decimal is not None and lon_decimal is not None:
                gps_data.append({
                    "timestamp": metadata.get("CreateDate", metadata.get("GPSDateTime")),
                    "lat": lat_decimal,
                    "lon": lon_decimal,
                    "frame": 0
                })
                print(f"GPS extraído: {lat_decimal}, {lon_decimal}")
        

        video_info = {
            "fecha": metadata.get("CreateDate"),
            "modelo": metadata.get("Model"),
            "duracion": metadata.get("Duration"),
            "fps": metadata.get("VideoFrameRate"),
            "nombre_archivo": os.path.basename(video_path),
            "gps_track": gps_data
        }
        
        return video_info

    except FileNotFoundError:
        print("Error: exiftool no está instalado. Instalar con: brew install exiftool")
        return None
    except Exception as e:
        print(f"Error al extraer metadatos: {e}")
        return None

def interpolar_gps_por_frame(gps_track: List[Dict], total_frames: int, fps: float) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Interpola coordenadas GPS para cada frame basado en el GPS track
    Retorna lista de (lat, lon) para cada frame
    """
    gps_por_frame = [(None, None)] * total_frames
    
    if not gps_track:
        print("⚠️ No hay datos GPS disponibles")
        return gps_por_frame
    

    if len(gps_track) == 1:
        lat = gps_track[0]['lat']
        lon = gps_track[0]['lon']
        gps_por_frame = [(lat, lon)] * total_frames
        print(f"📍 Usando GPS fijo: {lat}, {lon}")
        return gps_por_frame
    

    print(f"📍 Procesando track GPS con {len(gps_track)} puntos")
    

    gps_track.sort(key=lambda x: x.get('timestamp', ''))
    

    timestamps = []
    latitudes = []
    longitudes = []
    
    for point in gps_track:
        if point.get('timestamp') and point.get('lat') is not None and point.get('lon') is not None:
            try:
                ts = point['timestamp']
                if isinstance(ts, str):
                    if ':' in ts:
                        parts = ts.split(':')
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    else:
                        seconds = float(ts)
                else:
                    seconds = float(ts)
                
                timestamps.append(seconds)
                latitudes.append(float(point['lat']))
                longitudes.append(float(point['lon']))
            except:
                continue
    
    if not timestamps:
        print("⚠️ No se pudieron parsear timestamps GPS, usando primer punto")
        lat = gps_track[0]['lat']
        lon = gps_track[0]['lon']
        return [(lat, lon)] * total_frames
    

    duracion_total = total_frames / fps if fps > 0 else timestamps[-1]
    
    for frame_idx in range(total_frames):
        tiempo_actual = frame_idx / fps if fps > 0 else (frame_idx * timestamps[-1] / total_frames)
        

        if tiempo_actual <= timestamps[0]:
            lat = latitudes[0]
            lon = longitudes[0]
        elif tiempo_actual >= timestamps[-1]:
            lat = latitudes[-1]
            lon = longitudes[-1]
        else:
            # Interpolación lineal
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= tiempo_actual <= timestamps[i + 1]:
                    t = (tiempo_actual - timestamps[i]) / (timestamps[i + 1] - timestamps[i])
                    lat = latitudes[i] + t * (latitudes[i + 1] - latitudes[i])
                    lon = longitudes[i] + t * (longitudes[i + 1] - longitudes[i])
                    break
        
        gps_por_frame[frame_idx] = (round(lat, 6), round(lon, 6))
    
    print(f"✅ Interpolación GPS completada para {total_frames} frames")
    return gps_por_frame

def generar_enlace_google_maps(lat: float, lon: float) -> str:
    """Genera enlace de Google Maps"""
    return f"https://www.google.com/maps?q={lat},{lon}"

def es_mismo_bache(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    """Determina si dos coordenadas corresponden al mismo bache"""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return False
    diff_lat = abs(lat1 - lat2)
    diff_lon = abs(lon1 - lon2)
    return diff_lat < UMBRAL_DISTANCIA_GPS and diff_lon < UMBRAL_DISTANCIA_GPS

def procesar_video():
    """Procesa el video y guarda detecciones con GPS por frame"""
    
    print("\n" + "="*50)
    print("INICIANDO PROCESAMIENTO DE VIDEO")
    print("="*50)

    # Extraer metadatos con GPS track
    metadata = extraer_metadatos_por_frame(VIDEO_PATH)
    
    print("\n=== METADATOS DEL VIDEO ===")
    if metadata:
        print(f"Archivo: {metadata.get('nombre_archivo')}")
        print(f"Fecha: {metadata.get('fecha')}")
        print(f"Modelo: {metadata.get('modelo')}")
        print(f"Duración: {metadata.get('duracion')}")
        print(f"Puntos GPS en track: {len(metadata.get('gps_track', []))}")
    else:
        print("⚠️ No se pudieron extraer metadatos")
    
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


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {os.path.basename(VIDEO_PATH)}")
    print(f"📐 Resolución: {width}x{height}")
    print(f"🎞️ FPS: {fps}")
    print(f"📊 Total frames: {total_frames}")
    print(f"📁 Salida: {OUTPUT_PATH}\n")


    gps_por_frame = []
    if metadata and metadata.get('gps_track'):
        gps_por_frame = interpolar_gps_por_frame(
            metadata['gps_track'],
            total_frames,
            fps
        )
    else:
        print("⚠️ No hay datos GPS disponibles")
        gps_por_frame = [(None, None)] * total_frames


    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps/2,
        (width, height)
    )


    baches_detectados = {}
    detecciones_json = []
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
        

        gps_lat_actual, gps_lon_actual = gps_por_frame[frame_num - 1] if frame_num - 1 < len(gps_por_frame) else (None, None)
        

        if gps_lat_actual is None or gps_lon_actual is None:

            gps_lat_actual = 0.0
            gps_lon_actual = 0.0
            gps_valido = False
        else:
            gps_valido = True
        

        if frame_num % 100 == 0:
            progreso = (frame_num / total_frames) * 100
            if gps_valido:
                print(f"Progreso: {progreso:.1f}% - Frame {frame_num} - GPS: {gps_lat_actual:.6f}, {gps_lon_actual:.6f}")
            else:
                print(f"Progreso: {progreso:.1f}% - Frame {frame_num} - GPS: No disponible")
        
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
                        if gps_valido and es_mismo_bache(
                            gps_lat_actual, gps_lon_actual,
                            bache_info['lat'], bache_info['lon']
                        ):
                            baches_detectados[id_bache] = bache_info
                            bache_nuevo = False
                            break
                
                if bache_nuevo and gps_valido:
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
                    
                    print(f"✅ Nuevo bache ID {id_bache}: {nombre_espanol} en {gps_lat_actual:.6f}, {gps_lon_actual:.6f}")
                

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{nombre_espanol} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        

        total_baches = len(baches_detectados)
        cv2.rectangle(frame, (10, 10), (450, 100), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Baches unicos: {total_baches}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Frame: {frame_num}/{total_frames}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        

        if gps_valido:
            gps_text = f"GPS: {gps_lat_actual:.6f}, {gps_lon_actual:.6f}"
        else:
            gps_text = "GPS: No disponible"
        
        cv2.putText(
            frame,
            gps_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        out.write(frame)
        

        frame_display = cv2.resize(frame, (1280, 720))
        cv2.imshow("Deteccion de Baches", frame_display)
        
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
        "metadata_video": {
            "archivo": metadata.get('nombre_archivo') if metadata else os.path.basename(VIDEO_PATH),
            "fecha": metadata.get('fecha') if metadata else None,
            "modelo": metadata.get('modelo') if metadata else None,
            "fps": fps,
            "total_frames": total_frames,
            "duracion_segundos": total_frames / fps
        },
        "gps_info": {
            "tipo": "punto_unico",
            "coordenadas_iniciales": {
                "lat": gps_por_frame[0][0] if gps_por_frame and gps_por_frame[0][0] else None,
                "lon": gps_por_frame[0][1] if gps_por_frame and gps_por_frame[0][1] else None
            } if gps_por_frame else None
        },
        "total_baches_unicos": len(baches_detectados),
        "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_procesado": os.path.basename(VIDEO_PATH),
        "detecciones_por_frame": detecciones_json
    }
    
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 JSON guardado: {JSON_PATH}")
    print(f"🎥 Video guardado: {OUTPUT_PATH}")
    print("\n=== RESUMEN ===")
    print(f"📍 Total baches únicos: {len(baches_detectados)}")
    print(f"📸 Total detecciones: {len(detecciones_json)}")
    
    if detecciones_json:
        print("\n📌 Ejemplo de detecciones:")
        for i, bache in enumerate(detecciones_json[:5]):
            print(f"   {i+1}. Frame {bache['frame']} - ID {bache['id']}: {bache['clase']} - {bache['google_maps']}")
    
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