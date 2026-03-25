from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict, List
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
import time
import threading
from queue import Queue
import torch

MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/test7.MOV"
OUTPUT_PATH = "output_con_metadatos.MOV"
JSON_PATH = "detecciones_baches.json"
CAPTURAS_DIR = "detecciones_baches"

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

class OptimizedExifExtractor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.gps_data = []
        self.load_exif_data()
    
    def load_exif_data(self):
        try:
            print("Extrayendo metadatos EXIF...")
            result = subprocess.run(
                ["exiftool", "-ee", "-G3", "-j", "-n", self.video_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            data = json.loads(result.stdout)[0]
            
            doc_data = defaultdict(dict)
            for key, value in data.items():
                match = re.match(r'(Doc\d+):(.+)', key)
                if match:
                    doc_id = match.group(1)
                    field = match.group(2)
                    doc_data[doc_id][field] = value
            
            for i in range(1, len(doc_data) + 1):
                doc_key = f"Doc{i}"
                if doc_key in doc_data:
                    doc_info = doc_data[doc_key]
                    lat = doc_info.get('GPSLatitude')
                    lon = doc_info.get('GPSLongitude')
                    
                    if lat is not None and lon is not None:
                        self.gps_data.append({
                            'segundo': i - 1,
                            'latitud': float(lat),
                            'longitud': float(lon),
                            'velocidad': float(doc_info.get('GPSSpeed', 0)) if doc_info.get('GPSSpeed') else None,
                            'acelerometro': doc_info.get('Accelerometer', '000 000 000')
                        })
            
            print(f"Extraídos {len(self.gps_data)} segundos de datos GPS")
            
        except Exception as e:
            print(f"Error al extraer EXIF: {e}")
            self.gps_data = []
    
    def get_gps_for_second(self, segundo: int) -> Optional[Dict]:
        if not self.gps_data:
            return None
        
        for data in self.gps_data:
            if data['segundo'] == segundo:
                return data
        
        for i in range(len(self.gps_data) - 1):
            if self.gps_data[i]['segundo'] <= segundo <= self.gps_data[i + 1]['segundo']:
                t = (segundo - self.gps_data[i]['segundo']) / (self.gps_data[i + 1]['segundo'] - self.gps_data[i]['segundo'])
                lat = self.gps_data[i]['latitud'] + (self.gps_data[i + 1]['latitud'] - self.gps_data[i]['latitud']) * t
                lon = self.gps_data[i]['longitud'] + (self.gps_data[i + 1]['longitud'] - self.gps_data[i]['longitud']) * t
                
                return {
                    'segundo': segundo,
                    'latitud': round(lat, 6),
                    'longitud': round(lon, 6),
                    'velocidad': self.gps_data[i].get('velocidad'),
                    'acelerometro': self.gps_data[i].get('acelerometro', '000 000 000')
                }
        
        return None

def generar_enlace_google_maps(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"

def es_mismo_bache(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    return abs(lat1 - lat2) < UMBRAL_DISTANCIA_GPS and abs(lon1 - lon2) < UMBRAL_DISTANCIA_GPS

def guardar_captura_bache(frame, bbox, clase, confianza, bache_id, gps_info, segundo):
    """Guarda una captura del bache con rectángulo y overlay de información mejorado"""
    # Crear directorio si no existe
    if not os.path.exists(CAPTURAS_DIR):
        os.makedirs(CAPTURAS_DIR)
    

    img_captura = frame.copy()
    

    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_captura, (x1, y1), (x2, y2), (0, 255, 0), 4)
    

    panel_height = 380
    panel_width = 1000
    

    overlay = img_captura.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    img_captura = cv2.addWeighted(overlay, 1.0, img_captura, 0.15, 0)
    
    
    y_offset = 53
    cv2.putText(img_captura, "BACHE DETECTADO", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 2)
    

    y_offset += 53
    cv2.putText(img_captura, f"TIPO: {clase.upper()}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    

    y_offset += 53
    cv2.putText(img_captura, f"GPS: {gps_info['latitud']:.6f}, {gps_info['longitud']:.6f}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    

    y_offset += 45
    if gps_info.get('velocidad'):
        cv2.putText(img_captura, f"VELOCIDAD: {gps_info['velocidad']:.1f} km/h", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    else:
        cv2.putText(img_captura, f"VELOCIDAD: No disponible", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    

    y_offset += 45
    cv2.putText(img_captura, f"SEGUNDO: {segundo}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)


    y_offset += 45
    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(img_captura, f"{fecha_hora}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
    

    height, width = img_captura.shape[:2]

    target_width = 1024
    target_height = 768
    
    if width > target_width or height > target_height:
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_captura = cv2.resize(img_captura, (new_width, new_height))
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CAPTURAS_DIR}/bache_{bache_id:03d}_{clase}_{timestamp}.jpg"
    

    cv2.imwrite(filename, img_captura, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    return filename

def dibujar_overlay(frame, total_baches, lat, lon, velocidad, acelerometro, fps_procesamiento):
    """Dibuja overlay de información con texto más grande y visible"""
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (10, 10), (950, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 1.0, frame, 0.0, 0)
    
    cv2.rectangle(frame, (10, 10), (950, 280), (0, 255, 255), 2)
    
    cv2.putText(frame, f"BACHES DETECTADOS: {total_baches}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
    
    if lat and lon:
        cv2.putText(frame, f"GPS: {lat:.6f}, {lon:.6f}", 
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    if velocidad:
        cv2.putText(frame, f"VELOCIDAD: {velocidad:.1f} km/h", 
                    (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    return frame

def procesar_video():
    print("\n" + "="*50)
    print("DETECTOR DE BACHES - OPTIMIZADO PARA MACBOOK AIR")
    print("="*50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encuentra el modelo en {MODEL_PATH}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: No se encuentra el video en {VIDEO_PATH}")
        return
    
    exif_extractor = OptimizedExifExtractor(VIDEO_PATH)
    usar_simulacion = len(exif_extractor.gps_data) == 0
    
    print("\nCargando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Utilizando device: {device}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se puede abrir el video")
        return
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_seg = total_frames / fps
    
    print(f"\n Video: {os.path.basename(VIDEO_PATH)}")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duración: {duracion_seg:.1f} segundos")
    
    target_width = 1280
    target_height = 720
    
    if width > target_width:
        scale_x = target_width / width
        scale_y = target_height / height
        print(f"   ⚡ Escalando de {width}x{height} a {target_width}x{target_height}")
    else:
        target_width, target_height = width, height
    
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width, target_height)
    )
    
    baches_detectados = {}
    detecciones_json = []
    frame_num = 0
    last_gps_second = -1
    current_gps_info = None
    
    frame_skip = 1
    detection_counter = 0
    
    print("\nProcesando video........")
    print("Presiona 'q' para detener\n")
    
    start_time = time.time()
    frames_processed = 0
    last_fps_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        tiempo_actual = frame_num / fps
        
        segundo_actual = int(frame_num / fps)
        
        if segundo_actual != last_gps_second:
            last_gps_second = segundo_actual
            
            if not usar_simulacion:
                current_gps_info = exif_extractor.get_gps_for_second(segundo_actual)
            else:
                lat_sim = 19.38987 - (segundo_actual * 0.00005)
                lon_sim = -99.03481 - (segundo_actual * 0.00002)
                current_gps_info = {
                    'latitud': lat_sim,
                    'longitud': lon_sim,
                    'velocidad': 45.0,
                    'acelerometro': "000 000 000"
                }
        
        if width != target_width:
            frame_resized = cv2.resize(frame, (target_width, target_height))
        else:
            frame_resized = frame
        
        detection_counter += 1
        # if detection_counter >= frame_skip:
        #     detection_counter = 0
            
        if current_gps_info and current_gps_info.get('latitud'):
            try:
                results = model.track(frame_resized, conf=0.2, verbose=False, device='mps', vid_stride=2)
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        nombre_ingles = model.names[cls]
                        nombre_espanol = CLASES_ESPAÑOL.get(nombre_ingles, nombre_ingles)
                        
                        bache_id = f"{current_gps_info['latitud']:.5f}_{current_gps_info['longitud']:.5f}"
                        
                        if bache_id not in baches_detectados:
                            bache_cercano = False
                            for existing_id, existing_info in baches_detectados.items():
                                if es_mismo_bache(current_gps_info['latitud'], current_gps_info['longitud'],
                                                existing_info['lat'], existing_info['lon']):
                                    bache_cercano = True
                                    break
                            
                            if not bache_cercano:

                                x1_orig = int(x1 * (width / target_width))
                                x2_orig = int(x2 * (width / target_width))
                                y1_orig = int(y1 * (height / target_height))
                                y2_orig = int(y2 * (height / target_height))
                                

                                captura_path = guardar_captura_bache(
                                    frame, (x1_orig, y1_orig, x2_orig, y2_orig),
                                    nombre_espanol, conf, len(baches_detectados) + 1,
                                    current_gps_info, segundo_actual
                                )
                                
                                google_maps_link = generar_enlace_google_maps(
                                    current_gps_info['latitud'], 
                                    current_gps_info['longitud']
                                )
                                
                                deteccion = {
                                    "id": len(baches_detectados) + 1,
                                    "clase": nombre_espanol,
                                    "confianza": round(conf, 3),
                                    "latitud": current_gps_info['latitud'],
                                    "longitud": current_gps_info['longitud'],
                                    "velocidad_kmh": current_gps_info.get('velocidad'),
                                    "acelerometro": current_gps_info.get('acelerometro', '000 000 000'),
                                    "google_maps": google_maps_link,
                                    "segundo": segundo_actual,
                                    "frame": frame_num,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "captura": captura_path
                                }
                                
                                detecciones_json.append(deteccion)
                                
                                baches_detectados[bache_id] = {
                                    'lat': current_gps_info['latitud'],
                                    'lon': current_gps_info['longitud'],
                                    'clase': nombre_espanol,
                                    'segundo': segundo_actual,
                                    'captura': captura_path
                                }
                                
                                print(f"\n[BACHE] #{len(baches_detectados)}: {nombre_espanol}")
                                print(f"   GPS: {current_gps_info['latitud']:.5f}, {current_gps_info['longitud']:.5f}")
                                print(f"   Captura guardada: {captura_path}")
                                if current_gps_info.get('velocidad'):
                                    print(f"   Velocidad: {current_gps_info['velocidad']:.1f} km/h")
                        

                        if width != target_width:
                            x1 = int(x1 * (width / target_width))
                            x2 = int(x2 * (width / target_width))
                            y1 = int(y1 * (height / target_height))
                            y2 = int(y2 * (height / target_height))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # Texto más grande y con fondo
                        label = f"[DETECTADO]: {nombre_espanol})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2)
            
            except Exception as e:
                pass
        
        frames_processed += 1
        current_time = time.time()
        elapsed = current_time - last_fps_time
        if elapsed >= 1.0:
            proc_fps = frames_processed / elapsed
            print(f"Frame {frame_num}/{total_frames} | {proc_fps:.1f} fps | Baches: {len(baches_detectados)}", end='\r')
            frames_processed = 0
            last_fps_time = current_time
        
        frame = dibujar_overlay(
            frame,
            len(baches_detectados),
            current_gps_info['latitud'] if current_gps_info else None,
            current_gps_info['longitud'] if current_gps_info else None,
            current_gps_info.get('velocidad') if current_gps_info else None,
            current_gps_info.get('acelerometro') if current_gps_info else None,
            proc_fps if 'proc_fps' in locals() else None
        )
        
        if width != target_width:
            frame_out = cv2.resize(frame, (target_width, target_height))
        else:
            frame_out = frame
        
        out.write(frame_out)
        
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Deteccion Baches - MacBook Air", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcesamiento detenido")
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n\n" + "="*50)
    print("GUARDANDO RESULTADOS")
    print("="*50)
    
    resumen = {
        "video": os.path.basename(VIDEO_PATH),
        "fps_original": fps,
        "duracion_segundos": duracion_seg,
        "total_frames": total_frames,
        "total_baches_unicos": len(baches_detectados),
        "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "MacBook Air 2020",
        "optimizaciones": {
            "frame_skip": frame_skip,
            "resolucion_procesamiento": f"{target_width}x{target_height}",
            "resolucion_original": f"{width}x{height}"
        },
        "detecciones": detecciones_json
    }
    
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)
    
    print(f"\n Resultados guardados:")
    print(f"   JSON: {JSON_PATH}")
    print(f"   Video: {OUTPUT_PATH}")
    print(f"   Capturas: {CAPTURAS_DIR}/")
    print(f"\n RESUMEN FINAL:")
    print(f"   Total baches únicos: {len(baches_detectados)}")
    
    if detecciones_json:
        clases = [d['clase'] for d in detecciones_json]
        from collections import Counter
        conteo_clases = Counter(clases)
        print(f"\n   Tipos de baches detectados:")
        for clase, count in conteo_clases.most_common():
            print(f"      • {clase}: {count}")
        
        velocidades = [d['velocidad_kmh'] for d in detecciones_json if d.get('velocidad_kmh')]
        if velocidades:
            print(f"\n    Velocidades:")
            print(f"      • Promedio: {np.mean(velocidades):.1f} km/h")
            print(f"      • Máxima: {max(velocidades):.1f} km/h")
            print(f"      • Mínima: {min(velocidades):.1f} km/h")
    
    print("\nProcesamiento completado!")

if __name__ == "__main__":
    try:
        procesar_video()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()