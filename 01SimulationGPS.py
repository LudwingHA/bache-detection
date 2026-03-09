from ultralytics import YOLO
import cv2
import subprocess
import json
import re
from typing import Optional, Dict
import os
from datetime import datetime

# ================================
# CONFIGURACIONES
# ================================

MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/prueba8.MOV"
OUTPUT_PATH = "output_con_metadatos.MOV"
JSON_PATH = "detecciones_baches.json"

# GPS inicial (se actualizará con metadatos reales si existen)
gps_lat_actual = 19.432600
gps_lon_actual = -99.133200

# Para evitar duplicados - usamos un diccionario con más información
baches_detectados = {}  # clave: id, valor: dict con info de primera detección
detecciones_json = []

# Umbral de distancia mínima entre detecciones del mismo bache (en grados)
# Aproximadamente 2 metros (0.000018 grados ≈ 2 metros)
UMBRAL_DISTANCIA_GPS = 0.00002


# ================================
# DMS → DECIMAL
# ================================
def dms_a_decimal(dms_str: str) -> Optional[float]:
    """
    Convierte coordenadas en formato DMS (grados/minutos/segundos) a decimal.
    Ejemplo: "19 deg 25' 57.36\" N" → 19.4326
    """
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


# ================================
# SIMULAR MOVIMIENTO DESDE GPS REAL
# ================================
def simular_gps(lat, lon, paso=0.00002):
    """
    Simula movimiento del vehículo cuando no hay GPS real.
    """
    lat = lat + paso
    lon = lon + paso * 0.5
    return round(lat, 6), round(lon, 6)


# ================================
# GENERAR ENLACE DE GOOGLE MAPS
# ================================
def generar_enlace_google_maps(lat: float, lon: float) -> str:
    """
    Genera un enlace de Google Maps para la ubicación del bache.
    """
    return f"https://www.google.com/maps?q={lat},{lon}"


# ================================
# VERIFICAR SI ES EL MISMO BACHE (POR PROXIMIDAD)
# ================================
def es_mismo_bache(lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    """
    Verifica si dos coordenadas corresponden al mismo bache
    basado en una distancia umbral.
    """
    diff_lat = abs(lat1 - lat2)
    diff_lon = abs(lon1 - lon2)
    return diff_lat < UMBRAL_DISTANCIA_GPS and diff_lon < UMBRAL_DISTANCIA_GPS


# ================================
# EXTRAER METADATOS
# ================================
def extraer_metadatos(video_path: str) -> Optional[Dict]:
    """
    Extrae metadatos del video usando exiftool.
    """
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
        print("Error: exiftool no está instalado. Instálalo con: sudo apt install exiftool")
        return None
    except Exception as e:
        print(f"Error al extraer metadatos: {e}")
        return None


# ================================
# PROCESAR VIDEO
# ================================
def procesar_video():
    """
    Función principal que procesa el video y detecta baches.
    """
    global gps_lat_actual, gps_lon_actual

    print("\n" + "="*50)
    print("INICIANDO PROCESAMIENTO DE VIDEO")
    print("="*50)

    # ========================
    # METADATOS
    # ========================

    metadata = extraer_metadatos(VIDEO_PATH)

    print("\n=== METADATOS DEL VIDEO ===")

    if metadata:
        for k, v in metadata.items():
            if v:
                print(f"{k}: {v}")

        # Intentar obtener GPS real de los metadatos
        if metadata.get("gps_latitud") and metadata.get("gps_longitud"):
            lat_decimal = dms_a_decimal(metadata["gps_latitud"])
            lon_decimal = dms_a_decimal(metadata["gps_longitud"])

            if lat_decimal is not None and lon_decimal is not None:
                gps_lat_actual = lat_decimal
                gps_lon_actual = lon_decimal
                print(f"\n✅ GPS REAL DETECTADO: {gps_lat_actual}, {gps_lon_actual}")
            else:
                print("\n⚠️ No se pudo convertir GPS, se usará simulación")
        else:
            print("\n⚠️ Sin datos GPS en metadatos, se usará simulación")
    else:
        print("\n⚠️ No se pudieron extraer metadatos, se usará simulación")

    print("="*50 + "\n")

    # ========================
    # MODELO YOLO
    # ========================

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
        fps,
        (width, height)
    )

    frame_num = 0

    # Configurar el tracking
    results = model.track(
        source=VIDEO_PATH,
        conf=0.02,
        persist=True,
        stream=True,
        classes=[0, 1, 2, 3, 4, 5, 6, 7],
        verbose=False  # Reducir output
    )

    # ========================
    # LOOP PRINCIPAL
    # ========================

    print("Procesando video... (Presiona 'q' para detener)")

    for r in results:
        frame_num += 1

        # Mostrar progreso cada 100 frames
        if frame_num % 100 == 0:
            progreso = (frame_num / total_frames) * 100
            print(f"Progreso: {progreso:.1f}% - Baches detectados: {len(baches_detectados)}")

        # Actualizar GPS (simulado o real)
        gps_lat_actual, gps_lon_actual = simular_gps(
            gps_lat_actual,
            gps_lon_actual,
            paso=0.00002
        )

        tiempo_seg = frame_num / fps
        frame = r.orig_img

        # Procesar detecciones
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()
            confidences = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [1.0] * len(ids)

            for box, id_bache, cls, conf in zip(boxes, ids, clases, confidences):
                
                # Verificar si el bache ya fue detectado
                bache_nuevo = True
                
                if id_bache in baches_detectados:
                    bache_nuevo = False
                else:
                    # Verificar por proximidad GPS como respaldo
                    for bache_id, bache_info in baches_detectados.items():
                        if es_mismo_bache(
                            gps_lat_actual, gps_lon_actual,
                            bache_info['lat'], bache_info['lon']
                        ):
                            # Es el mismo bache, actualizar referencia
                            baches_detectados[id_bache] = bache_info
                            bache_nuevo = False
                            break

                if bache_nuevo:
                    # Generar enlace de Google Maps
                    google_maps_link = generar_enlace_google_maps(gps_lat_actual, gps_lon_actual)
                    
                    deteccion = {
                        "id": int(id_bache),
                        "clase": model.names[cls],
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
                    
                    # Guardar en el diccionario de baches detectados
                    baches_detectados[id_bache] = {
                        'lat': gps_lat_actual,
                        'lon': gps_lon_actual,
                        'clase': model.names[cls],
                        'frame': frame_num
                    }
                    
                    print(f"✅ Nuevo bache detectado! ID: {id_bache}, Clase: {model.names[cls]}, Ubicación: {gps_lat_actual}, {gps_lon_actual}")

                # Dibujar bounding box
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

        # Mostrar contador de baches en el frame
        total = len(baches_detectados)
        cv2.rectangle(frame, (20, 20), (380, 120), (0, 0, 0), -1)
        
        cv2.putText(
            frame,
            f"Baches unicos: {total}",
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
            print("\n⏹️ Procesamiento detenido por el usuario")
            break

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ========================
    # GUARDAR RESULTADOS
    # ========================

    print("\n" + "="*50)
    print("GUARDANDO RESULTADOS")
    print("="*50)

    # Crear resumen
    resumen = {
        "metadata_video": metadata if metadata else {},
        "total_baches_unicos": len(baches_detectados),
        "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_procesado": os.path.basename(VIDEO_PATH),
        "detecciones": detecciones_json
    }

    # Guardar JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    print(f"\n📊 Archivo JSON guardado: {JSON_PATH}")
    print(f"🎥 Video procesado guardado: {OUTPUT_PATH}")
    print("\n=== RESUMEN FINAL ===")
    print(f"📍 Total de baches únicos detectados: {len(baches_detectados)}")
    print(f"📝 Detecciones registradas: {len(detecciones_json)}")
    
    # Mostrar primeros 5 baches como ejemplo
    if detecciones_json:
        print("\n📌 Primeros baches detectados:")
        for i, bache in enumerate(detecciones_json[:5]):
            print(f"   {i+1}. ID: {bache['id']}, {bache['clase']} - {bache['google_maps']}")
    
    print("\n✅ Procesamiento completado!")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    try:
        procesar_video()
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()