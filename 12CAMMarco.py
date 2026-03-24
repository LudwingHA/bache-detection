import cv2
import pytesseract
from PIL import Image
import re
import json
from ultralytics import YOLO
import numpy as np
import os
import time
from datetime import datetime

# ================================
# CONFIGURACIÓN PARA MAC M1
# ================================
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# ================================
# CONFIGURACIÓN GENERAL
# ================================
VIDEO_PATH = "./videos-m/test5.mp4"
MODEL_PATH = "./best.pt"
REPORT_NAME = "reporte_baches_geolocalizados.json"

CONF_THRESHOLD = 0.1
DISTANCIA_MINIMA_METROS = 5

# Inicializar YOLO
model = YOLO(MODEL_PATH)
reporte = {}  # Diccionario para almacenar baches únicos
frame_count = 0
detecciones_totales = 0  # Contador para debug

# ================================
# FUNCIONES DE EXTRACCIÓN DE COORDENADAS
# ================================
def extraer_coordenadas_mejorado(frame):
    """
    Extrae coordenadas del formato: W:99.173391 N:19.35642
    """
    try:
        height, width = frame.shape[:2]
        
        # ROI para la esquina inferior izquierda
        roi_y_inicio = height - 100
        roi_y_fin = height
        roi_x_inicio = 0
        roi_x_fin = 400
        
        region = frame[roi_y_inicio:roi_y_fin, roi_x_inicio:roi_x_fin]
        
        if region.size == 0:
            return None, None
        
        # Preprocesamiento
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Mostrar ROI para debug (opcional)
        # cv2.imshow("ROI GPS", binary)
        
        # OCR
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789.:WNSEW "'
        texto = pytesseract.image_to_string(binary, config=custom_config)
        
        # Buscar patrón W:99.173391 N:19.35642
        patron_w = r"W:?\s*([\d.]+)"
        patron_n = r"N:?\s*([\d.]+)"
        
        match_w = re.search(patron_w, texto)
        match_n = re.search(patron_n, texto)
        
        if match_w and match_n:
            lon = float(match_w.group(1))
            lat = float(match_n.group(1))
            
            # W es Oeste (negativo)
            lon = -lon
            
            return round(lat, 6), round(lon, 6)
            
    except Exception as e:
        print(f"Error en extraer_coordenadas: {e}")
    
    return None, None

# ================================
# FUNCIONES DE GESTIÓN DE BACHES
# ================================
def calcular_distancia(coord1, coord2):
    """Calcula distancia aproximada entre dos coordenadas"""
    if None in coord1 or None in coord2:
        return float('inf')
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Aproximación: 1 grado ≈ 111 km
    delta_lat = abs(lat1 - lat2) * 111000
    delta_lon = abs(lon1 - lon2) * 111000 * abs(np.cos(np.radians((lat1 + lat2)/2)))
    
    return np.sqrt(delta_lat**2 + delta_lon**2)

def es_bache_nuevo(lat, lon):
    """Verifica si un bache es nuevo basado en proximidad geográfica"""
    for bache_id, bache in reporte.items():
        distancia = calcular_distancia((lat, lon), (bache['lat'], bache['lon']))
        if distancia < DISTANCIA_MINIMA_METROS:
            return False
    return True

# ================================
# PROCESO PRINCIPAL
# ================================
def procesar_sistema_integrado():
    global frame_count, detecciones_totales
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video")
        return

    print("=" * 50)
    print("🚀 INICIANDO SISTEMA DE DETECCIÓN DE BACHES")
    print("=" * 50)
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🎯 Modelo: {MODEL_PATH}")
    print(f"📊 Confianza mínima: {CONF_THRESHOLD}")
    print("=" * 50)
    
    ultimas_coords = (None, None)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extraer coordenadas cada 5 frames
        if frame_count % 5 == 0:
            lat, lon = extraer_coordenadas_mejorado(frame)
            if lat and lon:
                ultimas_coords = (lat, lon)
                print(f"📍 Coordenadas detectadas: {lat}, {lon}")
        
        lat, lon = ultimas_coords
        
        # Ejecutar detección
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
        
        # Verificar si hay detecciones
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Obtener IDs si existen
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                clases = boxes.cls.int().cpu().tolist()
                
                # DEBUG: Mostrar detecciones
                print(f"\n🔍 Frame {frame_count}: {len(ids)} objetos detectados")
                
                for idx, (b_id, cls) in enumerate(zip(ids, clases)):
                    clase_nombre = model.names[cls]
                    detecciones_totales += 1
                    
                    print(f"   Detección {idx+1}: ID={b_id}, Clase={clase_nombre}")
                    
                    # Solo procesar si tenemos coordenadas
                    if lat is not None and lon is not None:
                        # Verificar si es un bache nuevo
                        if es_bache_nuevo(lat, lon):
                            # Guardar en reporte
                            reporte[b_id] = {
                                "bache_id": b_id,
                                "tipo": clase_nombre,
                                "lat": lat,
                                "lon": lon,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "frame": frame_count,
                                "google_maps": f"https://www.google.com/maps?q={lat},{lon}"
                            }
                            print(f"   ✅ NUEVO BACHE GUARDADO! ID: {b_id}")
                            print(f"      📍 {lat}, {lon}")
                            
                            # Guardar inmediatamente después de cada nueva detección
                            with open(REPORT_NAME, 'w') as f:
                                json.dump(list(reporte.values()), f, indent=4)
                            print(f"      💾 Reporte actualizado")
                        else:
                            print(f"   ⏭️  Bache ya existente (cerca de otro detectado)")
                    else:
                        print(f"   ⚠️  No hay coordenadas GPS para esta detección")
            else:
                # Si no hay IDs, usar índice como identificador temporal
                clases = boxes.cls.int().cpu().tolist()
                print(f"\n🔍 Frame {frame_count}: {len(clases)} objetos detectados (sin tracking IDs)")
                
                for idx, cls in enumerate(clases):
                    clase_nombre = model.names[cls]
                    # Crear ID temporal basado en frame y posición
                    temp_id = f"temp_{frame_count}_{idx}"
                    
                    print(f"   Detección {idx+1}: Clase={clase_nombre}")
        
        # Visualización
        annotated_frame = results[0].plot()
        
        # Mostrar información en pantalla
        if lat and lon:
            info_text = [
                f"Frame: {frame_count}",
                f"GPS: {lat:.6f}, {lon:.6f}",
                f"Baches unicos: {len(reporte)}",
                f"Detecciones totales: {detecciones_totales}",
                "Presiona 'q' para salir"
            ]
        else:
            info_text = [
                f"Frame: {frame_count}",
                f"GPS: Buscando...",
                f"Baches unicos: {len(reporte)}",
                f"Detecciones totales: {detecciones_totales}",
                "Presiona 'q' para salir"
            ]
        
        y_pos = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        # Mostrar frame
        cv2.imshow("Sistema de Deteccion de Baches", annotated_frame)
        
        # Guardar reporte cada 100 frames
        if frame_count % 100 == 0:
            with open(REPORT_NAME, 'w') as f:
                json.dump(list(reporte.values()), f, indent=4)
            print(f"\n💾 Reporte guardado automáticamente - {len(reporte)} baches únicos")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Guardar resultados finales
    with open(REPORT_NAME, 'w') as f:
        json.dump(list(reporte.values()), f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("✅ PROCESO FINALIZADO")
    print("=" * 50)
    print(f"📊 Estadísticas:")
    print(f"   • Frames procesados: {frame_count}")
    print(f"   • Detecciones totales (con YOLO): {detecciones_totales}")
    print(f"   • Baches únicos guardados: {len(reporte)}")
    print(f"📁 Reporte guardado en: {REPORT_NAME}")
    
    if reporte:
        print("\n📋 Detalle de baches guardados:")
        for bache in reporte.values():
            print(f"   • ID {bache['bache_id']}: {bache['tipo']}")
            print(f"     📍 {bache['lat']}, {bache['lon']}")
            print(f"     🗺️ {bache['google_maps']}")
    else:
        print("\n⚠️  No se guardó ningún bache")
        print("   Posibles causas:")
        print("   • El modelo no está detectando baches")
        print("   • No se están extrayendo coordenadas GPS")
        print("   • El umbral de confianza es muy alto")

if __name__ == "__main__":
    # Verificar Tesseract
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("⚠️  Tesseract no encontrado. Instálalo con: brew install tesseract")
    
    # Verificar que existe el video
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video no encontrado: {VIDEO_PATH}")
    else:
        procesar_sistema_integrado()