import tensorflow as tf
import numpy as np
import cv2
import json
from ultralytics import YOLO

# Configuración
MODEL_PATH = "./best.pt"
TF_PATH = "./videos-m/prueba3.tfrecord"
REPORT_NAME = "reporte_final_baches.json"

model = YOLO(MODEL_PATH)
reporte = {}

def parse_record(record):
    # Definimos el esquema exacto que usamos al crear el archivo
    schema = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
    }
    return tf.io.parse_single_example(record, schema)

def procesar():
    dataset = tf.data.TFRecordDataset(TF_PATH)
    
    print(f"🚀 Procesando {TF_PATH}...")
    
    for i, data in enumerate(dataset.map(parse_record)):
        # Extraer datos
        img_bytes = data['image_raw'].numpy()
        lat = float(data['latitude'].numpy())
        lon = float(data['longitude'].numpy())
        
        # Decodificar imagen
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # YOLO Tracking
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            clases = results[0].boxes.cls.int().cpu().tolist()
            
            for b_id, cls in zip(ids, clases):
                if b_id not in reporte:
                    reporte[b_id] = {
                        "bache_id": b_id,
                        "tipo": model.names[cls],
                        "ubicacion": {"lat": lat, "lon": lon},
                        "mapa": f"https://www.google.com/maps?q={lat},{lon}"
                    }
        
        # Ver progreso
        cv2.imshow("Deteccion", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    
    # Guardar reporte
    with open(REPORT_NAME, 'w') as f:
        json.dump(list(reporte.values()), f, indent=4)
    print(f"✅ Proceso terminado. Reporte guardado en {REPORT_NAME}")

if __name__ == "__main__":
    procesar()