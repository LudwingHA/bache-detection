from ultralytics import YOLO
import cv2

# Configuraciones iniciales
MODEL_PATH = "./best.pt"
VIDEO_PATH = "./videos-m/prueba10.mov"
OUTPUT_PATH = "output_con_conteo.mp4"

# Set para almacenar IDs únicos de baches detectados
baches_detectados = set()

def procesar_video():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Propiedades del video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Usamos .track() en lugar de .predict()
    # persist=True mantiene los IDs entre frames
    results = model.track(source=VIDEO_PATH, 
                          conf=0.02, 
                          persist=True, 
                          stream=True,
                          classes=[6, 7]) # Asegúrate que 0 y 1 sean tus IDs de baches

    for r in results:
        frame = r.orig_img
        
        # Si el modelo detectó algo en este frame
        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist() # IDs únicos de este frame
            boxes = r.boxes.xyxy.int().cpu().tolist()
            clases = r.boxes.cls.int().cpu().tolist()

            for box, id_bache, cls in zip(boxes, ids, clases):
                # Agregamos el ID al set (si ya existe, no hace nada)
                baches_detectados.add(id_bache)

                # Dibujar bache
                x1, y1, x2, y2 = box
                nombre = model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{id_bache} {nombre}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar el contador total en la esquina del video
        total_baches = len(baches_detectados)
        cv2.rectangle(frame, (20, 20), (300, 80), (0, 0, 0), -1) # Fondo negro para el texto
        cv2.putText(frame, f"Baches Totales: {total_baches}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Conteo de Baches en Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"--- RESUMEN ---")
    print(f"Se detectaron un total de {len(baches_detectados)} baches únicos.")

if __name__ == "__main__":
    procesar_video()