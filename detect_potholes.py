from ultralytics import YOLO
import cv2


MODEL_PATH = "./best.pt"
model = YOLO(MODEL_PATH)


VIDEO_PATH = "video_test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))


TARGET_CLASSES = ["pothole", "pothole_deep"]

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break


    results = model(frame, conf=0.18)
    result = results[0]
    boxes = result.boxes

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)


    cv2.imshow("Deteccion de Baches en Video", frame)


    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()