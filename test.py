import cv2
from ultralytics import YOLO
import os
from multiprocess import Process

# Carga el modelo YOLO
model = YOLO('yolov8n.pt')

def run():
    capture = cv2.VideoCapture("rtsp://link", cv2.CAP_FFMPEG)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Realizar la detección de objetos con YOLO
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    p = Process(target=run)
    p.start()
    p.join()
