import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
from datetime import datetime

# 1. Încărcăm modelul pre-antrenat
model = YOLO('yolov8n.pt')

# 2. Definim link-ul de YouTube
youtube_url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk'

# Inițializăm accesul la stream-ul de YouTube folosind CamGear
print("Conectare la YouTube în curs...")
stream = CamGear(source=youtube_url, stream_mode=True, logging=True).start()

print("Procesare începută... Apasă tasta 'q' pe tastatură pentru a opri.")

frame_count = 0
log_interval = 30  # salvează o dată la 30 cadre (~1 secundă la 30 FPS)

# Opțional: scriem antetul CSV o singură dată
with open("raport_trafic.csv", "w", encoding="utf-8") as fisier:
    fisier.write("timestamp,persoane,masini\n")

while True:
    frame = stream.read()
    if frame is None:
        print("Stream-ul s-a terminat sau s-a întrerupt.")
        break

    frame_count += 1

    results = model(frame, classes=[0, 2], conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    numar_persoane = 0
    numar_masini = 0

    for box in results[0].boxes:
        cls = int(box.cls)
        if cls == 0:
            numar_persoane += 1
        elif cls == 2:
            numar_masini += 1

    # Timestamp curent
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Afișare pe imagine
    cv2.putText(annotated_frame, f"Persoane: {numar_persoane}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Mașini: {numar_masini}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Timp: {timestamp}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YouTube Live Object Detection", annotated_frame)

    # Print în consolă
    print(f"[{timestamp}] Persoane: {numar_persoane}, Mașini: {numar_masini}")

    # Logare doar la fiecare 30 cadre
    if frame_count % log_interval == 0:
        with open("raport_trafic.csv", "a", encoding="utf-8") as fisier:
            fisier.write(f"{timestamp},{numar_persoane},{numar_masini}\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()