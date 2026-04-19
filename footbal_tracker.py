import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

# ==========================================
# 1. Hardware și Model
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device folosit: {device}")

model = YOLO("yolov8n.pt")

# ==========================================
# 2. Video local din același folder
# ==========================================
video_path = "meci.mp4"   # schimbă aici dacă fișierul are alt nume
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Eroare: nu pot deschide video-ul {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback
fps = int(round(fps))

print(f"FPS video: {fps}")

# ==========================================
# 3. Funcție pentru format timp video
# ==========================================
def format_video_time(seconds):
    ore = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    secunde = int(seconds % 60)
    return f"{ore:02d}:{minute:02d}:{secunde:02d}"

# ==========================================
# 4. Extragerea culorii tricoului în HSV
# ==========================================
def extract_jersey_color_hsv(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame - 1))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame - 1))

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    y_start = y1 + int(h * 0.15)
    y_end = y1 + int(h * 0.40)
    x_start = x1 + int(w * 0.25)
    x_end = x2 - int(w * 0.25)

    if y_start >= y_end or x_start >= x_end:
        return None

    roi = frame[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return np.mean(hsv_roi, axis=(0, 1))

# ==========================================
# 5. Variabile pentru clustering și tracking
# ==========================================
kmeans = None
is_kmeans_trained = False
colors_for_training = []
team_labels = []
player_history = defaultdict(list)

team_colors_draw = {
    0: (0, 0, 255),      # echipa 1 - roșu
    1: (255, 0, 0),      # echipa 2 - albastru
    -1: (150, 150, 150)  # arbitri / necunoscut
}

frame_count = 0

# ==========================================
# 6. Inițializare CSV
# ==========================================
with open("raport_detectii.csv", "w", encoding="utf-8") as fisier:
    fisier.write("secunda_video,timp_video,numar_detectii\n")

print("Procesare începută. Apasă 'q' pentru oprire.")

# ==========================================
# 7. Bucla principală
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video terminat.")
        break

    frame_count += 1
    video_seconds = frame_count / fps
    timp_video = format_video_time(video_seconds)

    results = model.track(
        frame,
        classes=[0],               # doar persoane
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.15,
        iou=0.5,
        imgsz=736,
        half=(device == 'cuda'),
        verbose=False,
        device=device
    )

    # ==========================================
    # Număr detections
    # ==========================================
    numar_detectii = len(results[0].boxes)

    if results[0].boxes.id is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        current_frame_colors = []
        valid_indices = []

        for i, box in enumerate(boxes):
            color = extract_jersey_color_hsv(frame, box)
            if color is not None:
                current_frame_colors.append(color)
                valid_indices.append(i)

        # ==========================================
        # Antrenare KMeans
        # ==========================================
        if not is_kmeans_trained:
            colors_for_training.extend(current_frame_colors)

            if len(colors_for_training) > 150:
                print("Antrenez clasificarea culorilor pentru echipe...")

                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                labels = kmeans.fit_predict(colors_for_training)

                label_counts = Counter(labels)
                team_labels = [item[0] for item in label_counts.most_common(2)]

                is_kmeans_trained = True
                print("Echipe identificate. Modul live activ.")

        # ==========================================
        # Clasificare stabilă pe echipe
        # ==========================================
        else:
            if len(current_frame_colors) > 0:
                instant_labels = kmeans.predict(current_frame_colors)

                for j, index in enumerate(valid_indices):
                    x1, y1, x2, y2 = map(int, boxes[index])
                    track_id = int(track_ids[index])
                    instant_cluster = instant_labels[j]

                    player_history[track_id].append(instant_cluster)

                    if len(player_history[track_id]) > 45:
                        player_history[track_id].pop(0)

                    stable_cluster = Counter(player_history[track_id]).most_common(1)[0][0]

                    if stable_cluster == team_labels[0]:
                        team_id = 0
                    elif stable_cluster == team_labels[1]:
                        team_id = 1
                    else:
                        team_id = -1

                    draw_color = team_colors_draw[team_id]
                    thickness = 2 if team_id != -1 else 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)

                    label_text = f"E.{team_id + 1}|{track_id}" if team_id != -1 else f"Ref|{track_id}"

                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - 20)),
                        (x1 + text_width, y1),
                        draw_color,
                        -1
                    )

                    cv2.putText(
                        frame,
                        label_text,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

    # ==========================================
    # HUD pe ecran
    # ==========================================
    cv2.putText(
        frame,
        f"Obiecte: {numar_detectii}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Timp video: {timp_video}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ==========================================
    # Salvare în CSV o dată pe secundă
    # ==========================================
    if frame_count % fps == 0:
        with open("raport_detectii.csv", "a", encoding="utf-8") as fisier:
            fisier.write(f"{int(video_seconds)},{timp_video},{numar_detectii}\n")

        print(f"[{timp_video}] Obiecte detectate: {numar_detectii}")

    # ==========================================
    # Afișare
    # ==========================================
    frame_show = cv2.resize(frame, (1280, 720))
    cv2.imshow("Football Tracker - Video Local", frame_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 8. Curățare
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("Procesare terminată.")
print("Fișierul CSV a fost salvat ca: raport_detectii.csv")