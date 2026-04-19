# ⚽ Documentație Tehnică: Advanced Football Tracker

Acest document descrie arhitectura și conceptele tehnice ale aplicației de monitorizare și analiză tactică a meciurilor de fotbal, bazată pe Computer Vision și Machine Learning.

## 🏗️ 1. Arhitectura Sistemului

Sistemul este construit pe un pipeline liniar de procesare, optimizat pentru latență scăzută:

1.  **Ingestie Video**: Stream live via `yt-dlp` și `CamGear`.
2.  **Inference Engine**: Detectarea obiectelor (jucătorilor) folosind YOLOv8m.
3.  **Object Tracking**: Asocierea ID-urilor unice prin ByteTrack.
4.  **Feature Extraction**: Analiza culorii în spațiul HSV (zona centrală a corpului).
5.  **Unsupervised Learning**: Clasificarea automată a echipelor prin K-Means.
6.  **Temporal Filtering**: Stabilizarea deciziei prin vot majoritar (sliding window).
7.  **Vizualizare**: Overlay grafic în timp real prin OpenCV.

---

## 🔬 2. Tehnologii și Concepte Cheie

### A. YOLOv8 (You Only Look Once) - Detecție
Modelul YOLOv8m (Medium) este utilizat pentru a identifica jucătorii. 
- **Input**: Cadre redimensionate la 736px (`imgsz=736`) pentru a echilibra viteza cu detaliile.
- **Optimizare FP16**: Utilizarea preciziei la jumătate (`half=True`) pentru a maximiza throughput-ul pe GPU-uri NVIDIA (Tensor Cores).

### B. Spațiul de Culoare HSV vs. RGB
Spre deosebire de RGB, spațiul **HSV (Hue, Saturation, Value)** este mult mai robust în mediul exterior:
- **Hue (Nuanța)**: Permite identificarea culorii tricoului indiferent de intensitatea luminii.
- **Value (Luminozitatea)**: Permite ignorarea umbrelor aruncate de tribune sau de alți jucători pe suprafața echipamentului.

### C. K-Means Clustering - Clasificarea Echipelor
Deoarece nu știm dinainte ce culori au echipele, folosim K-Means pentru a învăța "live":
- **Algoritmul**: Grupează vectorii de culoare $(H,S,V)$ în $k$ clustere ($k=3$ pentru Echipa A, Echipa B și Alții).
- **Logica de selecție**: Cele mai populate două clustere sunt automat identificate ca fiind echipele de câmp, eliminând necesitatea configurării manuale.

### D. ByteTrack - Tracking Temporal
ByteTrack rezolvă problema ocluziilor (când un jucător trece prin fața altuia):
- Folosește un **Filtru Kalman** pentru a prezice traiectoria viitoare a jucătorului.
- Chiar dacă detecția dispare pentru câteva cadre, tracker-ul menține ID-ul activ bazat pe mișcarea predictibilă.

---

## 🛠️ 3. Cerințe de Instalare

Pentru a rula acest sistem, este necesar un mediu Python 3.9+ și următoarele biblioteci:

```bash
# Manager de pachete rapid 'uv'
uv pip install ultralytics opencv-python vidgear scikit-learn numpy torch yt-dlp
```

---

## ⚡ 4. Optimizări de Performanță (Producție)

Pentru implementări profesionale (ex: analiză live pe marginea terenului), se recomandă exportul modelului în format **TensorRT**:

```bash
# Comandă terminal
yolo export model=yolov8m.pt format=engine half=True
```

**Avantaje:**
- Reducerea latenței de la ~30ms la ~8ms pe cadru.
- Reducerea consumului de energie al procesorului grafic.
- Posibilitatea de a procesa fluxuri 4K în timp real.

---

## 📈 5. Logică de Stabilizare (Temporal Voting)

Pentru a evita schimbarea culorii box-ului unui jucător din cauza reflexiilor luminii, am implementat un sistem de memorie:
- Se păstrează un istoric de **45 de cadre** (aprox. 1.5 secunde) pentru fiecare jucător.
- Echipa afișată pe ecran este determinată de **Modulul Statistic** (valoarea cea mai frecventă) din acest istoric.
- Rezultatul este o interfață stabilă, fără "pâlpâiri" vizuale.

