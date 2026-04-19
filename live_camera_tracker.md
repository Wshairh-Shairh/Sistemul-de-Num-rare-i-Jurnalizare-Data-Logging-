# 🚦 Documentație Tehnică: Live Street Object Detector

Acest document descrie arhitectura și funcționalitatea unei aplicații de Computer Vision care analizează în timp real un flux video (Live Stream) de pe YouTube, cu scopul de a detecta și monitoriza pietonii, mașinile și motocicletele.

## 🏗️ 1. Arhitectura Sistemului și Fluxul de Date

Aplicația folosește un pipeline simplu, dar extrem de eficient, proiectat pentru procesare în timp real:

1. **Video Ingestion (CamGear)**: Se conectează la serverele YouTube și extrage cadrele video (frames) în timp real, direct în memoria RAM, fără a descărca fișiere locale.
2. **Inference (YOLOv8)**: Fiecare cadru este trimis către rețeaua neuronală YOLOv8 pentru analiză.
3. **Filtrare (Targeted Detection)**: Modelul este instruit să ignore obiectele irelevante (cum ar fi semafoare, câini, bănci) și să returneze exclusiv coordonatele pentru pietoni și vehicule.
4. **Rendering & Afișare (OpenCV)**: Rezultatele (bounding boxes și etichete) sunt desenate peste imaginea originală și afișate pe ecran.

---

## 🔬 2. Tehnologii Cheie Utilizate

### A. Modelul YOLOv8n (Nano)
În acest script s-a optat pentru `yolov8n.pt`. Sufixul **"n" (Nano)** indică faptul că acesta este cel mai mic și mai rapid model din familia YOLOv8. 
* **Avantaj**: Este perfect pentru procesare în timp real, capabil să ruleze cu un număr mare de cadre pe secundă (FPS) chiar și pe procesoare standard (CPU), nefiind absolut dependentă de o placă video dedicată.

### B. Vidgear (CamGear)
Spre deosebire de tradiționalul `cv2.VideoCapture`, care are deseori probleme cu stream-urile web complexe, `CamGear` folosește backend-ul `yt-dlp` pentru a decoda eficient fluxurile YouTube Live. Parametrul `stream_mode=True` îi spune bibliotecii să minimizeze latența (buffering-ul) pentru a menține un feed cu adevărat "live".

---

## 🧠 3. Logica de Detecție și Filtrare

Inima aplicației se află în această linie de cod:
`results = model(frame, classes=[0, 2, 3], conf=0.4, verbose=False)`

* **`classes=[0, 2, 3]`**: Modelul a fost antrenat pe setul de date COCO (care conține 80 de tipuri de obiecte). Pentru a nu aglomera ecranul inutil, extragem strict clasele care ne interesează într-un scenariu de trafic:
    * `0` = Persoană (Pieton)
    * `2` = Mașină
    * `3` = Motocicletă
* **`conf=0.4`**: Pragul de încredere (Confidence Threshold). Algoritmul va desena un chenar doar dacă este cel puțin 40% sigur că a identificat corect obiectul. Aceasta previne detecțiile false (False Positives), cum ar fi confundarea unui hidrant cu o persoană.
* **`verbose=False`**: Oprește printarea detaliilor de inferență în terminal pentru fiecare cadru, păstrând consola curată.

---

## 🛠️ 4. Instalare și Rulare

Pentru a rula acest script, ai nevoie de următoarele biblioteci instalate în mediul tău Python:

```bash
# Folosind pip standard (sau uv pip):
pip install ultralytics opencv-python vidgear yt-dlp
```

**Rularea scriptului:**
Odată pornit scriptul, va dura câteva secunde pentru rezolvarea link-ului de YouTube. Ulterior se va deschide fereastra "YouTube Live Object Detection".
* Pentru a închide aplicația curat și a elibera resursele memoriei, **apăsați tasta `q`** pe tastatură cât timp fereastra video este activă.

---

## 💡 5. Idei de Extindere (Next Steps)

Fiind un cod de bază modular, acesta poate fi extins ușor pentru cazuri de utilizare mai complexe:
* **Numărarea Traficului (Line Crossing)**: Trasarea unei linii virtuale pe stradă și incrementarea unui contor atunci când mașinile trec peste ea.
* **Heatmaps**: Monitorizarea zonelor unde pietonii staționează cel mai mult pe trotuar.
* **Alertă de viteză / ambuteiaje**: Integrarea modulului de tracking (ex: `ByteTrack`) pentru a calcula viteza relativă a mașinilor sau pentru a detecta când traficul stă pe loc.