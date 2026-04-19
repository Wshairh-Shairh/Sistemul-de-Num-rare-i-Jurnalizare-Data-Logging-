# 📝 Tema Practică: "Sistemul de Numărare și Jurnalizare (Data Logging)"

**Obiectivul temei:** Să modifici ambele scripturi pentru a număra câte obiecte sunt detectate în fiecare cadru și să salvezi aceste numere într-un fișier text sau CSV pentru a putea face grafice ulterior.

Această temă adaugă valoare comercială proiectelor tale: nu doar te uiți la un video, ci extragi **date**.

### ⚽ Cum se aplică la Proiectul 1 (Fotbal)
Vei număra câți jucători sunt vizibili pe ecran în fiecare moment.
* *Utilitate:* Dacă numărul scade drastic, poate indica faptul că acțiunea s-a mutat pe o parte a terenului, jocul este oprit sau camera a făcut zoom pe un antrenor.

### 🚦 Cum se aplică la Proiectul 2 (Trafic)
Vei număra separat mașinile și pietonii prezenți pe stradă.
* *Utilitate:* Poți aduna date timp de o oră și, analizând fișierul generat, poți spune exact în ce minut a fost traficul cel mai aglomerat.

---

## 🛠️ Pașii de rezolvare (Indicii de cod)

Tema este gândită să necesite adăugarea a doar **3-4 linii de cod** în interiorul buclei `while True:`. 

**Pasul 1: Numărarea obiectelor**
După ce modelul YOLO îți oferă `results`, poți afla foarte simplu câte obiecte a găsit folosind funcția `len()` (lungimea listei de rezultate).
```python
# Câte cutii (bounding boxes) au fost desenate pe ecran în acest cadru?
numar_detectii = len(results[0].boxes)
```

**Pasul 2: Afișarea pe ecran (Heads-Up Display)**
Vrem să vedem acest contor clar pe video. Folosește `cv2.putText` pentru a scrie numărul într-un colț al ecranului (de exemplu, stânga-sus).
```python
# Afișează textul pe frame-ul procesat
cv2.putText(frame, f"Obiecte detectate: {numar_detectii}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

**Pasul 3: Salvarea într-un fișier (Baza de date simplă)**
Pentru a nu pierde datele când închizi programul, deschide un fișier în modul "append" (`'a'`) – adică adaugă la sfârșit fără a șterge ce era înainte – și scrie numărul.
```python
# Deschidem un fișier și scriem pe un rând nou
with open("raport_trafic.csv", "a") as fisier:
    fisier.write(f"{numar_detectii}\n")
```
*(Extra tip: Pentru a nu salva de 30 de ori pe secundă, poți pune o condiție să salveze doar o dată la un anumit număr de cadre).*

