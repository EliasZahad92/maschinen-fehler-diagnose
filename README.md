Das ist eines meiner kleinen Projekte, die ich durchgeführt habe.
Die .mat Rohdaten habe ich aus dem Internet geladen. 
Es handelt sich um eine automatische Erkennung von Lagerfehlern über die Zeit. 
Die Daten hierzu enthalten verschiedene Zeiten und Vibrationsdaten zu verschiedenen Zuständen (Normal = Gut, B = Kugel Schaden, IR = innerer Ring Schaden, OR = äußerer Ring Schaden).

Die Daten mussten Segmentiert werden und die Parameter wurden mittels Python berechnet:
- Mittelwert
- Standardabweichung
- MAX- und MIN-Werte
- FFT

Diese Daten wurden dann mithilfe von Random Forest in verschiednene Klassen eingestuft.

In Kürze: 
- .mat Daten wurden eingelesen
- Segmentierung in verschiedenen Fenstergrößen (hier: 1024 Datenpunkte pro Fenster)
- Erstellung von 53 Fenstern pro Segment (Mittel, Stdaw., Max, Min & FFT)
- Randon Forest Modell zum trainieren eingebaut
- Erstellung Konfusionsmatrix, um Klassifizierung bewerten zu können
