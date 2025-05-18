# Abschlussprojekt: Autonomer Rennwagen

Dieses Projekt enthält eine Umsetzung einer modularen Pipeline für die Gymnasium Car Racing Simulation ([link](https://gymnasium.farama.org/environments/box2d/car_racing/)).


## 0. Anforderungen

* Python ≥ 3.12

## 1. Weitere Installationen

Folgende Installs müssen neben der Standartinstallation des Projektes (wie in der eig. ReadME) noch getätigt werden. Scikit-learn wird für die Lane Detection benötigt und Scipy für die Pfadplanung.

```bash
pip install scipy scikit-learn
```


## 2. Projektstruktur

```bash
working-dir/
│
├── car.py                    # Fahrzeug und Hauptpipeline
├── env_wrapper.py           # Gymnasium-Wrapper
├── input_controller.py      # Manuelle Steuerung
├── lane_detection.py        # Spurdetektion
├── lateral_control.py       # Querregelung
├── longitudinal_control.py  # Längsregelung
├── path_planning.py         # Pfadplanung
├── main.py                  # Pipeline-Loop
├── README.md                # Diese Datei
├── test_*.py                # Testdateien für einzelne Module
└── .gitignore               # Git-Konfiguration
```

## 3. Module

### 3.1 Spurdetektion – `lane_detection.py`

Dieses Modul erkennt linke und rechte Fahrspurbegrenzungen basierend auf Bildverarbeitung.

#### Vorgehensweise

1. **Vorverarbeitung**: Bild zuschneiden, Segmentierung via KMeans (für genauere Erklärung siehe unten), Entfernen von Curbs durch matchen der Farben.
2. **Kantenerkennung**: Laplace-Filter, Erosion, Normalisierung.
3. **Clustering**: DBSCAN zur Erkennung zusammenhängender Punkte um einzelne Lanes zu erkennen.
4. **Sortierung**: Punkte werden nach Fahrzeugnähe geordnet und ins links und rechts unterschieden.
5. **Fehlerbehandlung**: Fallback auf letzte gültige Spur bei Erkennungsfehlern.
6. **Debug-Visualisierung**: Darstellung der Spurpunkte in der Simulation.

#### Besonderheiten

##### Clustering der Farben mithilfe von KMeans

Der erste Ansatz war, das Bild einfach in ein Graubild umzuwandeln, bei dem jeder Pixel einfach der Durchschnitt aller drei Farbchannel ist.
Dies funktioniert für die Standardfarben recht gut, scheitert jedoch bei vielen Farbkombinationen der zufälligen Farben.

Hier wird der Ansatz verfolgt mithilfe eines KMeans Algorithmus, der Punkte in n-Dimensionalem Raum in eine Vorgegebene Anzahl an möglichst Zusammenhängenden Punkten einteilen kann, die unterschiedlichen Farben des Originalbildes in zwei Gruppen zu unterteilen (Straße/Hintergrund). Dies passiert, indem die drei Farbchannel der Pixel als Punkkoordinaten in 3-Dimensionalem Raum angesehen werden und anschließend alle Pixel mit der Farbe des Größeren Clusters auf 1 gesetzt werden. Der Rest bleibt 0.

Dadurch bekommt man ein klares Schwarz-Weiß Bild, unabhängig von der Farbkombination von Straße und Hintergrund, auf das perfekt eine Kantenerkennung mittels Laplacefilter angewendet werden kann. 

Visualisierbar mit `self.debug_image = binary_image` am ende der `detect()` Methode in der lane_detection.py.

##### Erosion

Nachdem ein Laplace-Filter über das Schwarz-Weiß Bild gelaufen ist, sind alle Kanten mit zwei Reihen an Pixeln markiert. Um dies zu verhindern, wird anschließend noch eine Erosion mit einem 2x2 Kernel über das Bild angewendet.

Ein glücklicker Nebeneffekt hier ist, dass die Straßenmarkierungen hiermit auch fast vollkommen entfallen.

### 3.2 Pfadplanung – `path_planning.py`

Dieses Modul plant eine Trajektorie in Form von Wegpunkten innerhalb der Fahrspur.

#### Überblick

Das System nutzt 3 (mit Startup 4) **Stages**, die modular aktiviert werden können.


#### Datenvorbereitung:

   * Am Anfang werden die linke und rechte Fahrspurpunkte auf dieselbe Länge geschnitten. Nach Berechnung des Mittelpfades werden außerdem die Punkte hinter dem Auto entfernt.

#### Stage 0: Startup-Modus (`startup_mode = True`)

* In den ersten Frames wird ein gerader Pfad nach vorne generiert, um den ursprünglichen Zoom-In besser zu handeln.

#### Stage 1: Mittelspurpfad (`just_use_middle_path = True`)

* Berechnet indexweise die Mitte zwischen linker und rechter Spur.
* Sehr Robust und standardmäßig aktiviert, auf diese Stage wurden die Parameter getuned.
* Hinweis: Durch die die Datenvorbereitung stimmen die Index teilweise nicht überein bzw. der gegenüberliegende Punkt der linken Spur ist nicht der richtige gleiche Index der Rechten Spur. Dies führt jedoch dazu, dass die Kurve am Anfang "natürlich geschnitten" wird und sobald die Indexe sich wieder beruhigen, man schon eine verbesserte Mittellinie gefahren ist.

#### Stage 2: Geradenerkennung (`check_straight_line = False`)

* Schaut ob eine Gerade vom Auto bis zum Mittelendpunkt valide ist (bzw. in der Spur liegt) und gibt diesen falls ja zurück
* Dadurch können sehr breite Kurven oder Doppelkurven stark optimeirt werden, aber da der Radius bei steilen Kurven sinkt, da man sich automatisch der inneren Spur bis zum Umschaltpunkt annähert, ist dies ebenfalls standartmäßig aus.
* Kann nicht alleine bentuzt werden und kann nur in Kombination mit Stufe 3 benutzt werden

#### Stage 3: Vektorbasierte Optimierung (`just_use_vectors = False`)

Diese Stage optimiert den Pfad entlang der Mittelspur unter Berücksichtigung der Tangentenrichtungen. Da bei manchen Tracks bzw. Kurven dieser Ansatz jedoch zum Ausreißen geführt hat, ist standartmäßig die Mittelspur aktiv, da diese robuster und performanter ist.

##### Funktionsweise

1. **Spline-Glättung**:

   * Die Mittelspur wird durch Spline-Interpolation geglättet und auf eine feste Punktanzahl reduziert.

2. **Geradheitstest**:

   * Überprüft, ob der Pfad weitgehend linear verläuft.
   * Bei sehr geraden Pfaden wird auf Vektorverschiebung verzichtet und die Mitte ausgegeben.

3. **Tangentennormalen-Berechnung**:

   * Aus den Positionen werden Tangenten- und Normalenvektoren bestimmt.
   * Die Tangente beschreibt die Bewegungsrichtung, die Normale die Kurvenrichtung.

4. **Punktanpassung bei Kurven**:

   * Bei signifikanten Richtungsänderungen (gemessen über Winkelabweichung) wird der Punkt entlang der Normalen verschoben.
   * Wie sehr die Punkte verschoben werden hängt von dem Parameter adjusment_weight ab.

5. **Zickzack-Erkennung**:

   * Entfernt oszillierende Pfade über zu große Abstnde von benachbarten Punkten, um einzelne Ausreißerpunkte zu entfernen.


#### Spezialfälle( Keine Spuren oder nur eine Seite gegeben):

* **Notfallpfad**: Wenn keine Spur oder nur die linke bzw. rechte Spur gegeben wird, wird ein gerader Pfad nach vorn oder zur erkannten Seite erzeugt.

#### Datennachbereitung:

   * Es wird am Ende immer ein Resampling mit Splines durchgeführt, um einen gleichmäßiger Abstand der Punkte zu gewährleisten, der sich an den Test-Modulen orientiert. Es wird die insgesamte Länge berechnet und mittels der gewünschten Länge (z.B. 6.5 Pixel) aus der Spline ein Pfad mit X Punkten erstellt.
   
#### Krümmungsberechnung

* Die Krümmung wird über drei Referenzpunkte (Anfang, Mitte, Ende) berechnet.

#### Parameter

| Parameter             | Beschreibung                                            | Standardwert |
| --------------------- | ------------------------------------------------------- | ------------ |
| `desired_distance`    | Abstand zwischen Wegpunkten im finalen Pfad             | `6.5`        |
| `adjustment_weight`   | Gewichtung für Kurvenanpassung in der Vektoroptimierung | `2.0`        |
| `min_curvature_value` | Minimaler Krümmungswert zur numerischen Stabilisierung  | `0.05`       |
| `startup_counter`     | "Frames", wie lange der Startup-Mode halten soll        | `60`         |



### 3.3 Querregelung – `lateral_control.py`

Dieses Modul implementiert einen vereinfachten **Stanley Controller**, der auf Querabweichung (Crosstrack Error, CTE) und Tangentenwinkel der Trajektorie basiert. Die Steuerung erfolgt rein positionsbasiert, wobei die aktuelle Fahrzeugposition fest auf `[48, 62]` gesetzt ist.
Zusätzlich verfügt der Regler über eine Dämpfung bei hohen Geschwindigkeiten um hier Schwindungen zu unterbinden.

Weiterhin wird für höhere Geschwindigkeiten ein **Pure Pursuit Controller** implementiert, der sich in einer von der Geschwindikeit (mit dem parameter `pure_pursuit_k` kontrollierbar) abhängigen Distanz vor dem Fahrzeug einen Punkt der Trajektorie aussucht und mithilfe eines Bycicle Models mit simpler Vektorenrechnung den nötigen Lenkwinkel errechnet um diesen Punkt zu schneiden.

Mit dem Parameter ``pure_pursuit_speed`` lässt sich die Geschwindigkeit einstellen, ab der der Pure Pursuit Controller benutzt wird.
Bei experiementellem Erproben hat sich herausgestellt, dass für diese Simulation ein Pure Pursuit Controller weitaus besser geeignet ist und auch in engen Kurven völlig ausreicht. Daher wurde dieser Parameter hier auf `0` gesetzt um konstant den Pure Pursuit Controller zu nutzen (sehr hohe Zahl um nur mit dem Stanley-Regler zu fahren).

#### Eigenschaften

* `k_e`: Verstärkung der Korrektur basierend auf Querfehler
* `k_s`: Geschwindigkeitsdämpfung zur Stabilisierung bei höherem Tempo
* `steering_gain`: Skalierung des Lenkergebnisses (z. B. zur Hardwareanpassung)
- `pure_pursuit_speed`: Geschwindigkeit, ab der der Pure Pursuit Controller verwendet wird
- `pure_pursuit_k`: Steuert die Distanz vor dem Auto, in der sich der angestrebte Punkt der Pure Pursuit Reglers befindet

#### Vergleich der Regler

Da der Stanley-Regler direkt versucht die Krümmung der Trajektorie nachzuahmen kann er in engen Kurven deutlich besser auf der Trajektorie bleiben wie der Pure-Pursuit Regler, tendiert aber bei höheren Geschwindigkeiten sehr schnell zum Schwingen, was mit guter Parametrieung zwar lange unterbunden aber nie komplett verhindert werden kann.

Der Pure-Pursuit Regler folgt in engen Kurven zwar nicht 100% der Trajektorie, sondern schneidet die Kurve etwas. Da dies aber in diesem Kontext sogar ein gewünschtes Verhalten ist (solang es nicht so stark passiert, dass er über das Gras fährt), wird hier rein der Pure Pursuit Controller verwendet.

### 3.4 Längsregelung – `longitudinal_control.py`



Die Längsregelung plant die Geschwindigkeit und erzeugt das Gas/Bremsverhalten mit einemmit einem **PID-Controller**. 

Die Zielgeschwindigkeit wird abhängig von Krümmung der Trajektorie und durchschnittlicher ABweichung des X-Wert von der Mitte der Trajektorienpunkte berechnet.
Wird eine hohe Krümmung in einer Trajektorie erkannt (weißt auf eine Hairpin Kurve hin), wechselt die Regelung in einen `Hairpin`-Modus, in dem nur die Krümmung für die Zielgeschwindigkeit beachtet wird. Dieser Modus besitzt auch einen "Cooldown", damit nicht direkt nach der Kurve wieder Vollgas gegeben wird, sondern dem Auto etwas Zeit gegeben wird komplett um die Kurve zu kommen.

Die Kürmmungs- und x-Abweichungswerte werden auch jeweils über einen kurzen Zeitraum in einer Liste gespeichert und nur deren Durchschnitt verwendet um kleine Outlier in der Trajektorie zu kompensieren und einen sanften Kurvenein- und austritt zu garantieren. 

Zudem kann während einer scharfen Kurve kein Gas gegeben werden und beim Übergang vom Hairpin Modues in den normalen Modus wird langsam immer mehr Gas "freigeschaltet" um einen Spinout zu vermeiden. 

Es gibt auch "gain" Parameter für Bremse und Gas um die Reglerwerte nochmal extra kontrollieren zu können (z.B. um bei der Bremse das Blockieren der Räder zu verhindern).

#### Eigenschaften

* **PID-Parameter:**

  * `kp`: Proportionalanteil
  * `ki`: Integralanteil
  * `kd`: Differentialanteil
  * `max_speed`: Die maximale Zielgeschwindigkeit
  * `min_speed`: Die minimale Zielgeschwindigkeit
  * `curvature_weight`: Faktor wie stark die Krümmung die normale Zielgeschwindigkeit beinflusst
  * `self.deviation_weight`: Faktor wie stark die x-Abweichung die normale Zielgeschwindigkeit beinflusst
  * `curvature_threshold`: "Totzone" des Krümmungswertes, unter dem vollgas gegeben wird
  * `deviation_threshold`: "Totzone" des Abweichungswertes, unter dem vollgas gegeben wird

  * `hairpin_curvature`: Krümmung, bei der in den Hairpin-Modues gewechselt wird
  * `hairpin_curvature_weight`: Einfluss der rümmung im Hairpin-Modues
  * `hairpin_max_speed`: Maximalgeschwindigkeit im Hairpin-Modues
  * `hairpin_timeout`: Anzahl der Frames, die der Regler nach der Kurve noch im Hairpin-Modus bleibt 

  * `accel_gain`: Zusätzlicher Faktor für den Gas-Endwert
  * `brake_gain`: Zusätzlicher Faktor für den Brems-Endwert 

* **Sicherheitsmechanismen:**
  * `steer_safety_threshold`: deaktiviert Beschleunigung bei zu starker Lenkung
  * Anti-Windup für Integrator (`integral_max`)
  * Historienfilter für Glättung der Abweichung und Kurvigkeit

#### Regelalgorithmus

1. Berechne Geschwindigkeitsfehler
2. Aktualisiere PID-Variablen (Fehler, Integral, Ableitung)
3. Berechne Steuergröße
4. Wandle Steuergröße in Gas oder Bremse um
5. Wende Sicherheitsfilter an:

   * Bei starkem Lenkverhalten wird Gas reduziert oder deaktiviert