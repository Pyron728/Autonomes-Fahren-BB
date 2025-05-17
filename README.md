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

1. **Vorverarbeitung**: Bild zuschneiden, Vorderwagen maskieren, Segmentierung via KMeans, Entfernen von Grasfarben.
2. **Kantenerkennung**: Laplace-Filter, Erosion, Normalisierung.
3. **Clustering**: DBSCAN zur Erkennung zusammenhängender Punkte.
4. **Sortierung**: Punkte werden nach Fahrzeugnähe geordnet.
5. **Fehlerbehandlung**: Fallback auf letzte gültige Spur bei Erkennungsfehlern.
6. **Debug-Visualisierung**: Darstellung der Spurpunkte in der Simulation.

### 3.2 Pfadplanung – `path_planning.py`

Dieses Modul plant eine Trajektorie in Form von Wegpunkten innerhalb der Fahrspur.

#### Überblick

Das System nutzt drei aufeinander aufbauende **Stages**, die modular aktiviert werden können.

#### Stage 1: Mittelspurpfad (`just_use_middle_path = True`)

* Berechnet indexweise die Mitte zwischen linker und rechter Spur.
* Robust und standardmäßig aktiviert.

#### Stage 2: Geradenerkennung (`check_straight_line = False`)

* Schaut ob eine Gerade vom Auto bis zum Mittelendpunkt valide ist (bzw. in der Spur liegt) und gibt diesen falls ja zurück
* Dadurch können sehr breite Kurven oder Doppelkurven stark optimeirt werden, aber da der Radius bei steilen Kurven sinkt, da man sich automatisch der inneren Spur bis zum Umschaltpunkt annähert, ist dies ebenfalls standartmäßig aus.
* Kann nicht alleine bentuzt werden und kann nur in Kombination mit Stufe 3 benutzt werden

#### Stage 3: Vektorbasierte Optimierung (`just_use_vectors = False`)

Diese Stage optimiert den Pfad entlang der Mittelspur unter Berücksichtigung der Tangentenrichtungen. Da bei manchen Tracks bzw. Kurven dieser Pfad jedoch zum Ausreßen geführt hat, ist standartmäßig die Mittelspur aktiv, da diese Robuster ist un der Score auch gut genug war und nicht diese Stage:

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

   * Entfernt oszillierende Pfade über Winkelgrenzen, um einzelne Ausreßerpunkte zu entfernen.

6. **Entfernung zu langer Segmente**:

   * Überlange Streckensegmente werden entfernt, um die Pfaddichte einheitlich zu halten.



#### Spezialfälle

* **Startmodus**: In den ersten Frames wird ein gerader Pfad generiert, da hier die gesamte Strecke sichtbar ist.
* **Notfallpfad**: Wenn keine Spur erkannt wird, wird ein alternativer Pfad nach vorn oder entlang der erkannten Seite erzeugt.

#### Resampling:

   * Es wird immer ein Resampling durchgeführt, um einen gleichmäßiger Abstand der Punkte zu gewährleisten, der sich an den Test-Modulen orientiert. Es wird die insgesamte Länge berechnet und dann mittels Interpolation die Anzahl an Punktne erstellt, die der gewünschten Länge entsprechen (z.B. 6.5 Pixel).

#### Krümmungsberechnung

* Die Krümmung wird über drei Referenzpunkte (Anfang, Mitte, Ende) berechnet.

#### Parameter

| Parameter             | Beschreibung                                            | Standardwert |
| --------------------- | ------------------------------------------------------- | ------------ |
| `desired_distance`    | Abstand zwischen Wegpunkten im finalen Pfad             | `6.5`        |
| `adjustment_weight`   | Gewichtung für Kurvenanpassung in der Vektoroptimierung | `2.0`        |
| `min_curvature_value` | Minimaler Krümmungswert zur numerischen Stabilisierung  | `0.05`       |
| `startup_counter`     | "Frames", wie lange der Startup-Mode halten soll        | `60`       |



### 3.2 Querregelung – `lateral_control.py`

Dieses Modul implementiert einen vereinfachten **Stanley Controller**, der auf Querabweichung (Crosstrack Error, CTE) und Tangentenwinkel der Trajektorie basiert. Die Steuerung erfolgt rein positionsbasiert, wobei die aktuelle Fahrzeugposition fest auf `[48, 62]` gesetzt ist.

#### Eigenschaften

* `k_e`: Verstärkung der Korrektur basierend auf Querfehler
* `k_s`: Geschwindigkeitsdämpfung zur Stabilisierung bei höherem Tempo
* `steering_gain`: Skalierung des Lenkergebnisses (z. B. zur Hardwareanpassung)

#### Methode

```python
control(trajectory: np.ndarray, speed: float) -> float
```

* **Input:**

  * `trajectory`: geplante Trajektorie als Liste von 2D-Punkten (`np.ndarray`)
  * `speed`: aktuelle Fahrzeuggeschwindigkeit
* **Output:**

  * berechneter Lenkwinkel in Radiant (float)

#### Vorgehen

1. Bestimmung des nächstgelegenen Punkts auf der Trajektorie vor dem Fahrzeug
2. Berechnung der Trajektorientierung (Tangente)
3. Projektion des Querfehlers auf die Vorderachse
4. Anwendung der Stanley-Regelung:

   $$
   \delta = \theta + \arctan\left(\frac{k_e \cdot CTE}{v + k_s}\right)
   $$

   wobei $\theta$ der Tangentenwinkel ist.

---

### 3.4 Längsregelung – `longitudinal_control.py`



Die Längsregelung plant die Geschwindigkeit und erzeugt das Gas/Bremsverhalten mit einemmit einem **PID-Controller**. 

#### Eigenschaften

* **Zielgeschwindigkeit basiert auf:**

  * **Querabweichung:** größere Abweichung → geringere Zielgeschwindigkeit
  * **Kurvigkeit:** höhere Kurvigkeit → geringere Zielgeschwindigkeit
* **PID-Parameter:**

  * `kp`: Proportionalanteil
  * `ki`: Integralanteil
  * `kd`: Differentialanteil
* **Sicherheitsmechanismen:**

  * `steer_safety_threshold`: deaktiviert Beschleunigung bei zu starker Lenkung
  * Anti-Windup für Integrator (`integral_max`)
  * Historienfilter für Glättung der Abweichung und Kurvigkeit

#### Methoden

##### `predict_target_speed(trajectory: np.ndarray) -> float`

Berechnet die Zielgeschwindigkeit aus:

* `x_deviation_target_speed()`: mittlere Querabweichung über die ersten 15 Punkte
* `curvature_target_speed()`: mittlere Winkeländerung entlang der Trajektorie
  → Beide Geschwindigkeiten werden gemittelt.

##### `control(current_speed: float, target_speed: float, steer: float) -> Tuple[float, float]`

* **Input:**

  * `current_speed`: aktuelle Fahrzeuggeschwindigkeit
  * `target_speed`: zuvor berechnete Zielgeschwindigkeit
  * `steer`: Lenkwinkel (z. B. von `LateralControl`)
* **Output:**

  * `gas`: Gaswert (0–1)
  * `brake`: Bremswert (0–1)

#### Regelalgorithmus

1. Berechne Geschwindigkeitsfehler
2. Aktualisiere PID-Variablen (Fehler, Integral, Ableitung)
3. Berechne Steuergröße
4. Wandle Steuergröße in Gas oder Bremse um
5. Wende Sicherheitsfilter an:

   * Bei starkem Lenkverhalten wird Gas reduziert oder deaktiviert