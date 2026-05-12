# Proximity Detection - Sadece Yakın Yüzleri Tanıma

## Sorun
Kamera yanından geçen kişiler bile tanınıyor. Sadece kameranın önünde duran kişiler tanınmalı.

## Çözümler

### 1. Yüz Boyutu Filtreleme (Bounding Box Size)

Hikvision event'lerinde gelen yüz koordinatlarını kontrol et, sadece belirli boyutun üzerindeki yüzleri tanı.

```python
def _on_hikvision_event(self, event_data):
    face_rect = event_data.get('faceRect', {})
    width = int(face_rect.get('width', 0))
    height = int(face_rect.get('height', 0))
    
    # Minimum yüz boyutu (pixel cinsinden)
    MIN_FACE_SIZE = 150  # 150x150 piksel ve üzeri
    
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        self.add_log(f"Yüz çok uzakta ({width}x{height}), tanıma iptal", "warning")
        return
    
    # Normal tanıma işlemi
    face_name = event_data.get('face_name', '')
    confidence = event_data.get('confidence', 0)
    
    if face_name and confidence > 0.7:
        self._process_recognition(face_name)
```

### 2. ROI (Region of Interest) - Merkez Bölge Tanımlama

Kameranın görüntüsünü bölgelere ayır, sadece merkez bölgedeki yüzleri tanı.

```python
class ProximityFilter:
    def __init__(self, image_width=640, image_height=480):
        self.width = image_width
        self.height = image_height
        
        # Merkez bölge (ROI) - %60 merkez alan
        self.roi_x = int(width * 0.2)  # %20 sol kenar boşluk
        self.roi_y = int(height * 0.1)  # %10 üst kenar boşluk
        self.roi_w = int(width * 0.6)   # %60 genişlik
        self.roi_h = int(height * 0.8)  # %80 yükseklik
        
    def is_in_roi(self, face_rect):
        """Yüz merkez bölgede mi kontrol et."""
        x = int(face_rect.get('x', 0))
        y = int(face_rect.get('y', 0))
        w = int(face_rect.get('width', 0))
        h = int(face_rect.get('height', 0))
        
        # Yüzün merkezi
        center_x = x + w // 2
        center_y = y + h // 2
        
        # ROI içinde mi kontrol et
        in_x = self.roi_x <= center_x <= self.roi_x + self.roi_w
        in_y = self.roi_y <= center_y <= self.roi_y + self.roi_h
        
        return in_x and in_y

# Kullanım
filter = ProximityFilter()

def _on_hikvision_event(self, event_data):
    face_rect = event_data.get('faceRect', {})
    
    if not self.proximity_filter.is_in_roi(face_rect):
        self.add_log("Yüz ROI bölgesinin dışında, tanıma iptal", "warning")
        return
    
    # Normal tanıma işlemi
    self._process_recognition(event_data)
```

### 3. Mesafe Tahmini (Face Size → Distance)

Yüz boyutunu kullanarak mesafe tahmini yap.

```python
class DistanceEstimator:
    def __init__(self):
        # Referans değerler
        self.reference_face_width = 200  # 1 metre uzaklıkta 200px
        self.reference_distance = 100    # 100 cm (1 metre)
        
    def estimate_distance(self, face_width):
        """Yüz genişliğinden mesafe tahmin et."""
        if face_width <= 0:
            return float('inf')
        
        # Basit perspektif formülü
        distance = (self.reference_face_width * self.reference_distance) / face_width
        return distance
    
    def is_close_enough(self, face_width, max_distance=150):
        """Yeterince yakın mı kontrol et (max_distance: cm cinsinden)."""
        distance = self.estimate_distance(face_width)
        return distance <= max_distance

# Kullanım
distance_estimator = DistanceEstimator()

def _on_hikvision_event(self, event_data):
    face_rect = event_data.get('faceRect', {})
    width = int(face_rect.get('width', 0))
    
    # Maksimum 1.5 metre tanıma mesafesi
    if not distance_estimator.is_close_enough(width, max_distance=150):
        estimated_distance = distance_estimator.estimate_distance(width)
        self.add_log(f"Mesafe çok uzak: {estimated_distance:.0f}cm", "warning")
        return
    
    # Normal tanıma işlemi
    self._process_recognition(event_data)
```

### 4. Zaman Bazlı Filtreleme (Duration)

Yüz belirli bir süre boyunca algılanmalı, geçici tespitleri yoksay.

```python
class DurationFilter:
    def __init__(self, min_frames=3, min_duration_ms=500):
        self.min_frames = min_frames
        self.min_duration_ms = min_duration_ms
        self.face_detections = {}  # {face_id: [timestamps]}
        
    def should_recognize(self, face_id, timestamp):
        """Yüz yeterince süre algılandı mı kontrol et."""
        if face_id not in self.face_detections:
            self.face_detections[face_id] = []
        
        self.face_detections[face_id].append(timestamp)
        
        # Eski tespitleri temizle (5 saniyeden eskilerini)
        now = timestamp
        self.face_detections[face_id] = [
            ts for ts in self.face_detections[face_id]
            if now - ts < 5000
        ]
        
        # Yeterince tespit var mı?
        if len(self.face_detections[face_id]) >= self.min_frames:
            first_detection = self.face_detections[face_id][0]
            duration = now - first_detection
            return duration >= self.min_duration_ms
        
        return False

# Kullanım
duration_filter = DurationFilter(min_frames=3, min_duration_ms=500)

def _on_hikvision_event(self, event_data):
    import time
    timestamp = int(time.time() * 1000)
    face_id = event_data.get('faceID', '')
    
    if not duration_filter.should_recognize(face_id, timestamp):
        self.add_log("Yüz yeterince süre algılanmadı", "warning")
        return
    
    # Normal tanıma işlemi
    self._process_recognition(event_data)
```

### 5. Kombine Çözüm (Önerilen)

Tüm yöntemleri birleştir en güvenilir çözüm:

```python
class ProximityDetector:
    def __init__(self, image_width=640, image_height=480):
        self.width = image_width
        self.height = image_height
        
        # ROI ayarları
        self.roi_x = int(image_width * 0.15)
        self.roi_y = int(image_height * 0.1)
        self.roi_w = int(image_width * 0.7)
        self.roi_h = int(image_height * 0.8)
        
        # Minimum yüz boyutu
        self.min_face_size = 150
        
        # Mesafe tahmin
        self.reference_face_width = 200
        self.reference_distance = 100
        self.max_distance = 150  # 1.5 metre
        
        # Duration filter
        self.face_detections = {}
        self.min_frames = 2
        self.min_duration_ms = 300
        
    def should_recognize(self, face_rect, face_id, timestamp):
        """Tüm filtreleri uygula."""
        x = int(face_rect.get('x', 0))
        y = int(face_rect.get('y', 0))
        w = int(face_rect.get('width', 0))
        h = int(face_rect.get('height', 0))
        
        # 1. Yüz boyutu kontrolü
        if w < self.min_face_size or h < self.min_face_size:
            return False, "Yüz çok küçük"
        
        # 2. ROI kontrolü
        center_x = x + w // 2
        center_y = y + h // 2
        
        in_roi = (self.roi_x <= center_x <= self.roi_x + self.roi_w and
                  self.roi_y <= center_y <= self.roi_y + self.roi_h)
        
        if not in_roi:
            return False, "Yüz ROI dışında"
        
        # 3. Mesafe kontrolü
        distance = (self.reference_face_width * self.reference_distance) / w
        if distance > self.max_distance:
            return False, f"Mesafe çok uzak ({distance:.0f}cm)"
        
        # 4. Duration kontrolü
        if face_id:
            if face_id not in self.face_detections:
                self.face_detections[face_id] = []
            
            self.face_detections[face_id].append(timestamp)
            
            # Eski tespitleri temizle
            now = timestamp
            self.face_detections[face_id] = [
                ts for ts in self.face_detections[face_id]
                if now - ts < 5000
            ]
            
            if len(self.face_detections[face_id]) < self.min_frames:
                return False, f"Yeterince süre değil ({len(self.face_detections[face_id])}/{self.min_frames})"
            
            first_detection = self.face_detections[face_id][0]
            duration = now - first_detection
            if duration < self.min_duration_ms:
                return False, f"Süre yetersiz ({duration}ms)"
        
        return True, "OK"

# main.py'de kullanım
class SecureFaceApp(QMainWindow):
    def __init__(self):
        # ... diğer init kodları ...
        
        # Proximity detector
        self.proximity_detector = ProximityDetector(
            image_width=640,
            image_height=480
        )
    
    def _on_hikvision_event(self, event_data):
        import time
        timestamp = int(time.time() * 1000)
        face_id = event_data.get('faceID', '')
        face_rect = event_data.get('faceRect', {})
        
        # Proximity kontrolü
        should_recognize, reason = self.proximity_detector.should_recognize(
            face_rect, face_id, timestamp
        )
        
        if not should_recognize:
            self.add_log(f"Tanıma reddedildi: {reason}", "warning")
            return
        
        # Normal tanıma işlemi
        face_name = event_data.get('face_name', '')
        confidence = event_data.get('confidence', 0)
        
        if face_name and confidence > 0.7:
            self.add_log(f"✅ {face_name} tanındı (yakın mesafe)", "success")
            self._trigger_auto_door(face_name)
```

### 6. Kamera Overlay Görselleştirme

ROI bölgesini kamera üzerinde göster:

```python
def _draw_roi_overlay(self, frame):
    """ROI bölgesini çiz."""
    detector = self.proximity_detector
    
    # ROI çerçevesi
    cv2.rectangle(
        frame,
        (detector.roi_x, detector.roi_y),
        (detector.roi_x + detector.roi_w, detector.roi_y + detector.roi_h),
        (0, 255, 0),  # Yeşil
        2
    )
    
    # Etiket
    cv2.putText(
        frame,
        "TANIMA BÖLGESI",
        (detector.roi_x, detector.roi_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

def update_frame(self):
    frame = self.camera.get_frame()
    if frame:
        # ROI overlay çiz
        self._draw_roi_overlay(frame)
        self._display_frame(frame)
```

## Ayarlar Paneli Ekleme

Kullanıcı proximity ayarlarını değiştirebilsin:

```python
# settings_window.py'e ekle
class ProximitySettings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Minimum yüz boyutu
        layout.addWidget(QLabel("Minimum Yüz Boyutu (px):"))
        self.min_face_size = QSpinBox()
        self.min_face_size.setRange(50, 500)
        self.min_face_size.setValue(150)
        layout.addWidget(self.min_face_size)
        
        # Maksimum mesafe
        layout.addWidget(QLabel("Maksimum Mesafe (cm):"))
        self.max_distance = QSpinBox()
        self.max_distance.setRange(50, 300)
        self.max_distance.setValue(150)
        layout.addWidget(self.max_distance)
        
        # Minimum süre
        layout.addWidget(QLabel("Minimum Süre (ms):"))
        self.min_duration = QSpinBox()
        self.min_duration.setRange(100, 2000)
        self.min_duration.setValue(300)
        layout.addWidget(self.min_duration)
        
        self.setLayout(layout)
```

## Özet

**En etkili çözüm:** Kombine yaklaşım (yüz boyutu + ROI + mesafe + duration)

**Önerilen ayarlar:**
- Minimum yüz boyutu: 150px
- ROI: %60-70 merkez bölge
- Maksimum mesafe: 150cm (1.5m)
- Minimum süre: 300ms

Bu ayarlarla, sadece kameranın önünde duran kişiler tanınacak, geçen kişiler yoksayılacak.
