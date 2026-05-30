import urllib.request
import cv2
import numpy as np
from src.pipeline import analyze_image
from src.risk.danger_zone import DangerZone

def main():
    print("1. Örnek yol fotoğrafı indiriliyor...")
    img_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    urllib.request.urlretrieve(img_url, "sample_road.jpg")
    
    print("2. Resim analiz ediliyor (YOLOv8 yükleniyor)...")
    # analyze_image fonksiyonu tüm işi yapar
    result = analyze_image("sample_road.jpg")
    
    print(f"-> Analiz bitti! {result.detection_count} adet nesne bulundu.")
    
    # 3. Sonuçları görselleştirelim (çizim yapalım)
    print("3. Bounding box'lar ve tehlike bölgesi çiziliyor...")
    image = cv2.imread("sample_road.jpg")
    h, w = image.shape[:2]
    
    # Tehlike bölgesini mavi renkle çizelim
    zone = DangerZone(w, h)
    polygon = zone.get_polygon()
    cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=3)
    
    # Tespit edilen nesneleri çizelim
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        
        # Eğer nesne tehlike bölgesindeyse Kırmızı, değilse Yeşil çizelim
        color = (0, 0, 255) if det.in_danger_zone else (0, 255, 0)
        
        # Kutuyu çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Yazıyı hazırla (Sınıf ismi + Tehlike durumu)
        label = f"{det.class_name}"
        if det.in_danger_zone:
            label += " (TEHLIKEDE)"
            
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Alt-orta noktayı (Tehlike kontrol noktası) işaretle
        cx, cy = det.bottom_center
        cv2.circle(image, (cx, cy), 5, color, -1)

    # Sonucu kaydet
    output_path = "test_sonucu.jpg"
    cv2.imwrite(output_path, image)
    print(f"4. İşlem tamam! Sonuç dosyası kaydedildi: {output_path}")

if __name__ == "__main__":
    main()
