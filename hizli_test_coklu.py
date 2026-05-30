import os
import cv2
from src.pipeline import analyze_image
from src.visualization.annotator import annotate_image

# 3 farklı senaryo için test fotoğrafları
TEST_FILES = [
    # 1. Klasik test (Yayalar ve otobüs)
    "sample_road.jpg",
    # 2. Şehir içi trafik ve yayalar
    "/home/beyz/.gemini/antigravity/brain/d5b7ab34-3b71-454e-b8f2-15b69886df8a/dashcam_city_1778870713364.png",
    # 3. Otoyol
    "/home/beyz/.gemini/antigravity/brain/d5b7ab34-3b71-454e-b8f2-15b69886df8a/dashcam_highway_1778870691837.png"
]

def process_image(filepath, index):
    print(f"\n[{index}/3] '{os.path.basename(filepath)}' analiz ediliyor...")
    
    if not os.path.exists(filepath):
        print(f"HATA: Dosya bulunamadı: {filepath}")
        return

    # 1. Pipeline'dan geçir (Tespit, Alan Kontrolü ve Risk Skorlaması yapılır)
    result = analyze_image(filepath)
    
    # 2. Orijinal resmi yükle
    image = cv2.imread(filepath)
    if image is None:
        print("HATA: Resim okunamadı.")
        return
        
    # 3. Yeni yazdığımız Görselleştirme Modülü ile çizimleri yap
    # Bu fonksiyon yeni skorları, sahne riskini ve renkli kutuları çizer
    annotated_image = annotate_image(image, result, draw_danger_zone=True)

    # 4. Sonucu kaydet
    output_path = f"test_sonucu_{index}.jpg"
    cv2.imwrite(output_path, annotated_image)
    
    print(f"[{index}/3] Tamamlandı! Toplam nesne: {result.detection_count}")
    print(f"-> Sahne Riski: {result.scene_risk.risk_level} ({result.scene_risk.reason})")
    print(f"-> Sonuç dosyası kaydedildi: {output_path}")

def main():
    print("Yeni Risk Skorlu Çoklu Fotoğraf Testi Başlıyor...\n")
    for i, filepath in enumerate(TEST_FILES, start=1):
        process_image(filepath, i)

if __name__ == "__main__":
    main()
