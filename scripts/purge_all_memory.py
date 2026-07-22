import os
import shutil
from pathlib import Path

def purge_all_memory():
    """
    Sistemi tamamen sıfırlar:
    - Eski eğitilmiş model ağırlıklarını (.pt) siler.
    - Geçici veri önbelleğini (data cache) temizler.
    - Eski log ve checkpoint klasörlerini boşaltır.
    """
    base_dir = Path(__file__).resolve().parent.parent
    
    dirs_to_purge = [
        base_dir / "checkpoints",
        base_dir / "test_checkpoints",
        base_dir / "logs",
        base_dir / "test_logs",
        base_dir / "aether" / "data" / "cache"
    ]
    
    purged_count = 0
    
    for folder in dirs_to_purge:
        if folder.exists():
            for file in folder.glob("*"):
                try:
                    if file.is_file() or file.is_symlink():
                        file.unlink()
                        purged_count += 1
                    elif file.is_dir():
                        shutil.rmtree(file)
                        purged_count += 1
                except Exception as e:
                    print(f"[UYARI] {file} silinirken hata: {e}")
            print(f"[TEMİZLENDİ] {folder.name} klasörü sıfırlandı.")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"[OLUŞTURULDU] {folder.name} temiz olarak açıldı.")
            
    print(f"\n[BASARILI] Toplam {purged_count} eski dosya/onbellek temizlendi. Sistem bombos ve sifirlandi.")

if __name__ == "__main__":
    purge_all_memory()

