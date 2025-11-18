import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict

BASE = Path("/Users/ayano/Desktop/all_results_with_labels")
FOLDERS = ["cardiomegaly", "mediastinal_widening", "3class"]

# Silmeden önce denemek için False yap: sadece neleri sileceğini yazar
DO_DELETE = True

def file_hash(p: Path, chunk=1024*1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def base_key(name_stem: str) -> str:
    """
    Dosya adının SONUNDAKI _<sayı> eklerini (bir veya birden çok) temizler.
    Örn:
      stem='..._results_1_1' -> '..._results'
      stem='..._losses_2'    -> '..._losses'
    Orta kısımlardaki 121, 3class gibi sayılara dokunmaz.
    """
    s = name_stem
    while re.search(r'_(\d+)$', s):
        s = re.sub(r'_(\d+)$', '', s)
    return s

def prefer_name(a: Path, b: Path) -> Path:
    """
    Aynı hash'teki iki dosyadan hangisi kalsın?
    - Daha kısa isimli olan (genelde fazladan ek yoktur) tercih edilir.
    - Uzunluk eşitse alfabetik olarak küçük olan tercih edilir.
    """
    if len(a.name) != len(b.name):
        return a if len(a.name) < len(b.name) else b
    return a if a.name < b.name else b

total_deleted = 0

for sub in FOLDERS:
    folder = BASE / sub
    if not folder.exists():
        print(f"[ATLA] Klasör yok: {folder}")
        continue

    # base_key+ext -> aynı içeriğe göre kümeler
    groups = defaultdict(list)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        key_stem = base_key(stem)
        key = (key_stem, p.suffix.lower())
        groups[key].append(p)

    for (key_stem, ext), files in groups.items():
        if len(files) < 2:
            continue

        # Aynı içeriğe sahip olanları hash’e göre grupla
        hash_buckets = defaultdict(list)
        for p in files:
            try:
                h = file_hash(p)
            except Exception as e:
                print(f"[HATA] Hash alınamadı: {p} -> {e}")
                continue
            hash_buckets[h].append(p)

        for h, same_files in hash_buckets.items():
            if len(same_files) < 2:
                continue

            # İçerik aynı: birini bırak, kalanları sil
            keep = same_files[0]
            for q in same_files[1:]:
                keep = prefer_name(keep, q)

            for q in same_files:
                if q == keep:
                    continue
                print(f"[SİL] {q}  (KALAN: {keep.name})")
                if DO_DELETE:
                    try:
                        q.unlink()
                        total_deleted += 1
                    except Exception as e:
                        print(f"  -> Silinemedi: {e}")

print(f"İşlem bitti. Silinen dosya sayısı: {total_deleted}")
