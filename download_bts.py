import requests
import zipfile
import os
import time

os.makedirs("data/raw", exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

# 完整链接，覆盖2019-2024年全部月份（有重叠，后面去重）
urls = [
    # 滚动批次（含2019-2024跨年数据）
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DD.DB28DS.WAC.201802.201901.REL01.09APR2019.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.201902.202001.REL01.06APR2020.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202002.202101.REL01.06APR2021.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202102.202201.REL01.05APR2022.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202202.202301.REL01.05APR2023.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202302.202401.REL01.01APR2024.zip",
    # 自然年批次（完整覆盖2023、2024全年）
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202301.202312.REL01.04MAR2024.zip",
    "https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/DB28SEG.DD.WAC.202401.202412.REL01.04MAR2025.zip",
]

# ── 下载 ──────────────────────────────────────
print("=" * 55)
print("Step 1: 下载")
print("=" * 55)
for url in urls:
    filename = url.split("/")[-1]
    outpath  = f"data/raw/{filename}"
    if os.path.exists(outpath):
        print(f"已存在跳过: {filename}")
        continue
    print(f"下载: {filename}...", end=" ", flush=True)
    try:
        r = session.get(url, timeout=180)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(outpath, "wb") as f:
                f.write(r.content)
            print(f"✓  ({len(r.content)//1024//1024} MB)")
        else:
            print(f"✗  状态码:{r.status_code}")
    except Exception as e:
        print(f"✗  错误:{e}")
    time.sleep(2)

# ── 解压 ──────────────────────────────────────
print("\n" + "=" * 55)
print("Step 2: 解压")
print("=" * 55)
for fname in sorted(os.listdir("data/raw")):
    if not fname.endswith(".zip"):
        continue
    zpath = f"data/raw/{fname}"
    asc_name = fname.replace(".zip", ".asc").lower()
    if os.path.exists(f"data/raw/{asc_name}"):
        print(f"已解压跳过: {fname}")
        continue
    print(f"解压 {fname}...", end=" ")
    try:
        z = zipfile.ZipFile(zpath)
        z.extractall("data/raw/")
        print(f"✓  {z.namelist()}")
    except Exception as e:
        print(f"✗  {e}")

# ── 汇总 ──────────────────────────────────────
print("\n" + "=" * 55)
print("data/raw/ 文件清单")
print("=" * 55)
for f in sorted(os.listdir("data/raw")):
    size = os.path.getsize(f"data/raw/{f}") // 1024 // 1024
    print(f"  {f}  ({size} MB)")