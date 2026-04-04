import pandas as pd
import sqlite3
import os
import glob

# 官方字段映射（完整验证版）
COL_MAP = {
    0:  "YEAR",
    1:  "MONTH",
    2:  "ORIGIN",
    5:  "ORIGIN_CITY",
    6:  "DEST",
    9:  "DEST_CITY",
    10: "CARRIER",
    13: "DISTANCE",
    14: "SERVICE_CLASS",
    16: "AIRCRAFT_TYPE",
    17: "AIRCRAFT_CONFIG",
    18: "DEPARTURES_PERFORMED",
    19: "DEPARTURES_SCHEDULED",
    21: "SEATS",
    22: "PASSENGERS",       
    26: "AIR_TIME",         
}

TARGET_CARRIERS = ["DL", "UA", "AA", "WN", "B6", "AS"]
TARGET_SERVICE  = ["F"]  # 定期客运

asc_files = sorted(glob.glob("data/raw/*.asc"))
print(f"找到 {len(asc_files)} 个文件\n")

all_dfs = []
for fpath in asc_files:
    fname = os.path.basename(fpath)
    print(f"读取 {fname}...", end=" ", flush=True)
    try:
        df = pd.read_csv(
            fpath, sep="|", header=None,
            on_bad_lines="skip", low_memory=False
        )
        df = df[df[10].isin(TARGET_CARRIERS)]
        df = df[df[14].isin(TARGET_SERVICE)]
        df = df[list(COL_MAP.keys())].copy()
        df.columns = list(COL_MAP.values())
        print(f"✓  {len(df):,} 行")
        all_dfs.append(df)
    except Exception as e:
        print(f"✗  {e}")

print("\n合并去重...")
df = pd.concat(all_dfs, ignore_index=True)
df = df.drop_duplicates()
df = df[df["YEAR"].between(2019, 2024)]
print(f"最终: {len(df):,} 行")

print("\n各航司各年记录数:")
print(df.groupby(["CARRIER","YEAR"]).size().unstack(fill_value=0))

print("\n数据验证:")
print(f"  SEATS范围:      {df['SEATS'].min()} - {df['SEATS'].max()}")
print(f"  PASSENGERS范围: {df['PASSENGERS'].min()} - {df['PASSENGERS'].max()}")
print(f"  DISTANCE范围:   {df['DISTANCE'].min()} - {df['DISTANCE'].max()} 英里")
print(f"  AIR_TIME范围:   {df['AIR_TIME'].min()} - {df['AIR_TIME'].max()} 分钟")
print(f"\n样本数据:")
print(df[df["CARRIER"]=="DL"].head(3).to_string())

# 存入SQLite
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/airline.db")
df.to_sql("flights_raw", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_cy ON flights_raw(CARRIER,YEAR,MONTH)")
conn.commit()
conn.close()
print(f"\n✓ 存入 data/airline.db，表: flights_raw，{len(df):,} 行")