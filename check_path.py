# check_path.py
import os
import pandas as pd
import numpy as np
import cv2

CSV = 'data/train.csv'
PATH_DIR = ''   # ← 你 train.py 里 --path_dir 传的值

df = pd.read_csv(CSV)
print("CSV columns:", df.columns.tolist())
print("Total rows:", len(df))
print()

# 打印前 5 行关键的列
key = 'path_path' if 'path_path' in df.columns else df.columns[0]
print(f"First 5 values of column '{key}':")
for i in range(5):
    print(f"  [{i}] {repr(df.iloc[i][key])}")
print()

# 检查前 20 个文件是否真的存在
fail = 0
for i in range(min(20, len(df))):
    p = str(df.iloc[i][key])
    paths = [x.strip() for x in p.split(';') if x.strip()]
    for sub in paths:
        # 尝试 3 种路径解析
        full_candidates = [
            sub,                                       # 原样
            os.path.join(PATH_DIR, os.path.basename(sub)) if PATH_DIR else None,
            os.path.join(os.path.dirname(CSV), sub),
        ]
        found = None
        for cand in full_candidates:
            if cand and os.path.exists(cand):
                found = cand
                break
        if found is None:
            fail += 1
            print(f"  [{i}] NOT FOUND: {repr(sub)}")
        else:
            img = cv2.imdecode(np.fromfile(found, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [{i}] cv2 FAILED to read: {found}")
                fail += 1
            else:
                print(f"  [{i}] OK {found}, "
                      f"shape={img.shape}, "
                      f"mean={img.mean():.1f}, std={img.std():.1f}")

print()
print(f"Total failures in first 20: {fail}")