"""
Cylinder Head Temperature (CHT) anomaly injector for 4-cylinder engines (no time dependency)
"""
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------- #
# 1. CONFIGURATION
# --------------------------------------------------------- #
# INPUT_FOLDERS = [
#     r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/C172R",
#     # r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/C172S",
#     # # "processed_data/DA42NG",
#     # r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/SR20",
#     r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/SR20G6",
# ]
# OUTPUT_ROOT = Path(r"C:\Users\DELL\Desktop\RTdetector-main (2)\abnormal_data\cht_drop")
# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

INPUT_FOLDERS = [
    "clean_data/C172S"
    # "normal_data/C172S",
    # # "processed_data/DA42NG",
    # "normal_data/SR20",
    # "normal_data/SR20G6",
]
OUTPUT_ROOT = Path("engine_power_loss/clean_data_C172S")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
# INPUT_FOLDERS = [
#     "processed_data/C172R",
#     "processed_data/C172S",
#     # "processed_data/DA42NG",
#     "processed_data/SR20",
#     "processed_data/SR20G6",
# ]
# OUTPUT_ROOT = Path("abnormal_data/cht_drop")
# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 1314
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# 气缸列
CHT_COLS = ["E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4"]

# 异常判定
SEG_RATIO_CHT = 0.02          # 异常段比例
SEG_LEN_POINTS = (30, 60)     # 异常段长度（以点计）
CHT_DROP = (1.5, 5.0)         # 每点骤降幅度 ℉

# 连锁反应
NORM_COL = "NormAc"
VSPD_COL = "VSpd"
NORM_GAIN_CHT = 0.01
VSPD_GAIN_CHT = 30



# --------------------------------------------------------- #
# 2. UTILITIES
# --------------------------------------------------------- #
def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    existing_cols = [c for c in cols if c in df.columns]
    df[existing_cols] = df[existing_cols].ffill()


# --------------------------------------------------------- #
# 3. CHOOSE CHT SEGMENTS
# --------------------------------------------------------- #
def choose_cht_segments(df) -> list[tuple[int, int, float]]:
    """随机挑选若干段异常 (按索引而非时间计算)"""
    total_points = len(df)
    if total_points < 100:
        return []

    n_seg = max(1, int(total_points * SEG_RATIO_CHT // np.mean(SEG_LEN_POINTS)))
    candidates = np.arange(total_points - max(SEG_LEN_POINTS))
    np.random.shuffle(candidates)

    segs = []
    for start in candidates:
        if len(segs) >= n_seg:
            break
        seg_len = np.random.randint(*SEG_LEN_POINTS)
        end = start + seg_len
        if end >= total_points:
            continue
        # 避免重叠
        if any((s <= start <= e) or (s <= end <= e) for s, e, *_ in segs):
            continue
        cht_drop = -np.random.uniform(*CHT_DROP)
        segs.append((start, end, cht_drop))
    return segs


# --------------------------------------------------------- #
# 4. INJECT CHT SEGMENT
# --------------------------------------------------------- #
def inject_cht_segment(df, s, e, cht_drop):
    seg_index = df.index[s:e + 1]
    n_points = len(seg_index)
    cht_series = np.linspace(0, cht_drop, n_points)

    # 随机选择1~4个气缸注入
    n_cyl = np.random.randint(1, 5)
    cols_to_drop = np.random.choice(CHT_COLS, size=n_cyl, replace=False)
    for col in cols_to_drop:
        if col in df.columns:
            df.loc[seg_index, col] += cht_series

    # 连锁反应
    avg_drop = cht_series.mean()
    if NORM_COL in df.columns:
        df.loc[seg_index, NORM_COL] += NORM_GAIN_CHT * avg_drop
    if VSPD_COL in df.columns:
        df.loc[seg_index, VSPD_COL] += VSPD_GAIN_CHT * avg_drop

    # 打标签
    df.loc[seg_index, "label"] = 1


# --------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------- #
meta = []
file_counter = 1

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    if not input_dir.is_dir():
        print(f"⚠️ 文件夹 {input_dir} 不存在，跳过")
        continue
    print(f"📂 正在处理: {input_dir}")

    prefix = f"{input_dir.parent.name}_{input_dir.name}"

    for fname in tqdm(os.listdir(input_dir), desc=f"processing {input_dir.name}"):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = input_dir / fname

        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.str.strip()
            existing_cht_cols = [c for c in CHT_COLS if c in df.columns]
            if not existing_cht_cols:
                print(f"[WARN] No CHT columns in {fname}, skipping.")
                continue

            numeric_cols = existing_cht_cols + [NORM_COL, VSPD_COL]
            to_numeric(df, numeric_cols)
            df["label"] = 0

            # 选择并注入异常段
            segs = choose_cht_segments(df)
            for s, e, cht_drop in segs:
                inject_cht_segment(df, s, e, cht_drop)
                meta.append(
                    dict(
                        file=f"{file_counter}.csv",
                        original_file=fname,
                        original_folder=str(input_dir),
                        start_idx=int(s),
                        end_idx=int(e),
                        cht_drop=float(cht_drop),
                        segment_length_points=int(e - s),
                    )
                )

            # 保存结果
            out_path = OUTPUT_ROOT / f"{file_counter}.csv"
            df.to_csv(out_path, index=False)

            file_counter += 1

            print(f"✅ 已生成异常文件: {out_path}")

        except Exception as e:
            print(f"❌ 处理 {fname} 失败: {str(e)}")
            continue

# --------------------------------------------------------- #
# 6. SAVE META
# --------------------------------------------------------- #
if meta:
    meta_path = OUTPUT_ROOT / "cht_drop_segments.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    with open(OUTPUT_ROOT / "config_cht_drop.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "segment_ratio": SEG_RATIO_CHT,
                "segment_length_points": SEG_LEN_POINTS,
                "cht_drop_range": CHT_DROP,
                "gain": {
                    "normac": NORM_GAIN_CHT,
                    "vspd": VSPD_GAIN_CHT,
                },
                "cht_columns": CHT_COLS,
                "random_seed": MASTER_SEED,
                "processed_files": len({item['file'] for item in meta}),
                "total_segments": len(meta)
            },
            fp,
            indent=2,
        )
    print(f"\n📊 异常元数据已保存至: {meta_path}")
else:
    print("\n⚠️ 未生成任何异常数据")

print(f"\n✅ CHT drop anomalies injected into {len({item['file'] for item in meta})} files.")
