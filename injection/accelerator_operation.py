"""
Throttle-burst anomaly injector (no-time, fixed)

"""

import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

# ========== 配置 ==========
# INPUT_FOLDERS = [
#     r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/C172R",
#     # r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/C172S",
#     # # "processed_data/DA42NG",
#     # r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/SR20",
#     r"C:\Users\DELL\Desktop\RTdetector-main (2)\processed_data/SR20G6",
# ]
# OUTPUT_ROOT = Path(r"C:\Users\DELL\Desktop\RTdetector-main (2)\abnormal_data\accelerator_operation")
# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

INPUT_FOLDERS = [
    "clean_data/SR20G6"
    # "normal_data/C172S",
    # # "processed_data/DA42NG",
    # "normal_data/SR20",
    # "normal_data/SR20G6",
]
OUTPUT_ROOT = Path("accelerator_operation/clean_data_SR20G6")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# 异常判据
RPM_RATE_THRESHOLD = 800.0  # rpm / s
SEGMENT_RATIO = 0.03        # 每条航迹抽 3 % 样本点作为异常段起点
SEG_MIN_SEC = 2
SEG_MAX_SEC = 5
SLOPE_MULT = (1.0, 1.5)

# 连锁反应系数
FFLOW_GAIN = 0.06
OILP_GAIN = 0.003
TEMP_GAIN = 0.02


RPM_COL   = "E1 RPM"
FFLOW_COL = "E1 FFlow"
OILP_COL  = "E1 OilT"
CHT_COLS  = ["E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4"]
EGT_COLS  = ["E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4"]


def ensure_numeric(df: pd.DataFrame, cols):
    """确保指定列为数值，并用 ffill/bfill 填充（最终缺失用0）"""
    existing = [c for c in cols if c in df.columns]
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if existing:
        df[existing] = df[existing].ffill().bfill().fillna(0).astype(float)


def pick_segments(df, ratio) -> list:
    """随机挑选若干段异常（返回 (start_idx, end_idx, slope) 列表）"""
    total = len(df)
    if total < 20:
        return []
    # 期望 segment 数（每段平均 ~5s），保底1段
    n_segments = max(1, int(total * ratio // 5))
    candidates = np.arange(max(0, total - 6))
    np.random.shuffle(candidates)
    segments = []
    for idx in candidates:
        if len(segments) >= n_segments:
            break
        seg_len = int(np.ceil(np.random.uniform(SEG_MIN_SEC, SEG_MAX_SEC)))
        end_idx = idx + seg_len - 1  # inclusive
        if end_idx >= total:
            continue
        # 避免重叠
        if any((s <= idx <= e) or (s <= end_idx <= e) for s, e, _ in segments):
            continue
        slope = RPM_RATE_THRESHOLD * np.random.uniform(*SLOPE_MULT)
        slope *= np.random.choice([-1, 1])
        segments.append((idx, end_idx, slope))
    return segments


def mark_label(df, rows, cols):
    """为指定行写入异常列号。cols是要标记的特征列索引列表"""
    if "label" not in df.columns:
        df["label"] = pd.Series(["0"] * len(df), dtype="string")  # ✅ 明确是字符串
    else:
        df["label"] = df["label"].astype("string").fillna("0")

    lab_idx = df.columns.get_loc("label")
    for r in rows:
        prev = str(df.iat[r, lab_idx])
        if prev == "0":
            df.loc[r, "label"] = ",".join(map(str, cols))

        else:
            # 如果已存在其他异常列号，追加（去重）
            existing = set(prev.split(","))
            existing.update(map(str, cols))
            df.iat[r, lab_idx] = ",".join(sorted(existing, key=int))

def apply_segment(df: pd.DataFrame, start: int, end: int, slope: float):
    total = len(df)
    seg_len = end - start + 1
    if seg_len <= 0:
        return

    rows = np.arange(start, end + 1)
    rpm_idx = df.columns.get_loc(RPM_COL)
    mark_cols = [rpm_idx]

    # ---- 修改数值 ----
    if RPM_COL in df.columns:
        rpm_delta = slope * np.arange(seg_len, dtype=float)
        orig = df.iloc[rows, rpm_idx].astype(float).values
        df.iloc[rows, rpm_idx] = orig + rpm_delta

        # 连锁变量
        if FFLOW_COL in df.columns:
            idx = df.columns.get_loc(FFLOW_COL)
            df.iloc[rows, idx] += FFLOW_GAIN * rpm_delta
            mark_cols.append(idx)
        if OILP_COL in df.columns:
            idx = df.columns.get_loc(OILP_COL)
            df.iloc[rows, idx] += OILP_GAIN * rpm_delta
            mark_cols.append(idx)

    # 写标签
    mark_label(df, rows, mark_cols)

    # ---- 温度延迟 ----
    delay = random.randint(2, 8)
    temp_start = end + 1 + delay
    temp_end = temp_start + seg_len - 1
    if temp_start < total:
        temp_end = min(temp_end, total - 1)
        temp_rows = np.arange(temp_start, temp_end + 1)
        if temp_rows.size > 0:
            temp_delta_val = TEMP_GAIN * float(rpm_delta[-1])
            temp_cols = []
            for col in [c for c in CHT_COLS + EGT_COLS if c in df.columns]:
                idx = df.columns.get_loc(col)
                df.iloc[temp_rows, idx] += temp_delta_val
                temp_cols.append(idx)
            mark_label(df, temp_rows, temp_cols)



# ========== 批处理入口 ==========
anomaly_meta = []
file_counter = 1  # <--- 修改点 1: 初始化计数器

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    if not input_dir.is_dir():
        print(f"⚠️  文件夹 {input_dir} 不存在，跳过")
        continue
    print(f"📂 正在处理: {input_dir}")
    prefix = input_dir.name

    for fname in tqdm(sorted(os.listdir(input_dir)), desc=f"{input_dir.name}"):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = input_dir / fname
        try:
            df = pd.read_csv(fpath, low_memory=False, on_bad_lines="skip")
            # strip 列名，避免奇怪空格导致列名对不上
            df.columns = df.columns.str.strip()

            # 检查是否有 RPM 列（必须）
            if RPM_COL not in df.columns:
                print(f"❌ 跳过 {fname}: 缺少必要列 {RPM_COL}")
                continue

            # 确保数值列存在并为 float
            numeric_cols = [RPM_COL, FFLOW_COL, OILP_COL, *CHT_COLS, *EGT_COLS]
            ensure_numeric(df, numeric_cols)

            if len(df) < 20:
                print(f"❌ 跳过 {fname}: 数据少于 20 行")
                continue

            # 初始化标签列
            if "label" not in df.columns:
                df["label"] = "0"
            df["label"] = df["label"].astype("string").fillna("0")

            # 选段并注入
            segs = pick_segments(df, SEGMENT_RATIO)
            for s, e, k in segs:
                apply_segment(df, s, e, k)
                anomaly_meta.append({
                    "file": f"{file_counter}.csv",
                    "original_file": fname,
                    "original_folder": str(input_dir),
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "slope_rpm_per_s": float(k),
                    "segment_duration_sec": int(e - s + 1)
                })

            out_path = OUTPUT_ROOT / f"{file_counter}.csv"
            df.to_csv(out_path, index=False)

            file_counter += 1

            print(f"✅ 已生成: {out_path}")

        except Exception as exc:
            print(f"❌ 处理 {fname} 失败: {exc}")
            traceback.print_exc()
            continue

# 保存元数据与配置
if anomaly_meta:
    pd.DataFrame(anomaly_meta).to_csv(OUTPUT_ROOT / "anomaly_segments.csv", index=False)
    print(f"\n📊 异常元数据已保存至: {OUTPUT_ROOT/'anomaly_segments.csv'}")
else:
    print("\n⚠️ 未生成任何异常数据")

with open(OUTPUT_ROOT / "config.json", "w", encoding="utf-8") as fp:
    json.dump({
        "rpm_rate_threshold": RPM_RATE_THRESHOLD,
        "segment_ratio": SEGMENT_RATIO,
        "segment_duration_sec": [SEG_MIN_SEC, SEG_MAX_SEC],
        "slope_multiplier": SLOPE_MULT,
        "fflow_gain": FFLOW_GAIN,
        "oilp_gain": OILP_GAIN,
        "temp_gain": TEMP_GAIN,
        "random_seed": MASTER_SEED,
        "processed_files": len(anomaly_meta)
    }, fp, indent=2)

print(f"\n✅ 全部完成。输出目录: {OUTPUT_ROOT}，共生成 {len(anomaly_meta)} 段异常")
