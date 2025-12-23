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

# ========== Configuration ==========

INPUT_FOLDERS = [
    "clean_data/SR20G6"
]
OUTPUT_ROOT = Path("accelerator_operation/clean_data_SR20G6")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# Anomaly criteria
RPM_RATE_THRESHOLD = 800.0  # rpm / s
SEGMENT_RATIO = 0.03        # 3% of data points in each track as anomaly segment starting points
SEG_MIN_SEC = 2
SEG_MAX_SEC = 5
SLOPE_MULT = (1.0, 1.5)

# Chain reaction coefficients
FFLOW_GAIN = 0.06
OILP_GAIN = 0.003
TEMP_GAIN = 0.02


RPM_COL   = "E1 RPM"
FFLOW_COL = "E1 FFlow"
OILP_COL  = "E1 OilT"
CHT_COLS  = ["E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4"]
EGT_COLS  = ["E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4"]


def ensure_numeric(df: pd.DataFrame, cols):
    """Ensure specified columns are numeric, fill with ffill/bfill (finally fill missing with 0)"""
    existing = [c for c in cols if c in df.columns]
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if existing:
        df[existing] = df[existing].ffill().bfill().fillna(0).astype(float)


def pick_segments(df, ratio) -> list:
    """Randomly select anomaly segments (return list of (start_idx, end_idx, slope))"""
    total = len(df)
    if total < 20:
        return []
    # Expected number of segments (average ~5s per segment), minimum 1 segment
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
        # Avoid overlap
        if any((s <= idx <= e) or (s <= end_idx <= e) for s, e, _ in segments):
            continue
        slope = RPM_RATE_THRESHOLD * np.random.uniform(*SLOPE_MULT)
        slope *= np.random.choice([-1, 1])
        segments.append((idx, end_idx, slope))
    return segments


def mark_label(df, rows, cols):
    """Write anomaly column indices for specified rows. cols is a list of feature column indices to mark"""
    if "label" not in df.columns:
        df["label"] = pd.Series(["0"] * len(df), dtype="string")  # ✅ Explicitly string type
    else:
        df["label"] = df["label"].astype("string").fillna("0")

    lab_idx = df.columns.get_loc("label")
    for r in rows:
        prev = str(df.iat[r, lab_idx])
        if prev == "0":
            df.loc[r, "label"] = ",".join(map(str, cols))

        else:
            # If other anomaly column indices already exist, append (deduplicate)
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

    if RPM_COL in df.columns:
        rpm_delta = slope * np.arange(seg_len, dtype=float)
        orig = df.iloc[rows, rpm_idx].astype(float).values
        df.iloc[rows, rpm_idx] = orig + rpm_delta

        # Chain reaction variables
        if FFLOW_COL in df.columns:
            idx = df.columns.get_loc(FFLOW_COL)
            df.iloc[rows, idx] += FFLOW_GAIN * rpm_delta
            mark_cols.append(idx)
        if OILP_COL in df.columns:
            idx = df.columns.get_loc(OILP_COL)
            df.iloc[rows, idx] += OILP_GAIN * rpm_delta
            mark_cols.append(idx)

    # Write label
    mark_label(df, rows, mark_cols)

    # ---- Temperature delay ----
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



# ========== Batch Processing Entry Point ==========
anomaly_meta = []
file_counter = 1  

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    if not input_dir.is_dir():
        print(f"⚠️  Folder {input_dir} does not exist, skipping")
        continue
    print(f"📂 Processing: {input_dir}")
    prefix = input_dir.name

    for fname in tqdm(sorted(os.listdir(input_dir)), desc=f"{input_dir.name}"):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = input_dir / fname
        try:
            df = pd.read_csv(fpath, low_memory=False, on_bad_lines="skip")
            # Strip column names to avoid mismatches due to strange spaces
            df.columns = df.columns.str.strip()

            # Check if RPM column exists (required)
            if RPM_COL not in df.columns:
                print(f"❌ Skipping {fname}: Missing required column {RPM_COL}")
                continue

            # Ensure numeric columns exist and are float
            numeric_cols = [RPM_COL, FFLOW_COL, OILP_COL, *CHT_COLS, *EGT_COLS]
            ensure_numeric(df, numeric_cols)

            if len(df) < 20:
                print(f"❌ Skipping {fname}: Less than 20 rows of data")
                continue

            # Initialize label column
            if "label" not in df.columns:
                df["label"] = "0"
            df["label"] = df["label"].astype("string").fillna("0")

            # Select segments and inject anomalies
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

            print(f"✅ Generated: {out_path}")

        except Exception as exc:
            print(f"❌ Failed to process {fname}: {exc}")
            traceback.print_exc()
            continue

# Save metadata and configuration
if anomaly_meta:
    pd.DataFrame(anomaly_meta).to_csv(OUTPUT_ROOT / "anomaly_segments.csv", index=False)
    print(f"\n📊 Anomaly metadata saved to: {OUTPUT_ROOT/'anomaly_segments.csv'}")
else:
    print("\n⚠️ No anomaly data generated")

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

print(f"\n✅ All tasks completed. Output directory: {OUTPUT_ROOT}, generated {len(anomaly_meta)} anomaly segments")
