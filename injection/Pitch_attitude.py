"""
Pitch guard anomaly injector (no time dependency)
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


INPUT_FOLDERS = [
    "clean_data/SR20G6"
    # "normal_data/C172S",
    # # "processed_data/DA42NG",
    # "normal_data/SR20",
    # "normal_data/SR20G6",
]

OUTPUT_ROOT = Path("pitch_attitude/clean_data_SR20G6")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 1314
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# Anomaly conditions and parameters
PITCH_LOW = -15.0
PITCH_HIGH = 18.0
SEG_RATIO = 0.03         # Approximately 1.2% of data segments will have anomalies
SEG_DUR_SAMPLES = (1, 4)   # Segment duration in samples (equivalent to 1~4 seconds)
PITCH_ADD = (5, 15)        # ±5~15 degree change

# Chain reaction coefficients
NORM_GAIN = 0.08
LAT_GAIN = 0.03
VSPD_GAIN = 80
HDG_GAIN = 0.2

# Core columns
PITCH_COL = "Pitch"
NORM_COL = "NormAc"
LAT_COL = "LatAc"
VSPD_COL = "VSpd"
HDG_COL = "HDG"
SLIP_COL = "Slip"

# --------------------------------------------------------- #
# 2. FUNCTIONS
# --------------------------------------------------------- #
def to_numeric(df, cols):
    """Ensure specified columns are numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill()


def choose_segments(df) -> list[tuple[int, int, float]]:
    """Randomly select anomaly segments"""
    n = len(df)
    n_seg = max(1, int(n * SEG_RATIO // 5))
    segs = []
    for _ in range(n_seg * 2):  # Try multiple times to avoid overlap
        if len(segs) >= n_seg:
            break
        start = np.random.randint(0, n - 5)
        dur = np.random.randint(*SEG_DUR_SAMPLES)
        end = min(start + dur, n - 1)
        # Check for overlap
        if any((s <= start <= e) or (s <= end <= e) for s, e, *_ in segs):
            continue
        pitch_add = np.random.uniform(*PITCH_ADD) * np.random.choice([-1, 1])
        segs.append((start, end, pitch_add))
    return segs


def inject_segment(df, s, e, pitch_add):
    seg = slice(s, e + 1)
    df.loc[seg, PITCH_COL] += pitch_add

    # Ensure exceeding threshold
    df.loc[seg, PITCH_COL] = df.loc[seg, PITCH_COL].where(
        ~((df[PITCH_COL] > PITCH_LOW) & (df[PITCH_COL] < PITCH_HIGH)),
        PITCH_HIGH + 2 if pitch_add > 0 else PITCH_LOW - 2
    )

    affected_cols = []

    # Chain reaction & record affected column indices
    if NORM_COL in df.columns:
        df.loc[seg, NORM_COL] += NORM_GAIN * pitch_add
        affected_cols.append(df.columns.get_loc(NORM_COL))
    if SLIP_COL in df.columns and LAT_COL in df.columns:
        if df.loc[seg, SLIP_COL].abs().max() >= 1.0:
            df.loc[seg, LAT_COL] += LAT_GAIN * pitch_add * 0.5
            affected_cols.append(df.columns.get_loc(LAT_COL))
    if VSPD_COL in df.columns:
        df.loc[seg, VSPD_COL] += VSPD_GAIN * pitch_add
        affected_cols.append(df.columns.get_loc(VSPD_COL))
    if HDG_COL in df.columns:
        df.loc[seg, HDG_COL] += HDG_GAIN * pitch_add * (1 if pitch_add > 0 else 0.3)
        affected_cols.append(df.columns.get_loc(HDG_COL))

    # Current Pitch column is also anomalous
    affected_cols.append(df.columns.get_loc(PITCH_COL))

    # ====== Write label column ======
    df.loc[s:e + 1, "label"] = 1


# --------------------------------------------------------- #
# 3. MAIN LOOP
# --------------------------------------------------------- #
meta = []
file_counter = 1

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    if not input_dir.is_dir():
        print(f"⚠️ Skipping non-existent folder: {input_dir}")
        continue

    prefix = f"{input_dir.parent.name}_{input_dir.name}"
    print(f"📂 Processing: {input_dir}")

    for fname in tqdm(os.listdir(input_dir), desc=f"processing {input_dir.name}"):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = input_dir / fname
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.str.strip()
            if PITCH_COL not in df.columns:
                print(f"[WARN] Missing Pitch column, skipping: {fname}")
                continue

            # Convert to numeric
            numeric_cols = [PITCH_COL, NORM_COL, LAT_COL, VSPD_COL, HDG_COL, SLIP_COL]
            to_numeric(df, numeric_cols)
            df["label"] = 0
            df["label"] = df["label"].fillna(0).astype(int)


            # Randomly select and inject anomalies
            segs = choose_segments(df)
            for s, e, p_add in segs:
                inject_segment(df, s, e, p_add)
                meta.append({
                    "file": f"{file_counter}.csv",
                    "original_file": fname,
                    "original_folder": str(input_dir),
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "pitch_add": float(p_add),
                    "segment_length": int(e - s + 1),
                    "anomaly_type": "pitch_up" if p_add > 0 else "pitch_down"
                })

            # Save file
            out_path = OUTPUT_ROOT / f"{file_counter}.csv"

            df.to_csv(out_path, index=False)

            file_counter += 1
            print(f"✅ Generated: {out_path}")

        except Exception as e:
            print(f"❌ Failed to process file {fname}: {e}")


# --------------------------------------------------------- #
# 4. SAVE METADATA
# --------------------------------------------------------- #
if meta:
    meta_path = OUTPUT_ROOT / "pitch_guard_segments.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    print(f"\n📊 Anomaly metadata saved to: {meta_path}")

    with open(OUTPUT_ROOT / "config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "pitch_thresholds": {"low": PITCH_LOW, "high": PITCH_HIGH},
                "segment_ratio": SEG_RATIO,
                "segment_length_samples": SEG_DUR_SAMPLES,
                "pitch_add_range": PITCH_ADD,
                "gain": {
                    "normac": NORM_GAIN,
                    "latac": LAT_GAIN,
                    "vspd": VSPD_GAIN,
                    "hdg_gain": HDG_GAIN
                },
                "random_seed": MASTER_SEED,
                "processed_files_count": len({m['file'] for m in meta}),
                "total_anomaly_segments": len(meta)
            },
            fp,
            indent=2,
        )
    print(f"📄 Configuration saved to: {OUTPUT_ROOT / 'config.json'}")
else:
    print("\n⚠️ No anomaly data generated")

print("🎯 Pitch guard anomaly injection complete.")
