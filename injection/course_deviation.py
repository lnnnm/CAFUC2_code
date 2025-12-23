import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

# ========== 1. CONFIGURATION ==========
INPUT_FOLDERS = [
    "clean_data/SR20G6"

]

OUTPUT_ROOT = Path("course_deviation/clean_data_SR20G6")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- Anomaly injection parameters ---
TARGET_RATIO = 0.03  # Anomaly ratio
WINDOW_SIZE = 30  # Length of each anomaly segment (seconds)

# 120 knots is a common cruise speed for training aircraft like C172/SR20
ASSUMED_GROUND_SPEED_KNOTS = 120.0
# data sampling interval is 1 second
SAMPLING_INTERVAL_SECONDS = 1.0


# ========== 2. HELPER FUNCTION: DEAD RECKONING ==========
def calculate_new_position(lat_deg, lon_deg, bearing_deg, distance_m):
    """
    Calculate new latitude/longitude coordinates based on starting point,
    bearing angle, and distance.
    Uses spherical model for estimation.
    """
    R = 6378137.0  # Earth radius (meters)

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)

    dist_rad = distance_m / R  # Angular distance

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(dist_rad) +
                            math.cos(lat_rad) * math.sin(dist_rad) * math.cos(bearing_rad))

    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(dist_rad) * math.cos(lat_rad),
                                       math.cos(dist_rad) - math.sin(lat_rad) * math.sin(new_lat_rad))

    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


# ========== 3. CORE INJECTION FUNCTION ==========
def inject_realistic_course_deviation(df: pd.DataFrame, num_windows: int):
    """
    Inject physically realistic course deviation anomalies into DataFrame.
    Labeling strategy modified: anomaly behavior marked as 1, normal behavior marked as 0.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "label" not in df.columns:
        df["label"] = 0
  
    df["label"] = df["label"].fillna(0).astype(int)

    
    numeric_cols = ['Latitude', 'Longitude', 'TRK', 'HDG', 'WptDst']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns]).reset_index(drop=True)
    n = len(df)

    anomaly_info_list = []
    if n <= WINDOW_SIZE * 2 or num_windows == 0:
        return df, anomaly_info_list

    distance_per_step_m = (ASSUMED_GROUND_SPEED_KNOTS * 0.514444) * SAMPLING_INTERVAL_SECONDS

    processed_indices = set()
    for _ in range(num_windows):
        start = np.random.randint(0, n - WINDOW_SIZE)
        end = start + WINDOW_SIZE

        if any(i in processed_indices for i in range(start, end + 1)):
            continue

        affected_cols = []
        trk_offset = np.random.uniform(15, 30) * np.random.choice([-1, 1])
        hdg_offset = trk_offset + np.random.uniform(-2, 2)

        if 'TRK' in df.columns:
            df.loc[start:end, 'TRK'] += trk_offset
            affected_cols.append(df.columns.get_loc('TRK'))
        if 'HDG' in df.columns:
            df.loc[start:end, 'HDG'] += hdg_offset
            affected_cols.append(df.columns.get_loc('HDG'))

        if 'Latitude' in df.columns and 'Longitude' in df.columns and 'TRK' in df.columns:
            for i in range(start, end + 1):
                prev_lat = df.loc[i - 1, 'Latitude']
                prev_lon = df.loc[i - 1, 'Longitude']
                current_bearing = df.loc[i, 'TRK']
                new_lat, new_lon = calculate_new_position(prev_lat, prev_lon, current_bearing, distance_per_step_m)
                df.loc[i, 'Latitude'] = new_lat
                df.loc[i, 'Longitude'] = new_lon
            affected_cols.extend([df.columns.get_loc('Latitude'), df.columns.get_loc('Longitude')])

        if 'WptDst' in df.columns:
            dist_increase = np.linspace(0, np.random.uniform(500, 1500), WINDOW_SIZE + 1)
            df.loc[start:end, 'WptDst'] += dist_increase
            affected_cols.append(df.columns.get_loc('WptDst'))

        df.loc[start:end, 'label'] = 1

        anomaly_info_list.append({
            "start_idx": int(start), "end_idx": int(end), "affected_cols": sorted(list(set(affected_cols)))
        })
        processed_indices.update(range(start, end + 1))

    return df, anomaly_info_list


# ========== 4. BATCH PROCESSING ==========
anomaly_meta = []
file_counter = 1  

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    plane_type = input_dir.name
    print(f"📂 Processing: {plane_type}")

    for file in tqdm(list(input_dir.glob("*.csv"))):
        try:
            df = pd.read_csv(file, low_memory=False)
            if not any(col.strip() in df.columns for col in ["Latitude", "Longitude", "TRK"]):
                print(f"⚠️ Skipping file missing critical navigation columns: {file.name}")
                continue

            n = len(df)
            num_windows = max(1, int((TARGET_RATIO * n) / WINDOW_SIZE))

            abnormal_df, info_list = inject_realistic_course_deviation(df, num_windows)

            out_path = OUTPUT_ROOT / f"{file_counter}.csv"
            abnormal_df.to_csv(out_path, index=False)

            for info in info_list:
                anomaly_meta.append({
                    "file": f"{file_counter}.csv",
                    "original_file": file.name,
                    "plane_type": plane_type,
                    **info
                })

            file_counter += 1  
        except Exception as e:
            print(f"❌ Failed to process file {file.name}: {e}")
            continue

# ========== 5. SAVE METADATA ==========
if anomaly_meta:
    meta_path = OUTPUT_ROOT / "anomaly_segments.csv"
    pd.DataFrame(anomaly_meta).to_csv(meta_path, index=False)

    config = {
        "target_ratio": TARGET_RATIO,
        "window_size": WINDOW_SIZE,
        "assumed_ground_speed_knots": ASSUMED_GROUND_SPEED_KNOTS,
        "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
        "total_segments": len(anomaly_meta)
    }
    with open(OUTPUT_ROOT / "config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)

    print(f"\n📊 Anomaly metadata saved to: {meta_path}")
    print(f"📄 Configuration saved to: {OUTPUT_ROOT / 'config.json'}")
else:
    print("\n⚠️ No anomaly data generated")

print("\n🎯 Physically realistic course deviation anomaly injection completed!")
