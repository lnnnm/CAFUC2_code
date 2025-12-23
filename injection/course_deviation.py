import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

# ========== 1. 配置 ==========
INPUT_FOLDERS = [
    "clean_data/SR20G6"
    # "normal_data/C172S",
    # # "processed_data/DA42NG",
    # "normal_data/SR20",
    # "normal_data/SR20G6",
]

OUTPUT_ROOT = Path("course_deviation/clean_data_SR20G6")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- 异常注入参数 ---
TARGET_RATIO = 0.03  # 异常比例
WINDOW_SIZE = 30  # 每段异常长度（秒）

# --- 新增：物理模型参数 ---
# 假设一个合理的平均地速（单位：节 knots），因为原始数据没有提供
# 120节对于C172/SR20等训练飞机是一个常见的巡航速度
ASSUMED_GROUND_SPEED_KNOTS = 120.0
# 假设数据的采样间隔为1秒
SAMPLING_INTERVAL_SECONDS = 1.0


# ========== 2. 辅助函数：航位推算 ==========
def calculate_new_position(lat_deg, lon_deg, bearing_deg, distance_m):
    """
    根据起点、方位角和距离，计算新的经纬度坐标。
    使用球面模型进行估算。
    """
    R = 6378137.0  # 地球半径（米）

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)

    dist_rad = distance_m / R  # 角距离

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(dist_rad) +
                            math.cos(lat_rad) * math.sin(dist_rad) * math.cos(bearing_rad))

    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(dist_rad) * math.cos(lat_rad),
                                       math.cos(dist_rad) - math.sin(lat_rad) * math.sin(new_lat_rad))

    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


# ========== 3. 核心注入函数 ==========
def inject_realistic_course_deviation(df: pd.DataFrame, num_windows: int):
    """
    在 DataFrame 中注入物理上合理的航道偏移异常。
    标签策略已修改为：异常行为标记为1，正常行为标记为0。
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "label" not in df.columns:
        df["label"] = 0
    # 确保列是整数类型，并将可能存在的空值填充为0
    df["label"] = df["label"].fillna(0).astype(int)

    # 预处理数值列
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

        # --- 修改点2：将整个异常段落的标签设置为 1 ---
        df.loc[start:end, 'label'] = 1

        anomaly_info_list.append({
            "start_idx": int(start), "end_idx": int(end), "affected_cols": sorted(list(set(affected_cols)))
        })
        processed_indices.update(range(start, end + 1))

    return df, anomaly_info_list


# ========== 4. 批量处理 ==========
anomaly_meta = []
file_counter = 1  # 初始化文件计数器

for folder in INPUT_FOLDERS:
    input_dir = Path(folder)
    plane_type = input_dir.name
    print(f"📂 正在处理: {plane_type}")

    for file in tqdm(list(input_dir.glob("*.csv"))):
        try:
            df = pd.read_csv(file, low_memory=False)
            if not any(col.strip() in df.columns for col in ["Latitude", "Longitude", "TRK"]):
                print(f"⚠️ 跳过缺少关键导航列的文件: {file.name}")
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

            file_counter += 1  # 计数器自增
        except Exception as e:
            print(f"❌ 处理文件 {file.name} 失败: {e}")
            continue

# ========== 5. 保存元数据 ==========
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

    print(f"\n📊 异常元数据保存至: {meta_path}")
    print(f"📄 配置信息保存至: {OUTPUT_ROOT / 'config.json'}")
else:
    print("\n⚠️ 未生成任何异常数据")

print("\n🎯 物理上合理的航道偏移异常注入完成！")