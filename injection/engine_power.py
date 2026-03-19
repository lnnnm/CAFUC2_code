import os
import json
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_ROOT = Path("CAFUC2/normal_data")

OUTPUT_ROOT = Path("CAFUC2/abnormal_data/engine_power_loss")

MASTER_SEED = 1314
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

CHT_COLS = ["E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4"]

SEG_RATIO_CHT = 0.02
SEG_LEN_POINTS = (30, 60)

CHT_RISE = (10.0, 30.0)

NORM_COL = "NormAc"
VSPD_COL = "VSpd"
NORM_GAIN_CHT = -0.005
VSPD_GAIN_CHT = -15


def to_numeric(df, cols):
	for c in cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")
	existing_cols = [c for c in cols if c in df.columns]
	df[existing_cols] = df[existing_cols].ffill().fillna(0)


def choose_cht_segments(df) -> list:
	total_points = len(df)
	if total_points < 100: return []
	
	n_seg = max(1, int(total_points * SEG_RATIO_CHT // np.mean(SEG_LEN_POINTS)))
	candidates = np.arange(total_points - max(SEG_LEN_POINTS))
	np.random.shuffle(candidates)
	
	segs = []
	for start in candidates:
		if len(segs) >= n_seg: break
		seg_len = np.random.randint(*SEG_LEN_POINTS)
		end = start + seg_len
		if end >= total_points: continue
		if any((s <= start <= e) or (s <= end <= e) for s, e, _ in segs): continue
		
		cht_rise = np.random.uniform(*CHT_RISE)
		segs.append((start, end, cht_rise))
	return segs


def inject_cht_segment(df, s, e, cht_rise):
	seg_index = df.index[s:e + 1]
	n_points = len(seg_index)
	
	base_rise = np.linspace(0, cht_rise, n_points)
	noise = np.random.normal(0, 1.5, n_points)
	cht_series = base_rise + noise
	
	existing_cyls = [c for c in CHT_COLS if c in df.columns]
	if not existing_cyls: return
	n_cyl = np.random.randint(1, len(existing_cyls) + 1)
	cols_to_inject = np.random.choice(existing_cyls, size=n_cyl, replace=False)
	
	for col in cols_to_inject:
		df.loc[seg_index, col] += cht_series
	
	avg_rise = cht_series.mean()
	if NORM_COL in df.columns:
		df.loc[seg_index, NORM_COL] += NORM_GAIN_CHT * avg_rise
	if VSPD_COL in df.columns:
		df.loc[seg_index, VSPD_COL] += VSPD_GAIN_CHT * avg_rise
	
	df.loc[seg_index, "label"] = 1


meta = []

print(f"🚀 开始递归处理目录: '{INPUT_ROOT}'...")

for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
	if dirpath == str(INPUT_ROOT):
		continue
	
	aircraft_model = os.path.basename(dirpath)
	csv_files = [f for f in filenames if f.lower().endswith(".csv")]
	
	if not csv_files: continue
	
	print(f"\n📂 正在处理机型: {aircraft_model} ({len(csv_files)} 个文件)")
	
	target_out_dir = OUTPUT_ROOT / f"clean_data_{aircraft_model}"
	target_out_dir.mkdir(parents=True, exist_ok=True)
	
	for fname in tqdm(csv_files, desc=f"Injecting {aircraft_model}"):
		fpath = Path(dirpath) / fname
		
		try:
			df = pd.read_csv(fpath, low_memory=False)
			df.columns = df.columns.str.strip()
			
			existing_cht_cols = [c for c in CHT_COLS if c in df.columns]
			if not existing_cht_cols:
				continue
			
			to_numeric(df, existing_cht_cols + [NORM_COL, VSPD_COL])
			if "label" not in df.columns:
				df["label"] = 0
			
			segs = choose_cht_segments(df)
			for s, e, cht_rise in segs:
				inject_cht_segment(df, s, e, cht_rise)
				
				match = re.search(r'\d+', fname)
				file_id = str(int(match.group())) if match else fname.split('.')[0]
				
				meta.append({
					"Abnormal_File": f"{file_id}.csv",
					"Normal_File": fname,
					"Aircraft_Model": aircraft_model,
					"Start_Idx": int(s),
					"End_Idx": int(e),
					"Target_Rise_F": float(cht_rise)
				})
			
			if 'Latitude' in df.columns: df['Latitude'] = df['Latitude'].round(7)
			if 'Longitude' in df.columns: df['Longitude'] = df['Longitude'].round(7)
			other_cols = [c for c in df.columns if
			              c not in ['Latitude', 'Longitude', 'label', 'Time', 'Date', 'Flight_ID']]
			df[other_cols] = df[other_cols].round(3)
			
			match = re.search(r'\d+', fname)
			out_name = f"{int(match.group())}.csv" if match else fname
			
			df.to_csv(target_out_dir / out_name, index=False)
		
		except Exception as e:
			print(f"❌ 处理 {fname} 失败: {str(e)}")
			continue


if meta:
	meta_path = OUTPUT_ROOT / "anomaly_segments_engine.csv"
	pd.DataFrame(meta).to_csv(meta_path, index=False)
	print(f"\n📊 异常元数据已保存至: {meta_path}")
	print(
		f"✅ 全部完成！共处理并生成了 {len({item['Abnormal_File'] + item['Aircraft_Model'] for item in meta})} 个异常文件。")
else:
	print("\n⚠️ 未生成任何异常数据，请检查输入目录。")