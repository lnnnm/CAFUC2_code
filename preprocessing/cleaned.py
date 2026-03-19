import pandas as pd
import numpy as np
import os


def clean_and_format_data(df, window_size=30, iqr_multiplier=3.0):

	df = df.copy()
	
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	
	for col in numeric_cols:
		df[col] = df[col].interpolate(method='linear', limit_direction='both')
		
		rolling_median = df[col].rolling(window=window_size, center=True, min_periods=1).median()
		rolling_q1 = df[col].rolling(window=window_size, center=True, min_periods=1).quantile(0.25)
		rolling_q3 = df[col].rolling(window=window_size, center=True, min_periods=1).quantile(0.75)
		rolling_iqr = rolling_q3 - rolling_q1
		
		lower_bound = rolling_median - iqr_multiplier * rolling_iqr
		upper_bound = rolling_median + iqr_multiplier * rolling_iqr
		
		spike_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
		df.loc[spike_mask, col] = rolling_median[spike_mask]
	
	if 'Latitude' in df.columns:
		df['Latitude'] = df['Latitude'].round(7)
	if 'Longitude' in df.columns:
		df['Longitude'] = df['Longitude'].round(7)
	
	other_numeric = [c for c in numeric_cols if c not in ['Latitude', 'Longitude', 'label']]
	df[other_numeric] = df[other_numeric].round(3)
	
	return df


def process_nested_folders(input_root, output_root):

	print(f"🚀 开始递归处理目录: '{input_root}'...")
	
	success_count = 0
	
	for dirpath, dirnames, filenames in os.walk(input_root):
		for filename in filenames:
			if filename.lower().endswith('.csv'):
				
				input_file_path = os.path.join(dirpath, filename)
				
				relative_path = os.path.relpath(dirpath, input_root)
				
				target_out_dir = os.path.join(output_root, relative_path)
				os.makedirs(target_out_dir, exist_ok=True)
				
				out_filename = filename if filename.startswith("cleaned_") else f"cleaned_{filename}"
				output_file_path = os.path.join(target_out_dir, out_filename)
				
				try:
					df = pd.read_csv(input_file_path, low_memory=False)
					
					if len(df) < 10:
						print(f"⚠️ 跳过 {os.path.join(relative_path, filename)}: 数据行数太少")
						continue
					
					df_clean = clean_and_format_data(df)
					df_clean.to_csv(output_file_path, index=False)
					
					success_count += 1
					print(f"✅ 成功: [{relative_path}] {filename} -> {out_filename}")
				
				except Exception as e:
					print(f"❌ 错误: 处理 {os.path.join(relative_path, filename)} 失败: {e}")
	
	print(f"\n🎉 全部处理完成！共成功清洗 {success_count} 个文件。输出目录: {output_root}")


if __name__ == '__main__':
	
	INPUT_FOLDER = 'raw_data'
	
	OUTPUT_FOLDER = 'normal_data'
	
	process_nested_folders(INPUT_FOLDER, OUTPUT_FOLDER)