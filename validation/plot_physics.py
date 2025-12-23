import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
root_dir = r"/home/feiyuanzaixian/桌面/data"

# Normal baseline file (C172S)
normal_path = os.path.join(root_dir, "normal/C172S/cleaned_0001_filled.csv")

# Four anomaly files
paths = {
    'throttle': os.path.join(root_dir, "abnormal/accelerator_operation/clean_data_C172S/1.csv"),
    'engine':   os.path.join(root_dir, "abnormal/engine_power_loss/clean_data_C172S/1.csv"),
    'course':   os.path.join(root_dir, "abnormal/course_deviation/clean_data_C172S/1.csv"),
    'pitch':    os.path.join(root_dir, "abnormal/pitch_attitude/clean_data_C172S/1.csv")
}

# ==========================================
# 2. INTELLIGENT WINDOW SEARCH FUNCTION
# ==========================================
def find_best_window(df, pad=30):
    if 'label' not in df.columns: return 100, 200
    # Compatible with string or numeric labels
    is_anomaly = df['label'].astype(str).apply(lambda x: x.split('.')[0] != '0')
    indices = df.index[is_anomaly].to_numpy()
    if len(indices) == 0: return 100, 200
    # Simple clustering to find longest segment
    split_locs = np.where(np.diff(indices) > 10)[0] + 1
    segments = np.split(indices, split_locs)
    best = max(segments, key=len)
    return max(0, best[0]-pad), min(len(df), best[-1]+pad)

# ==========================================
# 3. DATA READING
# ==========================================
dfs = {}
windows = {}
try:
    df_norm = pd.read_csv(normal_path)
    for key, p in paths.items():
        if os.path.exists(p):
            dfs[key] = pd.read_csv(p)
            windows[key] = find_best_window(dfs[key])
            print(f"✅ Read {key}: window {windows[key]}")
        else:
            print(f"⚠️ File does not exist: {p} (will skip plotting)")
            dfs[key] = None
except Exception as e:
    print(f"❌ Reading error: {e}")

# ==========================================
# 4. PLOTTING (2x2 layout)
# ==========================================
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# --- (A) Throttle Surge (throttle) ---
ax = axes[0, 0]
if dfs['throttle'] is not None:
    s, e = windows['throttle']
    t = dfs['throttle'].index[s:e]
    # RPM (left axis)
    l1, = ax.plot(t, dfs['throttle']['E1 RPM'][s:e], 'r-', label='RPM (Action)', linewidth=2)
    ax.set_ylabel('RPM', color='r', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='r')
    # EGT (right axis)
    ax2 = ax.twinx()
    l2, = ax2.plot(t, dfs['throttle']['E1 EGT1'][s:e], 'b--', label='EGT (Delayed)', linewidth=2)
    ax2.set_ylabel('EGT (°F)', color='b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax.set_title('(A) Throttle Surge: Thermal Lag', fontweight='bold', loc='left')
    ax.legend(handles=[l1, l2], loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)

# --- (B) Engine Cooling (cooling) ---
ax = axes[0, 1]
if dfs['engine'] is not None:
    s, e = windows['engine']
    col = 'E1 CHT1' if 'E1 CHT1' in dfs['engine'].columns else dfs['engine'].columns[0]
    ax.plot(dfs['engine'].index[s:e], df_norm[col][s:e], 'gray', linestyle=':', label='Normal Baseline', linewidth=2)
    ax.plot(dfs['engine'].index[s:e], dfs['engine'][col][s:e], 'orange', label='Injected Fault', linewidth=2.5)
    ax.set_ylabel('Cylinder Head Temp (°F)', fontweight='bold')
    ax.set_title('(B) Engine Cooling Failure', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

# --- (C) Course Deviation (course - special handling: plot trajectory) ---
ax = axes[1, 0]
if dfs['course'] is not None:
    s, e = windows['course']
    # Plot Latitude vs Longitude (trajectory plot)
    ax.plot(df_norm['Longitude'][s:e], df_norm['Latitude'][s:e], 'gray', linestyle=':', label='Original Path', linewidth=2)
    ax.plot(dfs['course']['Longitude'][s:e], dfs['course']['Latitude'][s:e], 'purple', label='Deviated Path', linewidth=2.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(C) Flight Path Deviation (Trajectory)', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    # Maintain aspect ratio
    ax.axis('equal')

# --- (D) Pitch Excursion (pitch) ---
ax = axes[1, 1]
if dfs['pitch'] is not None:
    s, e = windows['pitch']
    t = dfs['pitch'].index[s:e]
    # Pitch (left axis)
    l1, = ax.plot(t, dfs['pitch']['Pitch'][s:e], 'g-', label='Pitch Attitude', linewidth=2)
    ax.set_ylabel('Pitch (deg)', color='g', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='g')
    # NormAc (right axis)
    ax2 = ax.twinx()
    l2, = ax2.plot(t, dfs['pitch']['NormAc'][s:e], 'm--', label='Normal Accel (G)', linewidth=2)
    ax2.set_ylabel('Load Factor (G)', color='m', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('(D) Pitch Excursion: Aerodynamic Coupling', fontweight='bold', loc='left')
    ax.legend(handles=[l1, l2], loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('Figure1_FullValidation.pdf', dpi=300)
print("🎉 4-in-1 final figure generated: Figure1_FullValidation.pdf")
plt.show()
