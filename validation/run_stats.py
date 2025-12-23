import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
root_dir = r"/home/feiyuanzaixian/桌面/data" 

def get_path(anomaly_type, filename="1.csv"):
    return os.path.join(root_dir, f"abnormal/{anomaly_type}/clean_data_C172S/{filename}")


base_path = os.path.join(root_dir, "normal/C172S/cleaned_0001_filled.csv")

# Define task list: (ID, display name, folder name)
tasks = [
    (0, "Normal Baseline", None),
    (1, "Throttle Surge", "accelerator_operation"), 
    (2, "Engine Cooling", "engine_power_loss"),
    (3, "Course Deviation", "course_deviation"),
    (4, "Pitch Excursion", "pitch_attitude")
]

# ==========================================
# 2. READ AND SAMPLE DATA
# ==========================================
X_list, y_list = [], []

print("🚀 Starting data reading...")

for lbl, name, folder in tasks:
    # Normal data
    if lbl == 0:
        if os.path.exists(base_path):
            df = pd.read_csv(base_path)
            cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in cols if 'label' not in c.lower()]
            # Take 600 points from normal data
            sample = df.sample(n=min(600, len(df)), random_state=42)
            X_list.append(sample[cols].fillna(0).values)
            y_list.append(np.ones(len(sample)) * lbl)
            print(f"✅ {name}: read {len(sample)} points")
        else:
            print(f"❌ Cannot find normal file: {base_path}")
        continue

    # Anomaly data
    collected_samples = []
    points_needed = 100  # Target number of points

    for i in range(1, 10):  # Try first 10 files
        p = get_path(folder, f"{i}.csv")
        if not os.path.exists(p): continue

        try:
            df = pd.read_csv(p)
            cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in cols if 'label' not in c.lower()]

            # Filter anomaly points (label != 0)
            if 'label' in df.columns:
                # Compatible with string or numeric
                is_abn = df['label'].astype(str).apply(lambda x: x.split('.')[0] != '0')
                abn_data = df[is_abn]

                if len(abn_data) > 0:
                    collected_samples.append(abn_data[cols].fillna(0).values)
                    if sum(len(x) for x in collected_samples) >= points_needed:
                        break  # Stop if enough points collected
        except:
            pass

    # Merge all data for this category
    if len(collected_samples) > 0:
        X_cat = np.vstack(collected_samples)
        if len(X_cat) > 300:
            indices = np.random.choice(len(X_cat), 300, replace=False)
            X_cat = X_cat[indices]

        X_list.append(X_cat)
        y_list.append(np.ones(len(X_cat)) * lbl)
        print(f"✅ {name}: successfully extracted {len(X_cat)} anomaly points (from multiple files)")
    else:
        print(f"⚠️ {name}: No anomaly label points found! (Please check label column)")

# Merge all data
if len(X_list) > 0:
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # ==========================================
    # 3. t-SNE CALCULATION
    # ==========================================
    print("🔄 Computing t-SNE (please wait)...")
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_scaled)

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.style.use('default')
    plt.figure(figsize=(10, 8))

    # Define colors and markers
    # Normal, Throttle, Course, Engine, Pitch
    colors = ['lightgrey', '#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', '^', 's', 'D', 'v']

    # Plot in task list order to ensure correct legend
    for lbl, name, _ in tasks:
        mask = y == lbl
        if np.sum(mask) > 0:
            plt.scatter(X_emb[mask, 0], X_emb[mask, 1],
                        c=colors[lbl], label=name,
                        alpha=0.6 if lbl == 0 else 0.9,
                        s=30 if lbl == 0 else 60,  
                        edgecolors='white', linewidth=0.5, marker=markers[lbl])

    plt.title('Feature Distribution Visualization (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9, loc='best')
    plt.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure2_Fixed.pdf', dpi=300)
    print("🎉 Distribution plot generated: Figure2_Fixed.pdf (please check if Pitch Excursion is included)")
    plt.show()
else:
    print("❌ No valid data read, cannot plot.")
