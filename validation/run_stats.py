import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 配置路径 (请确认你的路径正确)
# ==========================================
root_dir = r"/home/feiyuanzaixian/桌面/data"  # 修改为你真实的路径


# 为了保证能取到足够多的异常点，我们不仅读取 1.csv，如果不够会自动尝试 2.csv, 3.csv
def get_path(anomaly_type, filename="1.csv"):
    return os.path.join(root_dir, f"abnormal/{anomaly_type}/clean_data_C172S/{filename}")


base_path = os.path.join(root_dir, "normal/C172S/cleaned_0001_filled.csv")

# 定义任务列表: (ID, 显示名称, 文件夹名)
tasks = [
    (0, "Normal Baseline", None),
    (1, "Throttle Surge", "accelerator_operation"),  # 注意你的拼写
    (2, "Engine Cooling", "engine_power_loss"),
    (3, "Course Deviation", "course_deviation"),
    (4, "Pitch Excursion", "pitch_attitude")
]

# ==========================================
# 2. 读取并采样 (核心修改：增强了对少量数据的鲁棒性)
# ==========================================
X_list, y_list = [], []

print("🚀 开始读取数据...")

for lbl, name, folder in tasks:
    # 正常数据
    if lbl == 0:
        if os.path.exists(base_path):
            df = pd.read_csv(base_path)
            cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in cols if 'label' not in c.lower()]
            # 正常数据取 600 个点
            sample = df.sample(n=min(600, len(df)), random_state=42)
            X_list.append(sample[cols].fillna(0).values)
            y_list.append(np.ones(len(sample)) * lbl)
            print(f"✅ {name}: 读取了 {len(sample)} 个点")
        else:
            print(f"❌ 找不到正常文件: {base_path}")
        continue

    # 异常数据 (尝试读取 1.csv 到 5.csv，直到凑够至少 50 个点)
    collected_samples = []
    points_needed = 100  # 目标点数

    for i in range(1, 10):  # 尝试前10个文件
        p = get_path(folder, f"{i}.csv")
        if not os.path.exists(p): continue

        try:
            df = pd.read_csv(p)
            cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in cols if 'label' not in c.lower()]

            # 筛选异常点 (label != 0)
            if 'label' in df.columns:
                # 兼容字符串或数字
                is_abn = df['label'].astype(str).apply(lambda x: x.split('.')[0] != '0')
                abn_data = df[is_abn]

                if len(abn_data) > 0:
                    collected_samples.append(abn_data[cols].fillna(0).values)
                    if sum(len(x) for x in collected_samples) >= points_needed:
                        break  # 够了就停
        except:
            pass

    # 合并该类别的所有数据
    if len(collected_samples) > 0:
        X_cat = np.vstack(collected_samples)
        # 如果点太多，抽样一下防止画图太慢；如果点太少，就全用
        if len(X_cat) > 300:
            indices = np.random.choice(len(X_cat), 300, replace=False)
            X_cat = X_cat[indices]

        X_list.append(X_cat)
        y_list.append(np.ones(len(X_cat)) * lbl)
        print(f"✅ {name}: 成功提取 {len(X_cat)} 个异常点 (来自多个文件)")
    else:
        print(f"⚠️ {name}: 未找到任何异常标签点！(请检查 label 列)")

# 合并所有
if len(X_list) > 0:
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # ==========================================
    # 3. t-SNE 计算
    # ==========================================
    print("🔄 正在计算 t-SNE (请稍候)...")
    # 为了让少量的 Pitch 点也能显示出来，perplexity 设小一点
    n_samples = X.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_scaled)

    # ==========================================
    # 4. 绘图
    # ==========================================
    plt.style.use('default')
    plt.figure(figsize=(10, 8))

    # 定义颜色和标记
    # Normal, Throttle, Course, Engine, Pitch
    colors = ['lightgrey', '#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', '^', 's', 'D', 'v']

    # 按照任务列表的顺序画，保证图例正确
    for lbl, name, _ in tasks:
        mask = y == lbl
        if np.sum(mask) > 0:
            plt.scatter(X_emb[mask, 0], X_emb[mask, 1],
                        c=colors[lbl], label=name,
                        alpha=0.6 if lbl == 0 else 0.9,
                        s=30 if lbl == 0 else 60,  # 异常点画大一点，显眼
                        edgecolors='white', linewidth=0.5, marker=markers[lbl])

    plt.title('Feature Distribution Visualization (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9, loc='best')
    plt.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure2_Fixed.pdf', dpi=300)
    print("🎉 修复版分布图已生成: Figure2_Fixed.pdf (请检查是否包含 Pitch Excursion)")
    plt.show()
else:
    print("❌ 没有读取到有效数据，无法绘图。")