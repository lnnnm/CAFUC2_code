import pandas as pd
import numpy as np
import os
import glob
import time
import warnings
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')


ROOT_DIR = "CAFUC2"
TARGET_MODEL = "C172S"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANOMALY_MAP = {
    "Throttle Surge": "accelerator_operation",
    "Course Deviation": "course_deviation",
    "Engine Cooling": "engine_power_loss",
    "Pitch Excursion": "pitch_attitude"
}

SEQ_LEN = 30
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3


# --- Model 1: LSTM Autoencoder (Reconstruction) ---
class LSTM_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTM_AE, self).__init__()
        self.seq_len = SEQ_LEN
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)  # h_n: [1, batch, hidden]
        # Repeat hidden state
        h_repeated = h_n.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
        rec, _ = self.decoder(h_repeated)
        return rec


# --- Model 2: Transformer Autoencoder (Reconstruction) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1)]


class TransformerAE(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x


# --- Model 3: Deep SVDD ---
class DeepSVDD_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(DeepSVDD_Net, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 16)  # Map to 16-dim sphere

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        feature = h_n[-1]
        return self.fc(feature)  # Output shape: [batch, 16]



def load_data(aircraft_model, anomaly_folder_name=None, max_files=20):
    data_list = []
    if anomaly_folder_name is None:
        search_path = os.path.join(ROOT_DIR, "normal_data", aircraft_model, "*.csv")
        is_normal = True
    else:
        sub_folder = f"clean_data_{aircraft_model}"
        search_path = os.path.join(ROOT_DIR, "abnormal_data", anomaly_folder_name, sub_folder, "*.csv")
        is_normal = False
    files = glob.glob(search_path)
    if not files: return None
    if len(files) > max_files: files = np.random.choice(files, max_files, replace=False)
    for f in files:
        try:
            df = pd.read_csv(f)
            if is_normal:
                df['label'] = 0
            else:
                if 'label' not in df.columns:
                    df['label'] = 1
                else:
                    df['label'] = df['label'].astype(str).apply(lambda x: 0 if x.split('.')[0] == '0' else 1)
            if len(df) > 5000: df = df.iloc[::2]
            data_list.append(df)
        except:
            pass
    if not data_list: return None
    full_df = pd.concat(data_list, ignore_index=True)
    drop_cols = ['label', 'Time', 'Date', 'Flight_ID', 'Unnamed: 0']
    feature_cols = [c for c in full_df.columns if c not in drop_cols]
    feature_cols = full_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return full_df[feature_cols], full_df['label']


def create_sequences(data, labels, seq_len):
    xs, ys = [], []
    step = 5
    for i in range(0, len(data) - seq_len, step):
        xs.append(data[i:(i + seq_len)])
        ys.append(1 if np.sum(labels[i:(i + seq_len)]) > 0 else 0)
    return np.array(xs), np.array(ys)


def get_metrics(y_true, y_scores):
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5
    best_f1, best_prec, best_rec = 0, 0, 0

    thresholds = np.percentile(y_scores, np.linspace(50, 99.5, 50))
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        if np.sum(y_pred) == 0: continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision_score(y_true, y_pred, zero_division=0)
            best_rec = recall_score(y_true, y_pred, zero_division=0)
    return auc, best_prec, best_rec, best_f1



print(f"🚀 Ultimate Benchmark (6 Models)")


print("1️⃣ Loading Training Data...")
df_train_X, _ = load_data(TARGET_MODEL, None, max_files=40)
train_cols = df_train_X.columns.tolist()

imputer = SimpleImputer(strategy='median')
scaler_std = StandardScaler()  # For Traditional
scaler_mm = MinMaxScaler()  # For Deep Learning

X_train_clean = imputer.fit_transform(df_train_X)
X_train_std = scaler_std.fit_transform(X_train_clean)
X_train_mm = scaler_mm.fit_transform(X_train_clean)

# OCSVM
if len(X_train_std) > 10000:
    idx = np.random.choice(len(X_train_std), 10000, replace=False)
    X_train_small = X_train_std[idx]
else:
    X_train_small = X_train_std


X_seq_train, _ = create_sequences(X_train_mm, np.zeros(len(X_train_mm)), SEQ_LEN)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq_train)), batch_size=BATCH_SIZE, shuffle=True)


res_trad = []
res_dl = []

# --- Traditional ---
print("\n🔥 Training Traditional Models...")
t0 = time.time()
m_if = IsolationForest(contamination=0.05, n_jobs=-1, random_state=RANDOM_SEED).fit(X_train_std)
t_if = time.time() - t0

t0 = time.time()
m_svm = OneClassSVM(kernel='rbf', nu=0.05).fit(X_train_small)
t_svm = time.time() - t0

t0 = time.time()
m_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1).fit(X_train_small)
t_lof = time.time() - t0

# --- Deep Learning ---
print("🔥 Training Deep Learning Models...")



def train_model(model, loader, mode='recon'):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    center = None

    # Deep SVDD Center Init
    if mode == 'svdd':
        model.eval()
        with torch.no_grad():
            # Init center c as mean of first batch
            for [bx] in loader:
                bx = bx.to(DEVICE)
                center = torch.mean(model(bx), dim=0)
                break
        model.train()

    t_start = time.time()
    for e in range(EPOCHS):
        for [bx] in loader:
            bx = bx.to(DEVICE)
            opt.zero_grad()
            out = model(bx)
            if mode == 'recon':
                loss = crit(out, bx)
            else:  # SVDD Loss: Distance to center
                loss = torch.mean(torch.sum((out - center) ** 2, dim=1))
            loss.backward()
            opt.step()
    return time.time() - t_start, center


# 1. LSTM-AE
m_lstm = LSTM_AE(input_dim=len(train_cols)).to(DEVICE)
t_lstm, _ = train_model(m_lstm, train_loader, 'recon')

# 2. Transformer
m_tf = TransformerAE(input_dim=len(train_cols)).to(DEVICE)
t_tf, _ = train_model(m_tf, train_loader, 'recon')

# 3. Deep SVDD
m_svdd = DeepSVDD_Net(input_dim=len(train_cols)).to(DEVICE)
t_svdd, svdd_center = train_model(m_svdd, train_loader, 'svdd')


print("\n📊 Evaluating...")
for nice_name, folder in ANOMALY_MAP.items():
    print(f"   -> {nice_name}...")
    res = load_data(TARGET_MODEL, folder, max_files=10)
    if res is None: continue
    X_test_raw, y_test = res
    X_test_raw = X_test_raw.reindex(columns=train_cols, fill_value=0)
    X_test_clean = imputer.transform(X_test_raw)

    # Traditional
    X_std = scaler_std.transform(X_test_clean)
    y_vals = y_test.values

    res_trad.append(("iForest", nice_name, *get_metrics(y_vals, -m_if.score_samples(X_std)), t_if))

    # OCSVM / LOF (Downsample for speed if needed)
    if len(X_std) > 5000:
        idx = np.arange(0, len(X_std), 5)
        X_sub, y_sub = X_std[idx], y_vals[idx]
    else:
        X_sub, y_sub = X_std, y_vals

    res_trad.append(("OCSVM", nice_name, *get_metrics(y_sub, -m_svm.score_samples(X_sub)), t_svm))
    res_trad.append(("LOF", nice_name, *get_metrics(y_sub, -m_lof.decision_function(X_sub)), t_lof))

    # Deep Learning
    X_mm = scaler_mm.transform(X_test_clean)
    X_seq_test, y_seq_test = create_sequences(X_mm, y_vals, SEQ_LEN)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq_test)), batch_size=BATCH_SIZE * 2, shuffle=False)


    def get_dl_scores(model, mode='recon', c=None):
        model.eval()
        scores = []
        with torch.no_grad():
            for [bx] in test_loader:
                bx = bx.to(DEVICE)
                out = model(bx)
                if mode == 'recon':
                    loss = torch.mean((out - bx) ** 2, dim=[1, 2])
                else:  # SVDD Distance
                    loss = torch.sum((out - c) ** 2, dim=1)
                scores.extend(loss.cpu().numpy())
        return np.array(scores)


    res_dl.append(("LSTM-AE", nice_name, *get_metrics(y_seq_test, get_dl_scores(m_lstm)), t_lstm))
    res_dl.append(("Transformer", nice_name, *get_metrics(y_seq_test, get_dl_scores(m_tf)), t_tf))
    res_dl.append(
        ("Deep SVDD", nice_name, *get_metrics(y_seq_test, get_dl_scores(m_svdd, 'svdd', svdd_center)), t_svdd))


def print_latex(data, title, label):
    print(f"\n% --- Table: {title} ---")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{|l|l|c|c|c|c|c|}")
    print(r"\hline")
    print(
        r"\textbf{Algorithm} & \textbf{Anomaly Type} & \textbf{AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Time (s)} \\")
    print(r"\hline")
    curr_algo = ""
    for row in data:
        algo, anom, auc, prec, rec, f1, tm = row
        algo_disp = f"\\multirow{{4}}{{*}}{{{algo}}}" if algo != curr_algo else ""
        if algo != curr_algo:
            print(r"\hline")
            curr_algo = algo
        print(f"{algo_disp} & {anom} & {auc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {tm:.1f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}%")
    print(r"}")
    print(f"\\caption{{{title}}}")
    print(f"\\label{{{label}}}")
    print(r"\end{table}")


print("\n" + "=" * 50)
print_latex(res_trad, "Baseline 1: Traditional point-based benchmarks.", "tab:trad")
print_latex(res_dl, "Baseline 2: Deep sequence-based benchmarks.", "tab:dl")
print("=" * 50)