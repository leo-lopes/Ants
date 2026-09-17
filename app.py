"""
Dashboard Interativo — Transporte Cooperativo de Formigas

PASSO 5-6: Streamlit + RDS AWS

Uso local:
    streamlit run app.py

Deploy em Streamlit Cloud:
    1. git push para GitHub
    2. Conectar GitHub em https://share.streamlit.io
    3. Pronto — live em https://seu-dashboard.streamlit.app

Credenciais:
    DATABASE_URL deve estar em secrets.toml ou variavel de ambiente
    Format: postgresql://user:password@host:port/dbname
"""

import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import pickle
import warnings
import torch
import torch.nn as nn
import boto3

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Formigas — Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pastas locais (modo de teste sem RDS/S3) -- coloque traj_radius.csv,
# fig3c.csv e ant_trajectories.csv em DATA_DIR, e os 9 arquivos de
# modelo em MODELS_DIR, para testar o dashboard inteiro localmente.
DATA_DIR = Path(st.secrets.get("LOCAL_DATA_DIR", "data") if hasattr(st, "secrets") else "data")
MODELS_DIR = Path(st.secrets.get("LOCAL_MODELS_DIR", "models") if hasattr(st, "secrets") else "models")


def get_data_mode():
    """
    'rds' se DATABASE_URL estiver configurada (secrets.toml ou env var),
    senao 'local' -- le os CSVs direto de DATA_DIR. Isso permite testar
    o dashboard inteiro sem precisar de RDS: basta copiar traj_radius.csv,
    fig3c.csv e ant_trajectories.csv (os mesmos arquivos do Google Drive)
    para a pasta 'data/' ao lado do app.py.
    """
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    return "rds" if db_url else "local"

# ============================================================
# CONEXAO RDS
# ============================================================

@st.cache_resource
def get_db_connection():
    """Conecta ao RDS PostgreSQL (cache para nao reconectar a cada rerun)"""
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    
    if not db_url:
        st.error("Variavel DATABASE_URL nao configurada")
        st.info("Configure em .streamlit/secrets.toml ou variavel de ambiente")
        st.stop()
    
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        # Testar conexao (text() e obrigatorio no SQLAlchemy 2.0 para strings SQL cruas)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        return engine
    except Exception as e:
        st.error(f"Erro ao conectar RDS: {e}")
        st.stop()

# ============================================================
# LOAD MODELOS
# ============================================================

@st.cache_resource
def load_models():
    """
    Carregar modelos da Fase 2 (RandomForest / regressao / KMeans) usados
    pela aba 'Predicoes ML'.

    Assim como os de rede neural, procura em MODELS_DIR e baixa do S3 se
    faltar -- necessario no Streamlit Cloud, onde o .gitignore mantem os
    binarios fora do repo.
    """
    models = {}

    if not ensure_models_local(LEGACY_MODEL_FILES, "Modelos ML (Fase 2)"):
        return models

    try:
        with open(MODELS_DIR / 'model1_classifier.pkl', 'rb') as f:
            models['classifier'] = pickle.load(f)
        with open(MODELS_DIR / 'model2_critical_point.pkl', 'rb') as f:
            models['regressor'] = pickle.load(f)
        with open(MODELS_DIR / 'model3_kmeans.pkl', 'rb') as f:
            models['kmeans'] = pickle.load(f)
    except Exception as e:
        # Nao so FileNotFoundError: incompatibilidade de versao do
        # scikit-learn entre o ambiente que gerou o pickle (Colab) e este
        # levanta ModuleNotFoundError/AttributeError -- sem este except
        # amplo, isso derrubaria o app inteiro, nao so esta aba.
        st.warning(
            f"Modelos ML (Fase 2) nao carregados: {type(e).__name__}: {e}. "
            "Se for incompatibilidade de versao, alinhe o scikit-learn do "
            "requirements.txt com o do Colab que gerou os .pkl."
        )
        return {}

    return models

# ============================================================
# ARQUITETURAS DOS MODELOS (Formiga Individual)
# ============================================================
# Devem bater exatamente com as classes usadas no treino
# (notebooks 04_lstm_3models.ipynb / 05_comparacoes_equacoes.ipynb),
# senao load_state_dict falha.

class AlignmentClassifier(nn.Module):
    """Modelo 1b: prediz sigma (lifter/vazio/puller) a partir de theta, nav, find, b."""
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class AnglePredictor(nn.Module):
    """Modelo 2: prediz phi (angulo de alinhamento) para sitios puller."""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze()


class MovementPredictor(nn.Module):
    """Modelo 3b: prediz deslocamento (dx, dy) ate o proximo frame."""
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


MODEL_FILES = [
    "model_1b_sigma_classifier.pt", "scaler_X_1b.pkl",
    "model_2_phi_predictor.pt", "scaler_X_2.pkl", "scaler_y_2.pkl",
    "model_3b_displacement.pt", "scaler_X_3b.pkl", "scaler_y_3b.pkl",
    "models_metadata.json",
]

# Modelos da Fase 2 (RandomForest / regressao polinomial / KMeans),
# usados pela aba "Predicoes ML". Ficam no MESMO prefixo models/ do S3.
LEGACY_MODEL_FILES = [
    "model1_classifier.pkl",
    "model2_critical_point.pkl",
    "model3_kmeans.pkl",
]


def ensure_models_local(arquivos, rotulo):
    """
    Garante que 'arquivos' existam em MODELS_DIR.

    Se faltar algum, baixa de s3://<bucket>/models/. Usado tanto pelos
    modelos da Fase 2 (.pkl) quanto pelos de rede neural (.pt) -- sem
    isso, uma pasta models/ vazia (como no Streamlit Cloud, onde o
    .gitignore exclui os binarios) deixaria a aba sem modelos mesmo com
    os arquivos presentes no bucket.

    Retorna True se todos os arquivos estao disponiveis localmente.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    missing = [f for f in arquivos if not (MODELS_DIR / f).exists()]

    if not missing:
        return True

    bucket = st.secrets.get("S3_BUCKET") or os.getenv("S3_BUCKET")
    aws_key = st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

    if not (bucket and aws_key and aws_secret):
        st.info(
            f"{rotulo}: nao encontrados em '{MODELS_DIR}/' "
            f"(faltando: {', '.join(missing)}) e credenciais S3 nao "
            "configuradas. Copie os arquivos do Drive para essa pasta, ou "
            "configure S3_BUCKET/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
        )
        return False

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name="us-east-1",
        )
        for fname in missing:
            s3.download_file(bucket, f"models/{fname}", str(MODELS_DIR / fname))
        return True
    except Exception as e:
        st.warning(f"{rotulo}: erro ao baixar do S3 ({e})")
        return False


@st.cache_resource
def load_ant_models():
    """
    Monta os 3 modelos de rede neural prontos para inferencia.

    Usa MODELS_DIR e, se faltar algo, baixa do S3 (ver ensure_models_local).
    Para teste local sem AWS: copie os 9 arquivos de
    Drive:.../ant-active-matter/models/ para MODELS_DIR (default "models/").

    So roda quando a aba 'Formiga Individual' ou 'Validacao do Modelo'
    e aberta pela primeira vez -- nao afeta o tempo de carga das abas
    de dados.
    """
    local_dir = MODELS_DIR

    if not ensure_models_local(MODEL_FILES, "Modelos de rede neural"):
        return None

    try:
        with open(local_dir / "models_metadata.json") as f:
            metadata = json.load(f)

        device = torch.device("cpu")

        feat_1b = metadata["model_1b_sigma_classifier"]["feature_cols"]
        model_1b = AlignmentClassifier(len(feat_1b), hidden_size=64, num_classes=3)
        model_1b.load_state_dict(torch.load(local_dir / "model_1b_sigma_classifier.pt", map_location=device))
        model_1b.eval()
        with open(local_dir / "scaler_X_1b.pkl", "rb") as f:
            scaler_X_1b = pickle.load(f)

        feat_2 = metadata["model_2_phi_predictor"]["feature_cols"]
        model_2 = AnglePredictor(len(feat_2), hidden_size=64)
        model_2.load_state_dict(torch.load(local_dir / "model_2_phi_predictor.pt", map_location=device))
        model_2.eval()
        with open(local_dir / "scaler_X_2.pkl", "rb") as f:
            scaler_X_2 = pickle.load(f)
        with open(local_dir / "scaler_y_2.pkl", "rb") as f:
            scaler_y_2 = pickle.load(f)

        feat_3b = metadata["model_3b_displacement"]["feature_cols"]
        model_3b = MovementPredictor(len(feat_3b), hidden_size=64, output_size=2)
        model_3b.load_state_dict(torch.load(local_dir / "model_3b_displacement.pt", map_location=device))
        model_3b.eval()
        with open(local_dir / "scaler_X_3b.pkl", "rb") as f:
            scaler_X_3b = pickle.load(f)
        with open(local_dir / "scaler_y_3b.pkl", "rb") as f:
            scaler_y_3b = pickle.load(f)

        return {
            "model_1b": model_1b, "scaler_X_1b": scaler_X_1b, "feat_1b": feat_1b,
            "model_2": model_2, "scaler_X_2": scaler_X_2, "scaler_y_2": scaler_y_2, "feat_2": feat_2,
            "model_3b": model_3b, "scaler_X_3b": scaler_X_3b, "scaler_y_3b": scaler_y_3b, "feat_3b": feat_3b,
        }
    except Exception as e:
        st.warning(f"Nao foi possivel carregar os modelos de rede neural: {e}")
        return None


def predict_ant(ant_models, theta, nav, find, b):
    """Roda os 3 modelos em sequencia para um unico ponto (theta, nav, find, b)."""
    values = {"theta": theta, "nav": nav, "find": find, "b": b, "phi": 0.0, "sigma": 0}

    X_1b = np.array([[values[c] for c in ant_models["feat_1b"]]])
    X_1b_scaled = ant_models["scaler_X_1b"].transform(X_1b)
    with torch.no_grad():
        logits = ant_models["model_1b"](torch.FloatTensor(X_1b_scaled))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    sigma_pred = int(np.argmax(probs)) - 1  # 0,1,2 -> -1,0,1

    X_2 = np.array([[values[c] for c in ant_models["feat_2"]]])
    X_2_scaled = ant_models["scaler_X_2"].transform(X_2)
    with torch.no_grad():
        phi_scaled = ant_models["model_2"](torch.FloatTensor(X_2_scaled)).numpy()
    phi_pred = float(ant_models["scaler_y_2"].inverse_transform(phi_scaled.reshape(-1, 1))[0][0])

    values["phi"] = phi_pred if sigma_pred == 1 else 0.0
    values["sigma"] = sigma_pred
    X_3b = np.array([[values[c] for c in ant_models["feat_3b"]]])
    X_3b_scaled = ant_models["scaler_X_3b"].transform(X_3b)
    with torch.no_grad():
        disp_scaled = ant_models["model_3b"](torch.FloatTensor(X_3b_scaled)).numpy()
    dx, dy = ant_models["scaler_y_3b"].inverse_transform(disp_scaled)[0]

    return {"probs": probs, "sigma_pred": sigma_pred, "phi_pred": phi_pred, "dx": float(dx), "dy": float(dy)}


# ============================================================
# QUERY FUNCOES
# ============================================================

@st.cache_data(ttl=300)  # Cache 5 min
def query_phase_transition(_engine, find_norm=None, n_ants=None):
    """Tabela phase_transition (RDS) ou fig3c.csv (local), com filtros opcionais."""
    if get_data_mode() == "local":
        path = DATA_DIR / "fig3c.csv"
        if not path.exists():
            st.error(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path).rename(columns={"N": "n_ants", "f_norm": "find_normalized"})

        if find_norm is not None:
            delta = 0.05
            df = df[(df["find_normalized"] >= find_norm - delta) & (df["find_normalized"] <= find_norm + delta)]
        if n_ants is not None:
            df = df[df["n_ants"] == n_ants]
        return df.sort_values("find_normalized")

    query = "SELECT * FROM phase_transition WHERE 1=1"
    params = []
    if find_norm is not None:
        query += " AND find_normalized >= %s AND find_normalized <= %s"
        delta = 0.05
        params.extend([find_norm - delta, find_norm + delta])
    if n_ants is not None:
        query += " AND n_ants = %s"
        params.append(n_ants)
    query += " ORDER BY find_normalized"

    try:
        # params precisa ser TUPLA (nao lista): pandas + SQLAlchemy 2.0 rejeita
        # lista simples com "List argument must consist only of tuples or dictionaries".
        # None quando nao ha filtro, para nao passar tupla vazia.
        return pd.read_sql(query, _engine, params=tuple(params) if params else None)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def query_simulations(_engine, radius=None):
    """Tabela simulations (RDS) ou traj_radius.csv (local), com filtro de raio."""
    if get_data_mode() == "local":
        path = DATA_DIR / "traj_radius.csv"
        if not path.exists():
            st.error(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        # traj_radius.csv e trajetoria bruta por frame (coluna 'speed'), nao
        # agregada -- alias para 'mean_speed' para bater com o schema RDS
        # (mesma granularidade que a tabela 'simulations' ja tem hoje: uma
        # linha por frame, nao por simulacao).
        df["mean_speed"] = df["speed"]

        if radius is not None:
            df = df[df["radius"] == radius]
        sort_cols = [c for c in ["radius", "run"] if c in df.columns]
        return df.sort_values(sort_cols) if sort_cols else df

    query = "SELECT * FROM simulations WHERE 1=1"
    params = []
    if radius is not None:
        query += " AND radius = %s"
        params.append(radius)
    # 'seed' NAO existe na tabela simulations (schema: radius, nav, find,
    # mean_speed, std_speed, source) -- ordenar por ela quebra a query.
    query += " ORDER BY radius, id"

    try:
        return pd.read_sql(query, _engine, params=tuple(params) if params else None)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def query_p_puller_curve(_engine):
    """P(puller) empirico, agrupado por find normalizado."""
    if get_data_mode() == "local":
        path = DATA_DIR / "ant_trajectories.csv"
        if not path.exists():
            st.error(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["find_normalized"] = (df["find"] / df["nav"]).round(2)
        grouped = df.groupby("find_normalized").agg(
            p_puller=("sigma", lambda s: (s == 1).mean()),
            n=("sigma", "size"),
        ).reset_index()
        return grouped.sort_values("find_normalized")

    query = """
        SELECT ROUND((find / nav)::numeric, 2) as find_normalized,
               AVG(CASE WHEN sigma = 1 THEN 1.0 ELSE 0.0 END) as p_puller,
               COUNT(*) as n
        FROM ant_trajectories
        GROUP BY find_normalized
        ORDER BY find_normalized
    """
    try:
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def query_phi_vs_theta(_engine):
    """Media de phi por bin de theta, so para sitios puller (sigma=1)."""
    if get_data_mode() == "local":
        path = DATA_DIR / "ant_trajectories.csv"
        if not path.exists():
            st.error(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = df[df["sigma"] == 1].copy()
        df["theta_bin"] = df["theta"].round(1)
        grouped = df.groupby("theta_bin").agg(
            phi_medio=("phi", "mean"), n=("phi", "size"),
        ).reset_index()
        return grouped.sort_values("theta_bin")

    query = """
        SELECT ROUND(theta::numeric, 1) as theta_bin,
               AVG(phi) as phi_medio, COUNT(*) as n
        FROM ant_trajectories
        WHERE sigma = 1
        GROUP BY theta_bin
        ORDER BY theta_bin
    """
    try:
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def query_displacement_sample(_engine, limit=3000):
    """
    Amostra aleatoria de pares (t -> t+1) por sitio -- mesma logica de
    pareamento usada no treino do Modelo 3b (notebook 05). No modo RDS,
    calculada no banco via LEAD() (window function), sem trazer a tabela
    inteira. No modo local, reproduz com groupby + shift no pandas.
    """
    if get_data_mode() == "local":
        path = DATA_DIR / "ant_trajectories.csv"
        if not path.exists():
            st.error(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df_sorted = df.sort_values(["run_id", "site_id", "t"])
        df_sorted["x_next"] = df_sorted.groupby(["run_id", "site_id"])["x"].shift(-1)
        df_sorted["y_next"] = df_sorted.groupby(["run_id", "site_id"])["y"].shift(-1)
        df_pairs = df_sorted.dropna(subset=["x_next", "y_next"]).copy()
        df_pairs["dx"] = df_pairs["x_next"] - df_pairs["x"]
        df_pairs["dy"] = df_pairs["y_next"] - df_pairs["y"]

        cols = ["theta", "phi", "sigma", "nav", "find", "b", "dx", "dy"]
        if len(df_pairs) > limit:
            df_pairs = df_pairs.sample(n=limit, random_state=42)
        return df_pairs[cols].reset_index(drop=True)

    query = f"""
        WITH pairs AS (
            SELECT theta, phi, sigma, nav, find, b, x, y,
                   LEAD(x) OVER (PARTITION BY run_id, site_id ORDER BY t) as x_next,
                   LEAD(y) OVER (PARTITION BY run_id, site_id ORDER BY t) as y_next
            FROM ant_trajectories
        )
        SELECT theta, phi, sigma, nav, find, b,
               (x_next - x) as dx, (y_next - y) as dy
        FROM pairs
        WHERE x_next IS NOT NULL
        ORDER BY random()
        LIMIT {limit}
    """
    try:
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()


# ============================================================
#  MODELOS DA FASE 2 -- INTROSPECCAO E FEATURES
# ============================================================
# Os .pkl foram treinados no notebook 02_ml_local_colab.ipynb com
# feature engineering (engineer_features_traj). Em vez de hardcodar a
# lista de features -- que muda conforme o CSV de origem tenha ou nao a
# coluna 'curvature' -- lemos feature_names_in_ do proprio pipeline em
# runtime. Assim a aba se adapta ao modelo que estiver carregado.

# Features derivadas das basicas, com a MESMA formula do notebook.
DERIVADAS_ML = {
    "puller_ratio": lambda v: v["n_pullers"] / (v["n_occupied"] + 1e-10),
    "lifter_ratio": lambda v: v["n_lifters"] / (v["n_occupied"] + 1e-10),
    "symmetry_breaking": lambda v: abs(v["n_pullers"] - v["n_lifters"]) / (v["n_occupied"] + 1e-10),
}

# Rotulos amigaveis + faixas default dos sliders (usadas se nao houver dados)
META_FEATURES = {
    "radius":     ("Raio da carga (b)",          0.5,  40.0, 4.0,  0.5),
    "speed":      ("Velocidade instantanea",     0.0,   1.5, 0.30, 0.01),
    "curvature":  ("Curvatura da trajetoria",    0.0,   5.0, 0.5,  0.05),
    "n_pullers":  ("Numero de pullers",          0,      60, 8,    1),
    "n_lifters":  ("Numero de lifters",          0,      60, 6,    1),
    "n_occupied": ("Sitios ocupados no total",   1,     120, 14,   1),
}


def nomes_features(obj):
    """Nomes das features que o estimador espera, na ordem correta."""
    nomes = getattr(obj, "feature_names_in_", None)
    if nomes is not None:
        return [str(n) for n in nomes]
    n = getattr(obj, "n_features_in_", None)
    return [f"f{i}" for i in range(n)] if n else []


def montar_vetor_features(nomes, valores_base):
    """
    Monta um DataFrame de 1 linha na ordem exata que o modelo espera,
    calculando as features derivadas a partir das basicas.
    """
    linha = dict(valores_base)
    for nome, formula in DERIVADAS_ML.items():
        if nome in nomes:
            linha[nome] = formula(linha)
    faltando = [n for n in nomes if n not in linha]
    for n in faltando:
        linha[n] = 0.0
    return pd.DataFrame([[linha[n] for n in nomes]], columns=nomes), faltando


COLS_TRAJ_ML = ["radius", "speed", "n_pullers", "n_lifters", "n_occupied"]


@st.cache_data(ttl=600)
def query_traj_features(_engine):
    """
    Trajetorias com as colunas que o Modelo 1 precisa, ja com as features
    derivadas calculadas. Retorna (df, origem) -- 'origem' explica de onde
    vieram os dados ou por que estao indisponiveis.

    Atencao: a tabela 'simulations' do RDS foi carregada pelo ETL apenas
    com colunas agregadas (radius, nav, find, mean_speed, std_speed) --
    n_pullers / n_lifters / n_occupied ficaram de fora. Por isso, no modo
    RDS os graficos de contexto desta aba podem ficar indisponiveis,
    enquanto a predicao interativa continua funcionando normalmente.
    """
    def enriquecer(df):
        X = df[[c for c in COLS_TRAJ_ML if c in df.columns]].copy()
        ocup = X["n_occupied"] + 1e-10
        X["puller_ratio"] = X["n_pullers"] / ocup
        X["lifter_ratio"] = X["n_lifters"] / ocup
        X["symmetry_breaking"] = (X["n_pullers"] - X["n_lifters"]).abs() / ocup
        return X.fillna(0)

    if get_data_mode() == "local":
        path = DATA_DIR / "traj_radius.csv"
        if not path.exists():
            return pd.DataFrame(), f"arquivo nao encontrado: {path}"
        df = pd.read_csv(path)
        faltando = [c for c in COLS_TRAJ_ML if c not in df.columns]
        if faltando:
            return pd.DataFrame(), f"faltam colunas em traj_radius.csv: {', '.join(faltando)}"
        return enriquecer(df), "traj_radius.csv (local)"

    try:
        amostra = pd.read_sql("SELECT * FROM simulations LIMIT 1", _engine)
        faltando = [c for c in COLS_TRAJ_ML if c not in amostra.columns]
        if faltando:
            return pd.DataFrame(), (
                f"a tabela 'simulations' no RDS nao tem: {', '.join(faltando)}. "
                "O ETL carregou apenas as colunas agregadas."
            )
        cols = ", ".join(COLS_TRAJ_ML)
        return enriquecer(pd.read_sql(f"SELECT {cols} FROM simulations", _engine)), "RDS"
    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================
# PAGINA PRINCIPAL
# ============================================================

st.title("Transporte Cooperativo de Formigas")
st.markdown("Dashboard interativo — Análise dos dados de simulação feito por (Gelblum et al., 2018)")

# Conectar (RDS) ou usar CSVs locais, dependendo de DATABASE_URL estar configurada
DATA_MODE = get_data_mode()

if DATA_MODE == "local":
    st.sidebar.info(
        f"Modo: **dados locais** ({DATA_DIR}/)\n\n"
        "Sem DATABASE_URL configurada — lendo traj_radius.csv, fig3c.csv "
        "e ant_trajectories.csv direto do disco."
    )
    engine = None
else:
    st.sidebar.success("Modo: **RDS** (nuvem)")
    engine = get_db_connection()

models = load_models()

# Abas principais
tab7,tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sobre o Projeto","Transição", "Raios", "Predições com ML", "Data",
    "Formiga Individual", "Validação do Modelo"
])

# ============================================================
# ABA 1: FIGURA 3c (FASE TRANSITION)
# ============================================================

with tab1:
    st.header("Fig. 3c — Transicao de Fase (Ordenado ↔ Desordenado)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # INTERATIVIDADE: Slider para find
        find_norm_user = st.slider(
            "Acoplamento normalizado (Find/N)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="find_slider"
        )
        
        # INTERATIVIDADE: Dropdown para N
        n_ants_options = [10, 20, 60]
        n_ants_user = st.selectbox(
            "Numero de formigas (N)",
            n_ants_options,
            key="n_ants_select"
        )
    
    with col2:
        st.info(
            f"Voce selecionou:\n"
            f"- Find normalizado: {find_norm_user:.2f}\n"
            f"- N formigas: {n_ants_user}\n"
            f"\nPonto critico teorico: Find = 0.5"
        )
    
    # QUERY dinamica (baseada nas selecoes do usuario)
    df_phase = query_phase_transition(engine, find_norm=find_norm_user, n_ants=n_ants_user)
    
    if not df_phase.empty:
        # GRAFICO 1: Todos os dados com destaque na selecao
        df_all = query_phase_transition(engine)
        
        fig = go.Figure()
        
        # Todos os pontos (fundo)
        for n in df_all['n_ants'].unique():
            df_n = df_all[df_all['n_ants'] == n]
            fig.add_trace(go.Scatter(
                x=df_n['find_normalized'],
                y=df_n['m_mean'],
                mode='markers',
                name=f"N={n}",
                opacity=0.3,
                marker=dict(size=5)
            ))
        
        # Selecao atual (destaque)
        fig.add_trace(go.Scatter(
            x=df_phase['find_normalized'],
            y=df_phase['m_mean'],
            mode='markers',
            name=f"SELECIONADO: N={n_ants_user}",
            marker=dict(size=10, color='red', symbol='star'),
            opacity=1.0
        ))
        
        # Ponto critico
        fig.add_vline(x=0.5, line_dash="dash", line_color="green", 
                     annotation_text="Ponto critico teorico")
        
        fig.update_layout(
            title=f"Transicao de Fase (N={n_ants_user})",
            xaxis_title="Find normalizado (F_ind/N)",
            yaxis_title="Parametro de ordem |m|",
            hovermode='closest',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ESTADISTICAS
        st.subheader("Estatisticas da Selecao")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valor medio de |m|", f"{df_phase['m_mean'].mean():.4f}")
        with col2:
            st.metric("Desvio padrao", f"{df_phase['m_std'].mean():.4f}")
        with col3:
            st.metric("Numero de registros", len(df_phase))
    else:
        st.warning("Nenhum dado encontrado para essa selecao")

# ============================================================
# ABA 2: FIGURA 2 (RAIOS)
# ============================================================

with tab2:
    st.header("Fig. 2 — Velocidade vs Raio da Carga")
    
    # Query todos os raios disponíveis
    df_sim = query_simulations(engine)
    
    if not df_sim.empty:
        # Agrupar por raio
        df_grouped = df_sim.groupby('radius').agg({
            'mean_speed': ['mean', 'std', 'count']
        }).reset_index()
        df_grouped.columns = ['radius', 'mean_speed', 'std_speed', 'n_samples']
        
        # GRAFICO: Velocidade vs Raio
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_grouped['radius'],
            y=df_grouped['mean_speed'],
            error_y=dict(
                type='data',
                array=df_grouped['std_speed'],
                visible=True
            ),
            mode='lines+markers',
            name='Velocidade media',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Velocidade Media vs Raio da Carga",
            xaxis_title="Raio (cm)",
            yaxis_title="Velocidade (cm/s)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com dados agregados
        st.subheader("Dados Agregados por Raio")
        st.dataframe(df_grouped, use_container_width=True)
    else:
        st.warning("Nenhum dado de trajetorias encontrado")

# ============================================================
# ABA 3: PREDICOES ML
# ============================================================

with tab3:
    st.header("Predicoes com Modelos ML")

    if not models:
        st.info(
            "Modelos da Fase 2 nao carregados. Coloque model1_classifier.pkl, "
            "model2_critical_point.pkl e model3_kmeans.pkl na pasta de modelos, "
            "ou configure as credenciais S3 para baixa-los automaticamente."
        )
    else:
        st.caption(
            "Tres modelos treinados sobre os dados de simulacao. Ajuste os "
            "controles e as predicoes atualizam na hora."
        )

        df_ctx, origem_ctx = query_traj_features(engine)

        sub1, sub2, sub3 = st.tabs([
            "Regime de movimento", "Ponto critico", "Clusters de comportamento",
        ])

        # ------------------------------------------------------------
        # MODELO 1 — Classificador de regime
        # ------------------------------------------------------------
        with sub1:
            clf = models.get("classifier")
            feats = nomes_features(clf)

            st.subheader("Classificador: ordenado ou desordenado")
            st.markdown(
                "RandomForest treinado para distinguir **movimento coordenado** "
                "(a carga avanca de forma balistica) de **cabo-de-guerra** "
                "(forcas se cancelam, movimento erratico). O rotulo de treino foi "
                "a velocidade estar acima ou abaixo da mediana."
            )

            with st.expander("Quais features o modelo usa"):
                base = [f for f in feats if f not in DERIVADAS_ML]
                deriv = [f for f in feats if f in DERIVADAS_ML]
                st.markdown(f"**{len(feats)} features**, nesta ordem: `{', '.join(feats)}`")
                st.markdown(
                    f"Voce controla as **{len(base)} basicas** nos sliders. As "
                    f"**{len(deriv)} derivadas** sao calculadas automaticamente "
                    "pelas mesmas formulas do treino:"
                )
                st.code(
                    "puller_ratio      = n_pullers / n_occupied\n"
                    "lifter_ratio      = n_lifters / n_occupied\n"
                    "symmetry_breaking = |n_pullers - n_lifters| / n_occupied",
                    language="text",
                )

            col_ctrl, col_res = st.columns([1, 1.4])

            with col_ctrl:
                st.markdown("**Configuracao da carga**")
                valores = {}
                for f in feats:
                    if f in DERIVADAS_ML:
                        continue
                    rotulo, vmin, vmax, vdef, passo = META_FEATURES.get(
                        f, (f, 0.0, 100.0, 1.0, 0.1)
                    )
                    # Se houver dados reais, ajusta a faixa ao que existe
                    if not df_ctx.empty and f in df_ctx.columns:
                        vmin = float(df_ctx[f].min())
                        vmax = float(df_ctx[f].max())
                        vdef = float(df_ctx[f].median())
                        if isinstance(passo, int):
                            vmin, vmax, vdef = int(vmin), int(max(vmax, vmin + 1)), int(vdef)
                    valores[f] = st.slider(
                        rotulo, vmin, vmax, vdef, step=passo, key=f"ml1_{f}"
                    )

                X_pred, ausentes = montar_vetor_features(feats, valores)

                st.markdown("**Derivadas (calculadas)**")
                derivadas_mostrar = {
                    f: float(X_pred.iloc[0][f]) for f in feats if f in DERIVADAS_ML
                }
                if derivadas_mostrar:
                    st.dataframe(
                        pd.DataFrame(
                            {"feature": list(derivadas_mostrar.keys()),
                             "valor": [f"{v:.4f}" for v in derivadas_mostrar.values()]}
                        ),
                        hide_index=True, use_container_width=True,
                    )
                if ausentes:
                    st.caption(f"Preenchidas com zero (sem controle): {', '.join(ausentes)}")

            with col_res:
                try:
                    classe = int(clf.predict(X_pred)[0])
                    proba = clf.predict_proba(X_pred)[0]
                    p_ordenado = float(proba[1])

                    if classe == 1:
                        st.success("ORDENADO — movimento coordenado, balistico")
                    else:
                        st.warning("DESORDENADO — cabo-de-guerra, movimento erratico")

                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_ordenado * 100,
                        number={"suffix": "%", "font": {"size": 36}},
                        title={"text": "Probabilidade de estar ordenado"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#C4622D"},
                            "steps": [
                                {"range": [0, 50], "color": "#EFEDE7"},
                                {"range": [50, 100], "color": "#DDE8E5"},
                            ],
                            "threshold": {
                                "line": {"color": "#232019", "width": 3},
                                "thickness": 0.8, "value": 50,
                            },
                        },
                    ))
                    fig_g.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
                    st.plotly_chart(fig_g, use_container_width=True)

                    m1, m2 = st.columns(2)
                    m1.metric("P(desordenado)", f"{proba[0]:.1%}")
                    m2.metric("P(ordenado)", f"{proba[1]:.1%}")

                except Exception as e:
                    st.error(f"Erro na predicao: {type(e).__name__}: {e}")
                    st.caption(f"O modelo espera estas features: {feats}")

            st.divider()

            # --- Analise de sensibilidade ---
            st.markdown("**Sensibilidade: como a predicao muda ao varrer uma feature**")
            st.caption(
                "Mantem todas as outras fixas nos valores acima e varre uma delas "
                "de ponta a ponta. Onde a curva cruza 50%, o modelo muda de opiniao."
            )

            base_feats = [f for f in feats if f not in DERIVADAS_ML]
            f_varrer = st.selectbox(
                "Feature a varrer", base_feats,
                index=base_feats.index("speed") if "speed" in base_feats else 0,
                key="ml1_sweep",
            )

            try:
                rot, vmin_s, vmax_s, _, _ = META_FEATURES.get(
                    f_varrer, (f_varrer, 0.0, 100.0, 1.0, 0.1)
                )
                if not df_ctx.empty and f_varrer in df_ctx.columns:
                    vmin_s, vmax_s = float(df_ctx[f_varrer].min()), float(df_ctx[f_varrer].max())

                grade = np.linspace(vmin_s, vmax_s, 80)
                linhas = []
                for g in grade:
                    v = dict(valores)
                    v[f_varrer] = g
                    Xg, _ = montar_vetor_features(feats, v)
                    linhas.append(Xg.iloc[0].values)
                X_sweep = pd.DataFrame(linhas, columns=feats)
                probs_sweep = clf.predict_proba(X_sweep)[:, 1]

                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=grade, y=probs_sweep, mode="lines", name="P(ordenado)",
                    line=dict(color="#C4622D", width=3),
                ))
                fig_s.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                annotation_text="fronteira de decisao")
                fig_s.add_vline(x=valores[f_varrer], line_dash="dot", line_color="#3D6B63",
                                annotation_text="valor atual")
                fig_s.update_layout(
                    xaxis_title=rot, yaxis_title="P(ordenado)",
                    yaxis=dict(range=[0, 1]), height=340, showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.warning(f"Nao foi possivel montar a varredura: {e}")

            # --- Importancia das features ---
            try:
                rf = clf.named_steps.get("rf") if hasattr(clf, "named_steps") else None
                if rf is not None and hasattr(rf, "feature_importances_"):
                    st.markdown("**O que o modelo mais olha**")
                    imp = pd.DataFrame({
                        "feature": feats,
                        "importancia": rf.feature_importances_,
                    }).sort_values("importancia")

                    fig_i = go.Figure(go.Bar(
                        x=imp["importancia"], y=imp["feature"], orientation="h",
                        marker_color="#3D6B63",
                    ))
                    fig_i.update_layout(
                        xaxis_title="Importancia (RandomForest)",
                        height=max(240, 34 * len(feats)),
                        margin=dict(l=10, r=10, t=10, b=40),
                    )
                    st.plotly_chart(fig_i, use_container_width=True)
                    st.caption(
                        f"Feature dominante: **{imp.iloc[-1]['feature']}** "
                        f"({imp.iloc[-1]['importancia']:.1%} da decisao). "
                        "Vale lembrar que o rotulo de treino foi derivado da "
                        "propria velocidade — entao 'speed' aparecer no topo e "
                        "esperado, nao uma descoberta."
                    )
            except Exception as e:
                st.caption(f"Importancia indisponivel: {e}")

        # ------------------------------------------------------------
        # MODELO 2 — Ponto critico
        # ------------------------------------------------------------
        with sub2:
            reg = models.get("regressor")

            st.subheader("Detector do ponto critico")
            st.markdown(
                "Regressao polinomial (grau 3 + Ridge) ajustada ao parametro de "
                "ordem **|m|** em funcao do acoplamento normalizado **F_ind**. "
                "A teoria preve a transicao em F_ind = 0.5."
            )

            try:
                # Range real dos dados de treino -- fora dele o polinomio de
                # grau 3 extrapola e a derivada dispara nas bordas, o que
                # produziria um "ponto critico" que e puro artefato.
                df_fase = query_phase_transition(engine)
                if not df_fase.empty:
                    f_min = float(df_fase["find_normalized"].min())
                    f_max = float(df_fase["find_normalized"].max())
                else:
                    f_min, f_max = 0.0, 1.0

                grade_f = np.linspace(0.0, 1.05, 300).reshape(-1, 1)
                m_pred = reg.predict(grade_f)
                d1 = np.gradient(m_pred, grade_f.ravel())

                dentro = (grade_f.ravel() >= f_min) & (grade_f.ravel() <= f_max)
                d1_dentro = d1[dentro]
                f_dentro = grade_f.ravel()[dentro]

                idx_incl = int(np.argmax(np.abs(d1_dentro)))
                f_max_incl = float(f_dentro[idx_incl])

                # So faz sentido chamar de "maior inclinacao" se o maximo for
                # INTERNO. Se cair colado numa das bordas do intervalo, e a
                # curva simplesmente subindo/descendo ate o limite dos dados --
                # nao um ponto de transicao. Margem de 8% de cada lado.
                margem = 0.08 * (f_max - f_min)
                maximo_interno = (f_max_incl > f_min + margem) and (f_max_incl < f_max - margem)

                col_a, col_b = st.columns([1, 2])

                with col_a:
                    f_escolhido = st.slider(
                        "F_ind normalizado", 0.0, 1.05, 0.5, step=0.01, key="ml2_find"
                    )
                    m_no_ponto = float(reg.predict(np.array([[f_escolhido]]))[0])
                    st.metric("|m| previsto", f"{m_no_ponto:.4f}")

                    if maximo_interno:
                        st.metric(
                            "Maior inclinacao", f"F = {f_max_incl:.3f}",
                            delta=f"{f_max_incl - 0.5:+.3f} vs teoria (0.5)",
                        )
                        st.caption(
                            "Ponto onde |m| muda mais rapido dentro do intervalo "
                            "coberto pelos dados. Heuristica, nao um ajuste de "
                            "ponto critico propriamente dito."
                        )
                    else:
                        st.warning("Sem transicao abrupta detectavel")
                        st.caption(
                            f"No intervalo dos dados (F de {f_min:.2f} a {f_max:.2f}), "
                            "a inclinacao maxima da curva ajustada cai colada na "
                            "borda, nao num ponto interno. Ou seja: o ajuste "
                            "polinomial nao exibe uma transicao abrupta que a "
                            "derivada consiga localizar. Cravar um numero aqui "
                            "seria artefato do ajuste, nao fisica. Compare a curva "
                            "laranja com os pontos reais ao lado — o R² modesto "
                            "deste modelo (ver notebook 02) e consistente com isso."
                        )

                with col_b:
                    fig_c = go.Figure()
                    if not df_fase.empty:
                        for n in sorted(df_fase["n_ants"].unique()):
                            sub = df_fase[df_fase["n_ants"] == n]
                            fig_c.add_trace(go.Scatter(
                                x=sub["find_normalized"], y=sub["m_mean"],
                                mode="markers", name=f"N = {n}",
                                marker=dict(size=7, opacity=0.65),
                            ))
                    fig_c.add_trace(go.Scatter(
                        x=grade_f.ravel(), y=m_pred, mode="lines",
                        name="Modelo polinomial",
                        line=dict(color="#C4622D", width=3.5),
                    ))
                    fig_c.add_vline(x=0.5, line_dash="dash", line_color="gray",
                                    annotation_text="teoria")
                    if f_max < 1.05:
                        fig_c.add_vrect(
                            x0=f_max, x1=1.05, fillcolor="gray", opacity=0.12,
                            line_width=0, annotation_text="extrapolacao",
                            annotation_position="top left",
                        )
                    fig_c.add_trace(go.Scatter(
                        x=[f_escolhido], y=[m_no_ponto], mode="markers",
                        name="seu ponto",
                        marker=dict(size=15, color="#3D6B63", symbol="diamond"),
                    ))
                    fig_c.update_layout(
                        xaxis_title="F_ind normalizado",
                        yaxis_title="Parametro de ordem |m|",
                        height=420, legend=dict(orientation="h", y=-0.2),
                    )
                    st.plotly_chart(fig_c, use_container_width=True)

                with st.expander("Derivada da curva ajustada"):
                    st.caption(
                        "A area cinza marca a regiao onde o modelo esta "
                        "extrapolando para fora dos dados de treino. Picos de "
                        "derivada ali nao significam nada fisicamente."
                    )
                    fig_d = go.Figure(go.Scatter(
                        x=grade_f.ravel(), y=d1, mode="lines",
                        line=dict(color="#3D6B63", width=2.5),
                    ))
                    if f_max < 1.05:
                        fig_d.add_vrect(x0=f_max, x1=1.05, fillcolor="gray",
                                        opacity=0.12, line_width=0)
                    fig_d.add_hline(y=0, line_color="gray", line_width=1)
                    fig_d.update_layout(
                        xaxis_title="F_ind normalizado", yaxis_title="d|m| / dF_ind",
                        height=280, showlegend=False,
                    )
                    st.plotly_chart(fig_d, use_container_width=True)

            except Exception as e:
                st.error(f"Erro no Modelo 2: {type(e).__name__}: {e}")

        # ------------------------------------------------------------
        # MODELO 3 — Clusters
        # ------------------------------------------------------------
        with sub3:
            st.subheader("Clusters de comportamento")
            st.markdown(
                "KMeans agrupando as trajetorias por comportamento dinamico, "
                "sem usar rotulo nenhum. Cada cluster e um regime que o sistema "
                "visita."
            )

            km_obj = models.get("kmeans")
            # No notebook, o modelo 3 foi salvo como TUPLA (kmeans, scaler)
            if isinstance(km_obj, (tuple, list)) and len(km_obj) == 2:
                kmeans, scaler_km = km_obj
            else:
                kmeans, scaler_km = km_obj, None

            feats_km = nomes_features(scaler_km) if scaler_km is not None else nomes_features(kmeans)

            if not feats_km:
                st.warning("Nao foi possivel identificar as features do KMeans.")
            else:
                st.caption(
                    f"K = {getattr(kmeans, 'n_clusters', '?')} clusters | "
                    f"features: `{', '.join(feats_km)}`"
                )

                if df_ctx.empty:
                    st.info(
                        f"Graficos de contexto indisponiveis: {origem_ctx}. "
                        "A atribuicao de cluster do seu ponto (abaixo) continua "
                        "funcionando."
                    )
                else:
                    faltam_km = [f for f in feats_km if f not in df_ctx.columns]
                    if faltam_km:
                        st.info(f"Faltam colunas para o grafico: {', '.join(faltam_km)}")
                    else:
                        amostra = df_ctx.sample(
                            n=min(4000, len(df_ctx)), random_state=42
                        ).copy()
                        # DataFrame (nao .values): o scaler foi ajustado com
                        # nomes de coluna, e passar array cru dispara
                        # UserWarning do sklearn a cada rerun
                        X_km = amostra[feats_km]
                        if scaler_km is not None:
                            X_km = scaler_km.transform(X_km)
                        amostra["cluster"] = kmeans.predict(X_km).astype(str)

                        eixo_x = st.selectbox("Eixo X", feats_km, index=0, key="ml3_x")
                        eixo_y = st.selectbox(
                            "Eixo Y", feats_km,
                            index=1 if len(feats_km) > 1 else 0, key="ml3_y",
                        )

                        fig_k = go.Figure()
                        for c in sorted(amostra["cluster"].unique(), key=int):
                            sub = amostra[amostra["cluster"] == c]
                            fig_k.add_trace(go.Scatter(
                                x=sub[eixo_x], y=sub[eixo_y], mode="markers",
                                name=f"Cluster {c}",
                                marker=dict(size=5, opacity=0.55),
                            ))
                        fig_k.update_layout(
                            xaxis_title=eixo_x, yaxis_title=eixo_y, height=440,
                            legend=dict(orientation="h", y=-0.2),
                        )
                        st.plotly_chart(fig_k, use_container_width=True)

                        st.markdown("**Perfil medio de cada cluster**")
                        perfil = amostra.groupby("cluster")[feats_km].mean().round(4)
                        perfil["n_amostras"] = amostra.groupby("cluster").size()
                        st.dataframe(perfil, use_container_width=True)

                st.divider()
                st.markdown("**Em qual cluster cai a sua configuracao?**")
                st.caption("Usa os mesmos valores definidos na aba 'Regime de movimento'.")

                try:
                    valores_km = {}
                    for f in feats_km:
                        if f in DERIVADAS_ML:
                            continue
                        valores_km[f] = st.session_state.get(f"ml1_{f}")

                    base_ml1 = {
                        k.replace("ml1_", ""): v
                        for k, v in st.session_state.items()
                        if k.startswith("ml1_") and k != "ml1_sweep"
                    }

                    if base_ml1:
                        X_user, _ = montar_vetor_features(feats_km, base_ml1)
                        Xu = X_user
                        if scaler_km is not None:
                            Xu = scaler_km.transform(X_user)
                        cluster_user = int(kmeans.predict(Xu)[0])

                        cc1, cc2 = st.columns([1, 2])
                        cc1.metric("Cluster atribuido", f"{cluster_user}")
                        with cc2:
                            st.dataframe(
                                pd.DataFrame({
                                    "feature": feats_km,
                                    "valor": [f"{float(X_user.iloc[0][f]):.4f}" for f in feats_km],
                                }),
                                hide_index=True, use_container_width=True,
                            )
                    else:
                        st.caption("Abra a aba 'Regime de movimento' primeiro para definir os valores.")
                except Exception as e:
                    st.warning(f"Nao foi possivel atribuir cluster: {type(e).__name__}: {e}")

# ============================================================
# ABA 4: RAW DATA
# ============================================================

with tab4:
    st.header("Dados Brutos")
    
    data_table = st.selectbox("Selecionar tabela", ["phase_transition", "simulations"])
    
    if data_table == "phase_transition":
        df = query_phase_transition(engine)
    else:
        df = query_simulations(engine)
    
    st.subheader(f"Tabela: {data_table}")
    st.write(f"Total de registros: {len(df)}")
    st.dataframe(df, use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {data_table}.csv",
        data=csv,
        file_name=f"{data_table}.csv",
        mime="text/csv"
    )

# ============================================================
# ABA 5: FORMIGA INDIVIDUAL (rede neural ao vivo)
# ============================================================

with tab5:
    st.header("Formiga Individual — Previsao por Rede Neural")
    st.caption(
        "3 modelos MLP (sem vazamento de dados) treinados para prever a decisao "
        "de alinhamento, o angulo e o deslocamento de cada formiga -- sem usar "
        "as equacoes fisicas do simulador."
    )

    ant_models = load_ant_models()

    if ant_models is None:
        st.info(
            "Configure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY e S3_BUCKET "
            "em secrets.toml para habilitar esta aba (os modelos sao baixados "
            "de s3://<bucket>/models/)."
        )
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Parametros")
            nav_in = st.slider("nav", 1, 60, 20, key="ant_nav")
            find_in = st.slider("find", 0.0, 3.0, 0.5, step=0.05, key="ant_find")
            theta_in = st.slider("theta", -3.14, 3.14, 0.0, step=0.05, key="ant_theta")
            b_in = st.slider("b", 0.5, 8.0, 4.0, step=0.1, key="ant_b")

        with col2:
            st.subheader("Previsao ao vivo")
            pred = predict_ant(ant_models, theta_in, nav_in, find_in, b_in)
            probs = pred["probs"]

            st.markdown("**P(alinhamento)**")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Lifter", f"{probs[0]:.1%}")
            pcol2.metric("Vazio", f"{probs[1]:.1%}")
            pcol3.metric("Puller", f"{probs[2]:.1%}")

            fig_bar = go.Figure(go.Bar(
                x=["Lifter", "Vazio", "Puller"], y=probs,
                marker_color=["#3D6B63", "#DBD7CD", "#C4622D"],
            ))
            fig_bar.update_layout(height=220, yaxis_title="Probabilidade", showlegend=False,
                                   margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(f"**Angulo phi previsto (se puller):** {pred['phi_pred']:.3f} rad")
            st.markdown(f"**Deslocamento previsto:** dx={pred['dx']:.4f}, dy={pred['dy']:.4f}")

            fig_vec = go.Figure()
            fig_vec.add_trace(go.Scatter(
                x=[0, pred["dx"]], y=[0, pred["dy"]], mode="lines+markers",
                line=dict(color="#C4622D", width=3), marker=dict(size=[4, 10]),
            ))
            fig_vec.update_layout(
                height=280, xaxis_title="dx", yaxis_title="dy", margin=dict(t=10, b=10),
                xaxis=dict(zeroline=True), yaxis=dict(zeroline=True, scaleanchor="x"),
            )
            st.plotly_chart(fig_vec, use_container_width=True)

# ============================================================
# ABA 6: VALIDACAO DO MODELO (rede neural vs dados reais vs teoria)
# ============================================================

with tab6:
    st.header("Validacao do Modelo — Rede Neural vs Dados Reais vs Teoria")

    ant_models = load_ant_models()

    if ant_models is None:
        st.info("Configure as credenciais S3 (ver aba anterior) para habilitar esta aba.")
    else:
        # --- Comparacao 1: P(puller) vs find normalizado ---
        st.subheader("P(puller) vs find normalizado")

        df_emp = query_p_puller_curve(engine)

        find_grid = np.linspace(0.01, 1.05, 100)
        nav_fixo, theta_fixo, b_fixo = 20.0, 0.0, 4.0
        rows = [[{"theta": theta_fixo, "nav": nav_fixo, "find": fn * nav_fixo, "b": b_fixo}[c]
                  for c in ant_models["feat_1b"]] for fn in find_grid]
        X_grid = ant_models["scaler_X_1b"].transform(np.array(rows))
        with torch.no_grad():
            probs_grid = torch.softmax(ant_models["model_1b"](torch.FloatTensor(X_grid)), dim=1).numpy()
        p_puller_nn = probs_grid[:, 2]

        fig1 = go.Figure()
        if not df_emp.empty:
            fig1.add_trace(go.Scatter(
                x=df_emp["find_normalized"], y=df_emp["p_puller"], mode="markers",
                name="Dados reais (RDS)",
                marker=dict(size=df_emp["n"] / df_emp["n"].max() * 20 + 4, color="#3D6B63"),
            ))
        fig1.add_trace(go.Scatter(x=find_grid, y=p_puller_nn, mode="lines", name="Rede neural (Modelo 1b)",
                                   line=dict(color="#C4622D", width=3)))
        fig1.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Ponto critico teorico")
        fig1.update_layout(xaxis_title="find normalizado", yaxis_title="P(puller)", height=420)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Comparacao 2: phi vs theta ---
        st.subheader("Angulo de alinhamento (phi) vs theta")

        df_phi = query_phi_vs_theta(engine)

        theta_grid = np.linspace(-np.pi, np.pi, 200)
        rows2 = [[{"theta": th, "nav": nav_fixo, "find": 0.5 * nav_fixo, "b": b_fixo}[c]
                   for c in ant_models["feat_2"]] for th in theta_grid]
        X_grid2 = ant_models["scaler_X_2"].transform(np.array(rows2))
        with torch.no_grad():
            phi_grid_scaled = ant_models["model_2"](torch.FloatTensor(X_grid2)).numpy()
        phi_grid = ant_models["scaler_y_2"].inverse_transform(phi_grid_scaled.reshape(-1, 1)).flatten()

        fig2 = go.Figure()
        if not df_phi.empty:
            fig2.add_trace(go.Scatter(x=df_phi["theta_bin"], y=df_phi["phi_medio"], mode="markers",
                                       name="Dados reais (RDS)", marker=dict(size=6, color="#3D6B63")))
        fig2.add_trace(go.Scatter(x=theta_grid, y=phi_grid, mode="lines", name="Rede neural (Modelo 2)",
                                   line=dict(color="#C4622D", width=3)))
        fig2.update_layout(xaxis_title="theta", yaxis_title="phi", height=420)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Comparacao 3: deslocamento previsto vs real ---
        st.subheader("Deslocamento previsto vs real (amostra de 3000 pontos)")

        df_disp = query_displacement_sample(engine, limit=3000)

        if not df_disp.empty:
            X_grid3 = ant_models["scaler_X_3b"].transform(df_disp[ant_models["feat_3b"]].values)
            with torch.no_grad():
                disp_pred_scaled = ant_models["model_3b"](torch.FloatTensor(X_grid3)).numpy()
            disp_pred = ant_models["scaler_y_3b"].inverse_transform(disp_pred_scaled)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_disp["dx"], y=disp_pred[:, 0], mode="markers",
                                           marker=dict(size=4, opacity=0.3, color="#3D6B63")))
                lims = [min(df_disp["dx"].min(), disp_pred[:, 0].min()),
                        max(df_disp["dx"].max(), disp_pred[:, 0].max())]
                fig3.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                           line=dict(color="#C4622D", dash="dash"), name="Previsao perfeita"))
                fig3.update_layout(xaxis_title="dx real", yaxis_title="dx previsto", height=380, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
            with dcol2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_disp["dy"], y=disp_pred[:, 1], mode="markers",
                                           marker=dict(size=4, opacity=0.3, color="#3D6B63")))
                lims = [min(df_disp["dy"].min(), disp_pred[:, 1].min()),
                        max(df_disp["dy"].max(), disp_pred[:, 1].max())]
                fig4.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                           line=dict(color="#C4622D", dash="dash"), name="Previsao perfeita"))
                fig4.update_layout(xaxis_title="dy real", yaxis_title="dy previsto", height=380, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

            from sklearn.metrics import r2_score
            r2 = r2_score(df_disp[["dx", "dy"]].values, disp_pred)
            st.metric("R² (dx + dy)", f"{r2:.4f}")
        else:
            st.warning("Nenhum dado de deslocamento encontrado")


# ============================================================
# ABA 7: DOCUMENTACAO (estilo artigo)
# ============================================================

with tab7:
    st.header("Estudo de um sistema biológico de formigas usando Mahine Learning e Redes Neurais ")
    st.markdown(
        "**Replicação computacional do modelo de carregadores acoplados e "
        "análise por aprendizado de máquina**"
    )

    st.divider()

    # ------------------------------------------------------------
    # RESUMO
    # ------------------------------------------------------------
    st.subheader("Resumo")
    st.markdown(
        """
Formigas *Paratrechina longicornis* transportam coletivamente objetos muito
maiores que um indivíduo. O comportamento macroscópico alterna entre fases:
períodos de movimento **balistico** coordenado e períodos de **cabo-de-guerra**,
em que as forças se cancelam e a carga fica quase parada. Gelblum et al.
propuseram que essa alternância emerge de uma regra local simples — cada formiga
decide, com base apenas na forca que sente, se puxa (*puller*) ou se apenas
sustenta (*lifter*) — sem nenhuma coordenacao central.

Este projeto reimplementa esse modelo em Python com dinâmica de Gillespie,
um modelo computacional para implementar Monte Carlo,
gera dados de simulação em duas granularidades (centro de massa da carga e
estado individual de cada sitio), persiste tudo em AWS (S3 + RDS PostgreSQL) e
treina seis modelos de aprendizado de maquina sobre esses dados — três
classicos e três redes neurais que tentam recuperar a dinamica **sem acesso as
equações**. O painel que voce esta lendo exibe todas as etapas de forma
interativa.
        """
    )

    st.divider()

    # ------------------------------------------------------------
    # 1. MODELO FISICO
    # ------------------------------------------------------------
    st.subheader("1. Modelo físico")

    st.markdown("#### 1.1 Geometria e estados")
    st.markdown(
        """
A carga é um disco rígido de raio $b$. Ao seu redor há $N_s$ sitios de
acoplamento igualmente espacados, fixos ao corpo da carga — portanto giram
junto com ela:
        """
    )
    st.latex(r"N_s = \max\left( \left\lfloor \frac{2\pi b}{\ell_{\rm form}} \right\rfloor,\ 6 \right)")
    st.markdown(
        """
Cada sitio $i$ tem posicao angular $\\theta_i$ e um estado discreto
$\\sigma_i \\in \\{-1, 0, +1\\}$:

| $\\sigma_i$ | Estado | Papel |
|---|---|---|
| $+1$ | **puller** | aplica forca ativa $f_0$ na direcao $\\theta_i + \\phi_i$ |
| $-1$ | **lifter** | sustenta a carga; reduz o atrito efetivo |
| $0$ | vazio | nenhuma formiga acoplada |

Um puller tambem carrega um ângulo de alinhamento $\\phi_i$, limitado a
$|\\phi_i| \\leq \\phi_{\\max}$. Lifters e sítios vazios tem $\\phi_i = 0$ por
construção — um detalhe que se mostrou importante na analise (secao 5).
        """
    )

    st.markdown("#### 1.2 Forças e dinâmica")
    st.markdown("A força total sobre o centro de massa e a soma vetorial dos pullers:")
    st.latex(r"\vec{F}_{\rm cm} = f_0 \sum_{i \in \rm pullers} \big( \cos(\theta_i + \phi_i),\ \sin(\theta_i + \phi_i) \big)")

    st.markdown(
        "O atrito cinético decresce linearmente com o número de lifters — e "
        "esse o mecanismo pelo qual eles ajudam sem puxar:"
    )
    st.latex(r"f_{\rm kin} = \max\big( f_{\rm kin,0} - \beta\, n_{\rm lifters},\ 0 \big)")

    st.markdown(
        "A carga so se move quando a força supera o atrito estático, "
        "$|\\vec{F}_{\\rm cm}| > f_{\\rm stat}$. Nesse caso, a força líquida é:"
    )
    st.latex(r"\vec{F}_{\rm liq} = \vec{F}_{\rm cm} \left( 1 - \frac{f_{\rm kin}}{|\vec{F}_{\rm cm}|} \right)")

    st.markdown("O torque vem da componente transversal do alinhamento dos pullers:")
    st.latex(r"\tau = f_0 \sum_{i \in \rm pullers} \sin(\phi_i)")

    st.markdown(
        "No regime superamortecido (numero de Reynolds baixo), velocidade "
        "e velocidade angular são proporcionais à força e ao torque:"
    )
    st.latex(r"\vec{v}_{\rm cm} = \frac{\vec{F}_{\rm liq}}{\gamma}, \qquad \omega = \frac{\tau_{\rm liq}}{\gamma_{\rm rot}}, \qquad \gamma_{\rm rot} = b\,\gamma")

    st.markdown("Cada sítio sente uma força local que combina translação e rotação:")
    st.latex(r"\vec{f}_{{\rm loc},i} = \vec{F}_{\rm cm} + \tau \big( -\sin\theta_i,\ \cos\theta_i \big)")

    st.markdown("#### 1.3 A regra de decisão")
    st.markdown(
        """
Cada formiga acoplada projeta a forca local que sente sobre a direção do próprio corpo:
        """
    )
    st.latex(r"x_i = \frac{\vec{f}_{{\rm loc},i} \cdot \hat{n}_i}{F_{\rm ind}}, \qquad \hat{n}_i = \big( \cos(\theta_i + \phi_i),\ \sin(\theta_i + \phi_i) \big)")

    st.markdown("As taxas de troca de papel dependem exponencialmente dessa projecao:")
    st.latex(r"k_{+1 \to -1} = k_c\, e^{-x_i} \qquad\qquad k_{-1 \to +1} = k_c\, e^{+x_i}")

    st.markdown(
        """
A leitura física: uma formiga que sente o grupo puxando **a favor** da sua
direção tende a permanecer puller; uma que sente o grupo puxando **contra**
tende a desistir e virar lifter. O parametro $F_{\\rm ind}$ é a escala de
"individualidade" — quanto menor, mais sensivel a formiga é a opinião coletiva.

Além das trocas de papel, o sistema tem acoplamento e desacoplamento de
formigas aos sitios:
        """
    )
    st.latex(r"R_{\rm att} = k_{\rm on}\, n_{\rm vazios} \max(N_{\rm av} - n_{\rm ocup},\ 0), \qquad R_{\rm det} = k_{\rm off}\, n_{\rm ocup}")

    st.markdown(
        "A taxa de desacoplamento $k_{\\rm off}$ assume valores diferentes conforme "
        "a carga esteja em movimento ou parada."
    )

    st.markdown("#### 1.4 Integração temporal")
    st.markdown(
        "A evolução usa o algoritmo de Gillespie: sorteia-se de um intervalo até o "
        "próximo evento estocastico a partir da taxa total,"
    )
    st.latex(r"\Delta t = -\frac{\ln u}{R_{\rm tot}}, \qquad u \sim \mathcal{U}(0,1)")
    st.markdown(
        "e, entre eventos, as equações de movimento são integradas "
        "deterministicamente com passo fixo $\\delta t$."
    )

    st.markdown("#### 1.5 Transicao de fase")
    st.markdown(
        """
O modelo mapeia num sistema tipo Ising com acoplamento efetivo: os estados
puller/lifter fazem o papel dos spins, e $F_{\\rm ind}$ o papel da temperatura.
A referência teórica situa a transicao ordem-desordem em
$F_{\\rm ind}/N_{\\rm av} = 0.5$. Abaixo desse valor, o consenso domina e a carga
avança de forma balística; acima, a individualidade vence e o sistema entra no
regime de cabo-de-guerra.
        """
    )

    with st.expander("Paramêtros da simulação (valores padrão)"):
        st.dataframe(
            pd.DataFrame([
                {"Simbolo": "f0", "Valor": 1.0, "Significado": "Forca de uma formiga (unidade de referencia)"},
                {"Simbolo": "f_kin,0", "Valor": 2.7, "Significado": "Atrito cinetico sem lifters"},
                {"Simbolo": "f_stat", "Valor": 3.0, "Significado": "Limiar de atrito estatico"},
                {"Simbolo": "beta", "Valor": 1.65, "Significado": "Reducao de atrito por lifter"},
                {"Simbolo": "b", "Valor": 4.0, "Significado": "Raio da carga (cm)"},
                {"Simbolo": "l_form", "Valor": 0.25, "Significado": "Espacamento entre sitios"},
                {"Simbolo": "phi_max", "Valor": 0.4534, "Significado": "Angulo maximo de alinhamento (rad)"},
                {"Simbolo": "gamma", "Valor": 25.0, "Significado": "Coeficiente de arrasto"},
                {"Simbolo": "k_on", "Valor": 1.7e-3, "Significado": "Taxa de acoplamento"},
                {"Simbolo": "k_off (mov / parado)", "Valor": "1.0e-2 / 3.5e-2", "Significado": "Taxa de desacoplamento"},
                {"Simbolo": "k_c", "Valor": 0.7, "Significado": "Taxa basal de decisao"},
                {"Simbolo": "F_ind", "Valor": 1.0, "Significado": "Escala de individualidade"},
                {"Simbolo": "N_av", "Valor": 20, "Significado": "Numero maximo de formigas"},
                {"Simbolo": "delta t", "Valor": 1e-2, "Significado": "Passo de integracao"},
            ]),
            hide_index=True, use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------
    # 2. METODOS
    # ------------------------------------------------------------
    st.subheader("2. Métodos")

    st.markdown(
        """
O projeto foi construido como um pipeline de ponta a ponta:

**(a) Simulacao.** `simulator_v2.py` implementa o modelo acima;
`run_simulations_v2.py` executa varreduras em paralelo. Dois níveis de
granularidade são gravados:

- *Centro de massa* — uma linha por quadro por simulação: posição, velocidade,
  contagem de pullers e lifters.
- *Por formiga* — uma linha por sítio por quadro: estado $\\sigma_i$, ângulos
  $\\theta_i$ e $\\phi_i$, posição absoluta. Esse conjunto (`ant_trajectories.csv`,
  cerca de 669 mil linhas) e o que permite treinar modelos sobre a decisão
  individual.

**(b) Análise exploratória e modelagem clássica.** Notebooks de EDA e de
aprendizado supervisionado/nao-supervisionado sobre os dados agregados.

**(c) Persistência em nuvem.** Arquivos brutos e modelos serializados no S3;
dados tabulares em RDS PostgreSQL, em três tabelas — `simulations`,
`phase_transition` e `ant_trajectories`.

**(d) Redes neurais.** Três perceptrons multicamada treinados sobre o conjunto
por formiga, com o objetivo de recuperar a dinêmica sem usar nenhuma das
equações da seção 1.

**(e) Painel.** Esta aplicação consulta o RDS, baixa os modelos do S3 e executa
inferência ao vivo.
        """
    )

    st.divider()

    # ------------------------------------------------------------
    # 3. MODELOS
    # ------------------------------------------------------------
    st.subheader("3. Modelos treinados")

    st.markdown("**Fase clássica** — sobre trajetórias do centro de massa:")
    st.dataframe(
        pd.DataFrame([
            {"Modelo": "Classificador de regime",
             "Algoritmo": "StandardScaler + RandomForest",
             "Alvo": "Ordenado vs desordenado",
             "Observacao": "Ver ressalva na secao 5"},
            {"Modelo": "Detector de ponto critico",
             "Algoritmo": "PolynomialFeatures(3) + Ridge",
             "Alvo": "Parametro de ordem |m|",
             "Observacao": "Ajuste modesto; ver secao 5"},
            {"Modelo": "Clusters de comportamento",
             "Algoritmo": "KMeans (K=7)",
             "Alvo": "Nao supervisionado",
             "Observacao": "Silhueta ~0.48"},
        ]),
        hide_index=True, use_container_width=True,
    )

    st.markdown(
        "**Fase de redes neurais** — sobre o conjunto por formiga. Cada modelo "
        "tenta recuperar uma parte da regra de decisao da secao 1.3:"
    )
    st.dataframe(
        pd.DataFrame([
            {"Modelo": "1b — Alinhamento",
             "Entrada": "theta, N_av, F_ind, b",
             "Saida": "sigma (lifter / vazio / puller)",
             "Pergunta": "QUANDO a formiga se alinha?"},
            {"Modelo": "2 — Angulo",
             "Entrada": "theta, N_av, F_ind, b",
             "Saida": "phi",
             "Pergunta": "QUAL o angulo de alinhamento?"},
            {"Modelo": "3b — Deslocamento",
             "Entrada": "theta, phi, sigma, N_av, F_ind, b",
             "Saida": "(dx, dy) ate o proximo quadro",
             "Pergunta": "PARA ONDE a formiga se move?"},
        ]),
        hide_index=True, use_container_width=True,
    )

    st.divider()

    # ------------------------------------------------------------
    # 4. GUIA DAS ABAS
    # ------------------------------------------------------------
    st.subheader("4. Guia das abas")

    st.markdown(
        """
**Transição.** Parametro de ordem $|m|$ contra o acoplamento
normalizado $F_{\\rm ind}/N_{\\rm av}$, para diferentes tamanhos de grupo. E a
assinatura da transicao de fase: a curva separa o regime coordenado do regime
de cabo-de-guerra. A linha de referencia marca o valor teorico $0.5$.

**Raios.** Velocidade media da carga contra o raio $b$. Cargas maiores
comportam mais sitios de acoplamento, o que muda o balanco entre forca coletiva
e atrito.

**Predições com ML.** Os tres modelos classicos, interativos. Voce ajusta a
configuracao fisica (raio, velocidade, numero de pullers e lifters) e ve a
classificacao de regime ao vivo, com analise de sensibilidade; a curva do ponto
critico ajustada sobre os dados reais; e em qual cluster de comportamento a sua
configuracao cai.

**Data.** Acesso direto as tabelas, para inspecao e exportacao.

**Formiga Individual.** As tres redes neurais em sequencia, para uma unica
formiga: dada uma configuracao local, qual a probabilidade de ela ser puller,
qual angulo ela adotaria e para onde se moveria. Nenhuma equacao da secao 1 e
usada aqui — so o que as redes aprenderam dos dados.

**Validação do Modelo.** Confronto entre o que as redes aprenderam e os dados
reais: probabilidade de ser puller contra $F_{\\rm ind}$ normalizado, angulo
contra posicao angular, e deslocamento previsto contra observado.
        """
    )

    st.divider()

    # ------------------------------------------------------------
    # 5. LIMITACOES
    # ------------------------------------------------------------
    st.subheader("5. Limitações e ressalvas")

    st.markdown(
        """
Esta seção registra problemas encontrados durante o desenvolvimento. Estao aqui
porque afetam como os resultados devem ser lidos.
        """
    )

    with st.expander("A alta acurácia do classificador de regime é em parte circular", expanded=True):
        st.markdown(
            """
O rótulo de treino foi construido como "velocidade acima ou abaixo da mediana",
e a própria velocidade está entre as features de entrada. O modelo, portanto,
em boa medida lê a resposta na pergunta. A acurácia proxima de 100% mede a
consistência do procedimento, não capacidade preditiva sobre o fenomeno.

O gráfico de importância de features na aba *Predicoes ML* mostra exatamente
isso: `speed` domina. Um teste mais informativo seria prever o regime a partir
apenas das contagens de pullers e lifters, sem a velocidade.
            """
        )

    with st.expander("Dois dos modelos neurais originais tinham vazamento de dados", expanded=True):
        st.markdown(
            """
Na primeira versao, o classificador de alinhamento recebia $\\phi$ entre as
entradas. Mas no simulador $\\phi$ e forcado a zero sempre que o sitio nao e
puller (secao 1.1) — ou seja, a entrada quase entregava a resposta. A rede
aprendia a regra trivial "$\\phi = 0 \\Rightarrow$ nao-puller", nao a fisica da
decisao.

O preditor de movimento original recebia $(x, y)$ e previa $(x, y)$: um
mapeamento praticamente identidade, com $R^2$ artificialmente alto.

Ambos foram substituidos pelas versoes corrigidas **1b** (sem $\\phi$) e **3b**
(preve o deslocamento $(dx, dy)$ a partir do estado, sem ver a posicao). Sao
essas as versoes usadas no painel. Os numeros das versoes originais nao devem
ser citados.
            """
        )

    with st.expander("O detector de ponto critico nao localiza uma transicao abrupta"):
        st.markdown(
            """
A curva polinomial ajustada e monotonica decrescente no intervalo coberto pelos
dados, sem maximo interno de inclinacao. A aba correspondente reporta isso
explicitamente em vez de exibir um numero: qualquer "ponto critico" extraido da
derivada cairia na borda do intervalo, sendo artefato de extrapolacao do
polinomio, nao fisica.

Isso e coerente com o ajuste modesto do modelo e sugere que um grau 3 sobre uma
unica variavel nao e a forma adequada de localizar a transicao.
            """
        )

    with st.expander("Cobertura de colunas no banco"):
        st.markdown(
            """
A tabela `simulations` foi carregada apenas com colunas agregadas
(`radius`, `nav`, `find`, `mean_speed`, `std_speed`). As contagens
`n_pullers`, `n_lifters` e `n_occupied` ficaram de fora do ETL.

Consequencia: no modo nuvem, os graficos de contexto da aba *Predicoes ML*
ficam indisponiveis, enquanto a predicao interativa segue funcionando. No modo
local, com o CSV completo, tudo funciona.
            """
        )

    with st.expander("Escopo dos dados"):
        st.markdown(
            """
Todos os dados sao **de simulacao**, nao experimentais. O projeto valida a
implementacao do modelo e explora o que metodos estatisticos recuperam dele —
nao constitui evidencia empirica independente sobre o comportamento das
formigas.

Alem disso, as redes neurais foram treinadas sobre uma grade finita de
parametros; extrapolar para regioes nao amostradas nao e justificado.
            """
        )

    st.divider()

    # ------------------------------------------------------------
    # 6. INFRAESTRUTURA
    # ------------------------------------------------------------
    st.subheader("6. Infraestrutura")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown(
            """
**Dados e armazenamento**
- S3: arquivos brutos e modelos serializados
- RDS PostgreSQL: `simulations`, `phase_transition`, `ant_trajectories`
- Agregacoes pesadas executadas no banco (`GROUP BY`, funcao de janela
  `LEAD`), nao no cliente
            """
        )
    with col_i2:
        st.markdown(
            """
**Aplicação**
- Streamlit, com cache por aba
- Modelos baixados do S3 uma vez por sessao, sob demanda
- Modo local alternativo: le CSVs do disco quando nao ha banco configurado
            """
        )

    st.caption(
        f"Sessao atual — fonte de dados: **{DATA_MODE.upper()}** | "
        f"modelos classicos: {'carregados' if models else 'ausentes'}"
    )

    st.divider()

    # ------------------------------------------------------------
    # REFERENCIAS
    # ------------------------------------------------------------
    st.subheader("Referencias")
    st.markdown(
        """
Gelblum, A., Pinkoviezky, I., Fonio, E., Ghosh, A., Gov, N. & Feinerman, O.
Ant groups optimally amplify the effect of transiently informed individuals.
*Nature Communications* **6**, 7729 (2015).

Feinerman, O., Pinkoviezky, I., Gelblum, A., Fonio, E. & Gov, N. S.
The physics of cooperative transport in groups of ants.
*Nature Physics* **14**, 683-693 (2018). — modelo implementado aqui (Box 1)

Gillespie, D. T. Exact stochastic simulation of coupled chemical reactions.
*The Journal of Physical Chemistry* **81**, 2340-2361 (1977). — algoritmo de
integracao estocastica
        """
    )


# ============================================================
# RODAPE
# ============================================================

st.divider()
st.markdown("""
**Dashboard interativo — Transporte Cooperativo de Formigas**

Dados: simulacoes de carregadores acoplados (Gelblum et al., 2018)
Base de dados: AWS RDS PostgreSQL
Modelos: scikit-learn (RF, SVR, KMeans)

[Relatorio original](https://www.nature.com/articles/s41567-018-0107-y) 
| [GitHub](https://github.com/leo-lopes/Ants) 
""")