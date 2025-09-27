
# app.py — v3.3 (Dashboards sem gráficos + conversão BRL/USD correta)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os
from datetime import datetime
import yfinance as yf

DB_PATH = "invest.db"
SEED_PATH = "seed_investimentos.csv"
REQUIRE_PIN = os.getenv("APP_PIN", "1234")

st.set_page_config(page_title="Investimentos – v3.3", page_icon="📈", layout="wide")

# ---------------------- Auth ----------------------
if "authed" not in st.session_state:
    st.session_state.authed = False
if not st.session_state.authed:
    with st.sidebar:
        st.subheader("🔒 Acesso")
        pin = st.text_input("Digite seu PIN", type="password")
        if st.button("Entrar"):
            if pin == REQUIRE_PIN:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("PIN incorreto.")
    st.stop()

# ---------------------- DB helpers ----------------------
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS movimentos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        ticket TEXT,
        nome TEXT,
        preco REAL,
        quantidade REAL,
        valor_investido REAL,
        compra_venda TEXT,
        onde TEXT,
        tipo TEXT,
        obs TEXT,
        country TEXT,
        categoria TEXT,
        month INTEGER,
        year INTEGER
    );
    """ )
    conn.commit()

def seed_if_empty(conn):
    cur = conn.execute("SELECT COUNT(*) FROM movimentos")
    count = cur.fetchone()[0]
    if count == 0 and os.path.exists(SEED_PATH):
        df = pd.read_csv(SEED_PATH, parse_dates=['data'])
        ensure_cols = ['data','ticket','nome','preco','quantidade','valor_investido','compra_venda',
                       'onde','tipo','obs','country','categoria','month','year']
        for c in ensure_cols:
            if c not in df.columns: df[c] = np.nan
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df['month'] = df['data'].dt.month
        df['year'] = df['data'].dt.year
        df = df.sort_values('data')
        df.to_sql('movimentos', conn, if_exists='append', index=False)

def load_df(conn):
    return pd.read_sql_query("SELECT * FROM movimentos ORDER BY datetime(data) ASC", conn, parse_dates=['data'])

# ---------------------- Helpers ----------------------
def norm_country(s: str):
    if not isinstance(s, str):
        return "Brasil"
    s2 = s.strip().lower()
    if s2 in ["brasil", "brazil"]:
        return "Brasil"
    if s2 in ["eua", "usa", "united states", "estados unidos", "us"]:
        return "USA"
    if "crypto" in s2 or "cripto" in s2 or "btc" in s2 or "eth" in s2:
        return "Crypto"
    return s.strip()

@st.cache_data(ttl=3600)
def get_usd_brl():
    try:
        data = yf.download(['BRL=X'], period="5d", interval="1d", progress=False)
        if data is None or data.empty or 'Close' not in data:
            return np.nan
        close = data['Close'].ffill()
        if close is None or close.empty:
            return np.nan
        last = close.iloc[-1]
        if isinstance(last, pd.Series):
            return float(last.get('BRL=X', np.nan))
        return float(last)
    except Exception:
        return np.nan

# ---------------------- App ----------------------
conn = get_conn()
create_table(conn)
seed_if_empty(conn)

st.title("📊 Controle de Investimentos – v3.3")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["➕ Novo", "📋 Movimentos", "📊 Dashboards", "📦 Posições", "✏️ Editar/Excluir"])

st.caption("As abas ➕ Novo, 📋 Movimentos, 📦 Posições e ✏️ Editar/Excluir permanecem como na v3.2. Abaixo está apenas a nova aba de Dashboards.")

# ---- Dashboards (sem gráficos, com conversão) ----
with tab3:
    st.subheader("Dashboards")
    df = load_df(conn)
    if df.empty:
        st.info("Sem dados ainda.")
    else:
        ddf = df.copy()
        ddf['country_norm'] = ddf['country'].apply(norm_country)

        usd_brl = get_usd_brl()
        st.caption(f"USD/BRL usado: {usd_brl:,.4f}" if pd.notna(usd_brl) else "USD/BRL indisponível no momento.")

        def split_values(row):
            country = row['country_norm']
            val = float(row['valor_investido']) if pd.notna(row['valor_investido']) else 0.0
            if country == 'USA' and pd.notna(usd_brl) and usd_brl != 0:
                return pd.Series({'valor_local': val, 'moeda_local': 'USD', 'valor_brl': val * usd_brl})
            else:
                return pd.Series({'valor_local': val, 'moeda_local': 'BRL', 'valor_brl': val})

        ddf = pd.concat([ddf, ddf.apply(split_values, axis=1)], axis=1)

        total_investido_brl = ddf['valor_brl'].sum()
        st.metric("Valor total investido (BRL)", f"{total_investido_brl:,.2f}")

        st.markdown("### Por corretora/plataforma (com conversão)")
        by_onde_brl = ddf.groupby('onde', dropna=False)['valor_brl'].sum().reset_index().rename(columns={'valor_brl':'Aporte (BRL)'})
        predominante = ddf.groupby(['onde','moeda_local'])['valor_local'].sum().reset_index()
        idx = predominante.groupby('onde')['valor_local'].idxmax()
        moeda_pred = predominante.loc[idx, ['onde','moeda_local']]
        by_onde_local = ddf.groupby('onde')['valor_local'].sum().reset_index().rename(columns={'valor_local':'Aporte (Local)'})
        table_onde = by_onde_brl.merge(by_onde_local, on='onde', how='left').merge(moeda_pred, on='onde', how='left')
        table_onde = table_onde.rename(columns={'moeda_local':'Moeda Local (predominante)'})
        table_onde = table_onde.sort_values('Aporte (BRL)', ascending=False)
        st.dataframe(table_onde, use_container_width=True)

        st.markdown("### Por país (Local vs BRL)")
        by_country = ddf.groupby('country_norm').agg(
            aporte_local=('valor_local','sum'),
            aporte_brl=('valor_brl','sum'),
            moeda=('moeda_local', lambda x: x.value_counts().index[0] if len(x)>0 else 'BRL')
        ).reset_index().rename(columns={'country_norm':'País','aporte_local':'Aporte (Local)','aporte_brl':'Aporte (BRL)','moeda':'Moeda'})
        by_country = by_country.sort_values('Aporte (BRL)', ascending=False)
        st.dataframe(by_country, use_container_width=True)

        st.markdown("---")
        st.markdown("### Detalhes (tabelas por país)")
        with st.expander("Brazil"):
            br = ddf[ddf['country_norm']=='Brasil']
            if br.empty: st.info("Sem dados do Brasil.")
            else:
                det = br.groupby('ticket', dropna=True).agg(
                    quantidade_total=('quantidade', 'sum'),
                    aporte_local=('valor_local','sum'),
                    aporte_brl=('valor_brl','sum'),
                    primeira_data=('data','min'),
                    ultima_data=('data','max'),
                    onde_principal=('onde', lambda x: x.value_counts(dropna=True).index[0] if len(x.dropna())>0 else None)
                ).reset_index().sort_values('aporte_brl', ascending=False)
                st.dataframe(det, use_container_width=True)

        with st.expander("US"):
            us = ddf[ddf['country_norm']=='USA']
            if us.empty: st.info("Sem dados dos EUA.")
            else:
                det = us.groupby('ticket', dropna=True).agg(
                    quantidade_total=('quantidade', 'sum'),
                    aporte_local=('valor_local','sum'),
                    aporte_brl=('valor_brl','sum'),
                    primeira_data=('data','min'),
                    ultima_data=('data','max'),
                    onde_principal=('onde', lambda x: x.value_counts(dropna=True).index[0] if len(x.dropna())>0 else None)
                ).reset_index().sort_values('aporte_brl', ascending=False)
                st.dataframe(det, use_container_width=True)

        with st.expander("Crypto"):
            cr = ddf[ddf['country_norm']=='Crypto']
            if cr.empty: st.info("Sem dados de cripto.")
            else:
                det = cr.groupby('ticket', dropna=True).agg(
                    quantidade_total=('quantidade', 'sum'),
                    aporte_local=('valor_local','sum'),
                    aporte_brl=('valor_brl','sum'),
                    primeira_data=('data','min'),
                    ultima_data=('data','max'),
                    onde_principal=('onde', lambda x: x.value_counts(dropna=True).index[0] if len(x.dropna())>0 else None)
                ).reset_index().sort_values('aporte_brl', ascending=False)
                st.dataframe(det, use_container_width=True)
