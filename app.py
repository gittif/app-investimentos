
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import matplotlib.pyplot as plt

DB_PATH = "invest.db"
SEED_PATH = "seed_investimentos.csv"

st.set_page_config(page_title="Investimentos PJ/CPF", page_icon="📈", layout="wide")

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

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
    """)
    conn.commit()

def seed_if_empty(conn):
    cur = conn.execute("SELECT COUNT(*) FROM movimentos")
    count = cur.fetchone()[0]
    if count == 0 and os.path.exists(SEED_PATH):
        df = pd.read_csv(SEED_PATH, parse_dates=['data'])
        # Ensure correct schema
        for col in ['data','ticket','nome','preco','quantidade','valor_investido','compra_venda',
                    'onde','tipo','obs','country','categoria','month','year']:
            if col not in df.columns:
                df[col] = np.nan
        # insert
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df['month'] = df['data'].dt.month
        df['year'] = df['data'].dt.year
        df = df.sort_values('data')
        df.to_sql('movimentos', conn, if_exists='append', index=False)

def insert_movimento(conn, row: dict):
    conn.execute("""
        INSERT INTO movimentos 
        (data, ticket, nome, preco, quantidade, valor_investido, compra_venda, onde, tipo, obs, country, categoria, month, year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['data'], row['ticket'], row['nome'], row['preco'], row['quantidade'], row['valor_investido'],
        row['compra_venda'], row['onde'], row['tipo'], row['obs'], row['country'], row['categoria'],
        row['month'], row['year']
    ))
    conn.commit()

def load_df(conn):
    return pd.read_sql_query("SELECT * FROM movimentos ORDER BY datetime(data) ASC", conn, parse_dates=['data'])

conn = get_conn()
create_table(conn)
seed_if_empty(conn)

st.title("📊 Controle de Investimentos – Cadastro Rápido")
st.caption("Adicione novos aportes/vendas pelo celular, visualize e exporte seus dados.")

tab1, tab2, tab3 = st.tabs(["➕ Novo investimento", "📋 Movimentos", "📈 Dashboard"])

with tab1:
    st.subheader("Adicionar movimento")
    col1, col2 = st.columns(2)
    with col1:
        data = st.date_input("Data", value=datetime.today())
        ticket = st.text_input("Ticker (ex: PETR4, AAPL)", value="")
        nome = st.text_input("Nome do ativo (opcional)", value="")
        preco = st.number_input("Preço por unidade", min_value=0.0, step=0.01, format="%.2f")
        quantidade = st.number_input("Quantidade", min_value=0.0, step=1.0)
    with col2:
        compra_venda = st.selectbox("Operação", ["Compra", "Venda"])
        onde = st.text_input("Onde (corretora/plataforma)", value="")
        tipo = st.text_input("Tipo (ex: Brasil Ações, EUA Ações, FII, ETF)", value="")
        country = st.text_input("País", value="Brasil")
        categoria = st.text_input("Categoria (ex: RV, RF, FII, ETF)", value="RV")
        obs = st.text_area("Observações", value="")

    submitted = st.button("Salvar movimento")
    if submitted:
        valor_investido = float(preco) * float(quantidade)
        if compra_venda == "Venda":
            valor_investido = -valor_investido

        row = {
            "data": pd.to_datetime(data).strftime("%Y-%m-%d"),
            "ticket": ticket.strip().upper(),
            "nome": nome.strip(),
            "preco": float(preco),
            "quantidade": float(quantidade),
            "valor_investido": float(valor_investido),
            "compra_venda": compra_venda,
            "onde": onde.strip(),
            "tipo": tipo.strip(),
            "obs": obs.strip(),
            "country": country.strip(),
            "categoria": categoria.strip(),
        }
        row["month"] = pd.to_datetime(row["data"]).month
        row["year"] = pd.to_datetime(row["data"]).year

        # Quick validations
        if row["ticket"] == "" or row["preco"] <= 0 or row["quantidade"] <= 0:
            st.error("Preencha Ticker, Preço (> 0) e Quantidade (> 0).")
        else:
            insert_movimento(conn, row)
            st.success(f"Movimento salvo com sucesso para {row['ticket']} em {row['data']}.")

with tab2:
    st.subheader("Histórico de movimentos")
    df = load_df(conn)

    # Filtros
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        filtro_ticket = st.text_input("Filtrar por Ticker")
    with c2:
        filtro_ano = st.multiselect("Ano", sorted(df['year'].dropna().unique().tolist()))
    with c3:
        filtro_operacao = st.multiselect("Operação", ["Compra","Venda"])
    with c4:
        filtro_onde = st.text_input("Filtrar por 'Onde'")

    fdf = df.copy()
    if filtro_ticket:
        fdf = fdf[fdf['ticket'].str.contains(filtro_ticket.strip().upper(), na=False)]
    if filtro_ano:
        fdf = fdf[fdf['year'].isin(filtro_ano)]
    if filtro_operacao:
        fdf = fdf[fdf['compra_venda'].isin(filtro_operacao)]
    if filtro_onde:
        fdf = fdf[fdf['onde'].str.contains(filtro_onde, na=False, case=False)]

    st.dataframe(fdf[['data','ticket','nome','preco','quantidade','valor_investido','compra_venda','onde','tipo','categoria','obs']])

    # Exportações
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv = fdf.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Baixar CSV filtrado", data=csv, file_name="movimentos_filtrado.csv", mime="text/csv")
    with exp_col2:
        # Export all data as backup
        all_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Backup completo (CSV)", data=all_csv, file_name="movimentos_backup_completo.csv", mime="text/csv")

with tab3:
    st.subheader("Visão geral")
    df = load_df(conn)
    if df.empty:
        st.info("Sem dados ainda. Adicione um movimento.")
    else:
        # KPIs simples
        total_aportes = df.loc[df['valor_investido']>0, 'valor_investido'].sum()
        total_resgates = -df.loc[df['valor_investido']<0, 'valor_investido'].sum()
        num_tickers = df['ticket'].nunique()

        k1, k2, k3 = st.columns(3)
        k1.metric("Total aportado (R$)", f"{total_aportes:,.2f}")
        k2.metric("Total resgatado (R$)", f"{total_resgates:,.2f}")
        k3.metric("Ativos únicos", int(num_tickers))

        # Gráfico 1: Aportes por ano
        by_year = df.groupby('year', dropna=True)['valor_investido'].sum().reset_index().sort_values('year')
        fig1, ax1 = plt.subplots()
        ax1.bar(by_year['year'].astype(int), by_year['valor_investido'])
        ax1.set_title("Fluxo líquido por ano (R$)")
        ax1.set_xlabel("Ano")
        ax1.set_ylabel("R$")
        st.pyplot(fig1)

        # Gráfico 2: Top tickers por aporte líquido
        by_ticker = df.groupby('ticket', dropna=True)['valor_investido'].sum().reset_index().sort_values('valor_investido', ascending=False).head(15)
        fig2, ax2 = plt.subplots()
        ax2.barh(by_ticker['ticket'], by_ticker['valor_investido'])
        ax2.set_title("Top 15 – Aporte líquido por ticker (R$)")
        ax2.set_xlabel("R$")
        ax2.set_ylabel("Ticker")
        st.pyplot(fig2)

        # Tabela pivô por onde/corretora
        st.markdown("**Aporte líquido por corretora/plataforma (R$)**")
        piv_onde = df.pivot_table(index='onde', values='valor_investido', aggfunc='sum').sort_values('valor_investido', ascending=False)
        st.dataframe(piv_onde)

st.caption("Dica: Use o botão 'Backup completo (CSV)' para manter uma cópia local sempre que desejar.")
