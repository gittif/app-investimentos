# app.py — v3.3 (completo)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

DB_PATH = "invest.db"
SEED_PATH = "seed_investimentos.csv"
REQUIRE_PIN = os.getenv("APP_PIN", "1234")

st.set_page_config(page_title="Controle de Investimentos – v3.3", page_icon="📊", layout="wide")

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
    """)
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

def update_movimento(conn, row_id: int, updates: dict):
    sets = ", ".join([f"{k}=?" for k in updates.keys()])
    vals = list(updates.values()) + [row_id]
    conn.execute(f"UPDATE movimentos SET {sets} WHERE id=?", vals)
    conn.commit()

def delete_movimento(conn, row_id: int):
    conn.execute("DELETE FROM movimentos WHERE id=?", (row_id,))
    conn.commit()

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

# ---------------------- Prices & Positions ----------------------
def guess_ticker_symbol(row):
    t = str(row.get('ticket','')).upper().strip()
    country = str(row.get('country','')).strip().lower()
    tipo = str(row.get('tipo','')).strip().lower()
    if 'brasil' in country or 'b3' in tipo or 'fii' in tipo or 'brasil ações' in tipo:
        if t.endswith('.SA'): return t
        return t + '.SA'
    return t

def fetch_prices(tickers_unique):
    prices = {tk: np.nan for tk in tickers_unique}
    usd_brl = np.nan
    if not tickers_unique:
        return prices, usd_brl
    try:
        data = yf.download(tickers_unique, period="5d", interval="1d", progress=False)
        if data is None or data.empty or 'Close' not in data:
            return prices, usd_brl
        close = data['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame()
        close = close.ffill()
        if close.empty:
            return prices, usd_brl
        latest = close.iloc[-1]
        for tk in tickers_unique:
            try:
                prices[tk] = float(latest.get(tk, np.nan))
            except Exception:
                prices[tk] = np.nan
    except Exception:
        pass
    usd_brl = get_usd_brl()
    return prices, usd_brl

def build_positions(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), np.nan

    tmp = df.copy()
    tmp['ticker_fetch'] = tmp.apply(guess_ticker_symbol, axis=1)
    tmp['qtd_signed'] = np.where(tmp['compra_venda']=='Venda', -tmp['quantidade'], tmp['quantidade'])

    agg = tmp.groupby(['ticket','country'], dropna=False).agg(
        qtd_total=('qtd_signed','sum'),
        aporte=('valor_investido','sum')
    ).reset_index()

    compras = tmp[tmp['compra_venda'] == 'Compra'].groupby('ticket').apply(
        lambda g: (g['preco'] * g['quantidade']).sum() / max(g['quantidade'].sum(), 1e-9)
    ).rename('preco_medio').reset_index()
    positions = agg.merge(compras, on='ticket', how='left')

    fetch_map = tmp.drop_duplicates('ticket')[['ticket','ticker_fetch']]
    positions = positions.merge(fetch_map, on='ticket', how='left')

    tickers_unique = positions['ticker_fetch'].dropna().unique().tolist()
    price_map, usd_brl = fetch_prices(tickers_unique)

    positions['preco_atual'] = positions['ticker_fetch'].map(price_map)
    positions['moeda'] = np.where(positions['country'].str.lower().eq('brasil'), 'BRL', 'USD')
    positions['valor_atual_moeda'] = positions['preco_atual'] * positions['qtd_total']

    def to_brl(row):
        if row['moeda'] == 'USD' and pd.notna(row['valor_atual_moeda']) and pd.notna(usd_brl):
            return row['valor_atual_moeda'] * usd_brl
        return row['valor_atual_moeda']
    positions['valor_atual_brl'] = positions.apply(to_brl, axis=1)

    positions['pnl_brl'] = positions['valor_atual_brl'] - positions['aporte']
    positions['pnl_pct'] = np.where(positions['aporte'] != 0, positions['pnl_brl'] / positions['aporte'], np.nan)

    positions = positions.sort_values('valor_atual_brl', ascending=False)
    return positions, usd_brl

# ---------------------- UI ----------------------
conn = get_conn()
create_table(conn)
seed_if_empty(conn)

st.title("Controle de Investimentos – v3.3")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["➕ Novo", "📋 Movimentos", "📊 Dashboards", "📦 Posições", "✏️ Editar/Excluir"])

# ---- Novo
with tab1:
    st.subheader("Adicionar movimento")
    col1, col2 = st.columns(2)
    with col1:
        data = st.date_input("Data", value=datetime.today())
        ticket = st.text_input("Ticker (ex: PETR4, AAPL)")
        nome = st.text_input("Nome do ativo (opcional)")
        preco = st.number_input("Preço por unidade", min_value=0.0, step=0.01, format="%.2f")
        quantidade = st.number_input("Quantidade", min_value=0.0, step=1.0)
    with col2:
        compra_venda = st.selectbox("Operação", ["Compra", "Venda"])
        onde = st.text_input("Onde (corretora/plataforma)")
        tipo = st.text_input("Tipo (ex: Brasil Ações, EUA Ações, FII, ETF)")
        country = st.selectbox("País", ["Brasil","USA","Crypto"], index=0)
        categoria = st.text_input("Categoria (ex: RV, RF, FII, ETF)", value="RV")
        obs = st.text_area("Observações")

    if st.button("Salvar"):
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

        if row["ticket"] == "" or row["preco"] <= 0 or row["quantidade"] <= 0:
            st.error("Preencha Ticker, Preço (> 0) e Quantidade (> 0).")
        else:
            insert_movimento(conn, row)
            st.success(f"Movimento salvo para {row['ticket']} em {row['data']}.")

# ---- Movimentos (edição inline + País dropdown)
with tab2:
    st.subheader("Histórico de movimentos")
    df = load_df(conn)

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

    st.caption("✅ Edite diretamente as células e clique em **Salvar alterações**.")
    edited = st.data_editor(
        fdf[['id','data','ticket','nome','preco','quantidade','valor_investido','compra_venda','onde','tipo','country','categoria','obs']],
        num_rows="fixed",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "data": st.column_config.DateColumn("data"),
            "compra_venda": st.column_config.SelectboxColumn("compra_venda", options=["Compra","Venda"]),
            "country": st.column_config.SelectboxColumn("País", options=["Brasil","USA","Crypto"]),
        },
        use_container_width=True
    )

    if st.button("Salvar alterações"):
        merged = edited.merge(fdf[['id']], on='id', how='left')
        for _, row in merged.iterrows():
            orig = fdf[fdf['id']==row['id']].iloc[0]
            updates = {}
            for col in ['data','ticket','nome','preco','quantidade','compra_venda','onde','tipo','country','categoria','obs']:
                new_val = row[col]
                old_val = orig[col]
                if col == 'data':
                    new_val = pd.to_datetime(new_val, errors='coerce')
                    if pd.isna(new_val):
                        continue
                    new_val = new_val.strftime("%Y-%m-%d")
                    old_val = pd.to_datetime(old_val).strftime("%Y-%m-%d")
                if str(new_val) != str(old_val):
                    updates[col] = new_val
            if any(k in updates for k in ['preco','quantidade','compra_venda']):
                preco = float(updates.get('preco', row['preco']))
                qtd = float(updates.get('quantidade', row['quantidade']))
                oper = updates.get('compra_venda', row['compra_venda'])
                val = preco * qtd
                if oper == 'Venda':
                    val = -val
                updates['valor_investido'] = float(val)
            if 'data' in updates:
                dt = pd.to_datetime(updates['data'])
                updates['month'] = int(dt.month)
                updates['year'] = int(dt.year)
            if updates:
                update_movimento(conn, int(row['id']), updates)
        st.success("Alterações salvas.")
        st.rerun()

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button("⬇️ CSV filtrado", data=edited.to_csv(index=False).encode('utf-8'), file_name="movimentos_filtrado.csv", mime="text/csv")
    with exp2:
        st.download_button("⬇️ Backup completo (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="movimentos_backup_completo.csv", mime="text/csv")

# ---- Dashboards (sem gráficos + conversão BRL/USD)
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

# ---- Posições
with tab4:
    st.subheader("Posições e P&L (valores em BRL)")
    df = load_df(conn)
    if df.empty:
        st.info("Sem dados ainda.")
    else:
        pos, usd_brl = build_positions(df)

        if isinstance(pos, pd.DataFrame) and not pos.empty:
            k1, k2 = st.columns(2)
            k1.metric("Valor total (BRL)", f"{pos['valor_atual_brl'].sum():,.2f}")
            if usd_brl and not np.isnan(usd_brl):
                k2.metric("USD/BRL (yfinance)", f"{usd_brl:,.4f}")

            show_cols = ['ticket','qtd_total','preco_medio','preco_atual','valor_atual_brl','aporte','pnl_brl','pnl_pct','country']
            st.dataframe(pos[show_cols], use_container_width=True)

            top = pos.sort_values('pnl_brl', ascending=False).head(10)
            fig1, ax1 = plt.subplots(figsize=(6,3))
            ax1.barh(top['ticket'], top['pnl_brl'])
            ax1.set_title('Top P&L (BRL)')
            ax1.set_xlabel('BRL')
            ax1.set_ylabel('Ticker')
            st.pyplot(fig1, use_container_width=False)
        else:
            st.info("Não foi possível calcular posições ainda.")

# ---- Editar/Excluir
with tab5:
    st.subheader("Editar ou excluir movimentos (modo formulário)")
    df = load_df(conn)
    if df.empty:
        st.info("Sem dados.")
    else:
        row = st.selectbox("Selecione um ID", df['id'].tolist())
        reg = df[df['id']==row].iloc[0].to_dict()
        st.write("Registro atual:")
        st.json(reg)

        _safe = pd.to_datetime(reg['data'], errors='coerce')
        if pd.isna(_safe): _safe = datetime.today()

        with st.expander("Editar campos"):
            e_data = st.date_input("Data", value=_safe)
            e_preco = st.number_input("Preço", value=float(reg['preco']), step=0.01)
            e_quantidade = st.number_input("Quantidade", value=float(reg['quantidade']), step=1.0)
            e_compra_venda = st.selectbox("Operação", ["Compra","Venda"], index=0 if reg['compra_venda']=="Compra" else 1)
            e_onde = st.text_input("Onde", value=reg.get('onde') or "")
            e_tipo = st.text_input("Tipo", value=reg.get('tipo') or "")
            e_obs = st.text_area("Observações", value=reg.get('obs') or "")
            e_country = st.selectbox("País", ["Brasil","USA","Crypto"], index=0 if (reg.get('country') or "Brasil")=="Brasil" else (1 if (reg.get('country') or "")=="USA" else 2))
            e_categoria = st.text_input("Categoria", value=reg.get('categoria') or "RV")

            if st.button("Salvar edição"):
                novo_valor = float(e_preco) * float(e_quantidade)
                if e_compra_venda == "Venda":
                    novo_valor = -novo_valor
                updates = {
                    'data': pd.to_datetime(e_data).strftime("%Y-%m-%d"),
                    'preco': float(e_preco),
                    'quantidade': float(e_quantidade),
                    'valor_investido': float(novo_valor),
                    'compra_venda': e_compra_venda,
                    'onde': e_onde.strip(),
                    'tipo': e_tipo.strip(),
                    'obs': e_obs.strip(),
                    'country': e_country.strip(),
                    'categoria': e_categoria.strip(),
                    'month': pd.to_datetime(e_data).month,
                    'year': pd.to_datetime(e_data).year
                }
                update_movimento(conn, int(row), updates)
                st.success("Registro atualizado.")
                st.rerun()

    st.divider()
    if st.button("🗑️ Excluir registro selecionado"):
        delete_movimento(conn, int(row))
        st.warning("Registro excluído.")
        st.rerun()
