
# app.py — v3.6.1
# - Corrige bug de sintaxe na seção "Crypto" (Dashboards).
# - Mantém coluna ticker_oficial e preços pelo Yahoo Finance com fallback.
# - Conversão USD/BRL via yfinance (ou manual).
#
# Observação: este app assume Python 3.10+ e Streamlit 1.30+.

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

DB_PATH   = "invest.db"
SEED_PATH = "seed_investimentos.csv"
REQUIRE_PIN = os.getenv("APP_PIN", "1234")

st.set_page_config(page_title="Controle de Investimentos – v3.6.1", page_icon="📊", layout="wide")

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
    # Garantir coluna ticker_oficial (migração leve)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(movimentos)")]
    if "ticker_oficial" not in cols:
        try:
            conn.execute("ALTER TABLE movimentos ADD COLUMN ticker_oficial TEXT;")
            conn.commit()
        except Exception:
            pass

def seed_if_empty(conn):
    cur = conn.execute("SELECT COUNT(*) FROM movimentos")
    count = cur.fetchone()[0]
    if count == 0 and os.path.exists(SEED_PATH):
        df = pd.read_csv(SEED_PATH, parse_dates=['data'])
        ensure_cols = ['data','ticket','nome','preco','quantidade','valor_investido','compra_venda',
                       'onde','tipo','obs','country','categoria','month','year','ticker_oficial']
        for c in ensure_cols:
            if c not in df.columns: df[c] = np.nan
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df['month'] = df['data'].dt.month
        df['year']  = df['data'].dt.year
        df = df.sort_values('data')
        df.to_sql('movimentos', conn, if_exists='append', index=False)

def load_df(conn):
    return pd.read_sql_query(
        "SELECT * FROM movimentos ORDER BY datetime(data) ASC",
        conn, parse_dates=['data']
    )

def insert_movimento(conn, row: dict):
    conn.execute("""
        INSERT INTO movimentos 
        (data, ticket, nome, preco, quantidade, valor_investido, compra_venda, onde, tipo, obs, country, categoria, month, year, ticker_oficial)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['data'], row['ticket'], row['nome'], row['preco'], row['quantidade'], row['valor_investido'],
        row['compra_venda'], row['onde'], row['tipo'], row['obs'], row['country'], row['categoria'],
        row['month'], row['year'], row.get('ticker_oficial')
    ))
    conn.commit()

def update_movimento(conn, row_id: int, updates: dict):
    if not updates:
        return
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

def map_crypto_symbol(ticket: str):
    t = str(ticket).strip().upper()
    common = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    return common.get(t, t)

def guess_ticker_symbol(ticket: str, country: str, tipo: str = ""):
    t = str(ticket).upper().strip()
    country = (country or "").strip().lower()
    tipo = (tipo or "").strip().lower()
    if 'crypto' in country or 'cripto' in country:
        return map_crypto_symbol(t)
    if 'brasil' in country or 'b3' in tipo or 'fii' in tipo or 'brasil ações' in tipo:
        return t if t.endswith('.SA') else t + '.SA'
    return t

def suggest_official_from(ticket: str, country: str):
    return guess_ticker_symbol(ticket, country)

# ---------------------- Prices ----------------------
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

# ---------------------- Build positions ----------------------
def build_positions(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), np.nan

    tmp = df.copy()
    tmp['ticker_used'] = tmp.apply(
        lambda r: r['ticker_oficial'] if pd.notna(r.get('ticker_oficial')) and str(r.get('ticker_oficial')).strip() != "" 
        else guess_ticker_symbol(r.get('ticket',''), r.get('country',''), r.get('tipo','')), axis=1
    )

    qnum = pd.to_numeric(tmp['quantidade'], errors='coerce')
    tmp['qtd_signed'] = np.where(tmp['compra_venda']=='Venda', -qnum, qnum)

    agg = tmp.groupby(['ticket','country'], dropna=False).agg(
        qtd_total=('qtd_signed','sum'),
        aporte=('valor_investido','sum')
    ).reset_index()

    compras = tmp[tmp['compra_venda'] == 'Compra'].groupby('ticket').apply(
        lambda g: (pd.to_numeric(g['preco'], errors='coerce') * pd.to_numeric(g['quantidade'], errors='coerce')).sum()
                  / max(pd.to_numeric(g['quantidade'], errors='coerce').sum(), 1e-9)
    ).rename('preco_medio').reset_index()

    positions = agg.merge(compras, on='ticket', how='left')

    fetch_map = tmp.drop_duplicates('ticket')[['ticket','ticker_used']]
    positions = positions.merge(fetch_map, on='ticket', how='left')

    tickers_unique = positions['ticker_used'].dropna().unique().tolist()
    price_map, usd_brl = fetch_prices(tickers_unique)

    positions['preco_atual'] = pd.to_numeric(positions['ticker_used'].map(price_map), errors='coerce')
    positions['qtd_total']  = pd.to_numeric(positions['qtd_total'], errors='coerce').fillna(0.0)
    positions['aporte']     = pd.to_numeric(positions['aporte'], errors='coerce')

    positions['moeda'] = np.where(positions['country'].str.lower().eq('brasil'), 'BRL', 'USD')
    positions['valor_atual_moeda'] = positions['preco_atual'] * positions['qtd_total']

    def to_brl(row):
        if row['moeda'] == 'USD' and pd.notna(row['valor_atual_moeda']) and pd.notna(usd_brl):
            return row['valor_atual_moeda'] * usd_brl
        if row['moeda'] == 'USD' and pd.notna(row['valor_atual_moeda']) and pd.isna(usd_brl):
            return np.nan
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

st.title("Controle de Investimentos – v3.6.1")

tab1, tab2, tab3, tab4 = st.tabs(["➕ Novo", "📋 Movimentos", "📊 Dashboards", "📦 Posições"])

# ---- Novo
with tab1:
    st.subheader("Adicionar movimento")
    col1, col2 = st.columns(2)
    with col1:
        data = st.date_input("Data", value=datetime.today(), key="novo_data")
        ticket = st.text_input("Ticker (ex: PETR4, AAPL)", key="novo_ticker")
        nome = st.text_input("Nome do ativo (opcional)", key="novo_nome")
        preco = st.number_input("Preço por unidade", min_value=0.0, step=0.01, format="%.6f", key="novo_preco")
        quantidade = st.number_input("Quantidade", min_value=0.0, step=1.0, format="%.6f", key="novo_qtd")
        country = st.selectbox("País", ["Brasil","USA","Crypto"], index=0, key="novo_pais")
    with col2:
        compra_venda = st.selectbox("Operação", ["Compra", "Venda"], key="novo_op")
        onde = st.text_input("Onde (corretora/plataforma)", key="novo_onde")
        tipo = st.text_input("Tipo (ex: Brasil Ações, EUA Ações, FII, ETF)", key="novo_tipo")
        categoria = st.text_input("Categoria (ex: RV, RF, FII, ETF)", value="RV", key="novo_cat")
        obs = st.text_area("Observações", key="novo_obs")
        sugestao = suggest_official_from(ticket, country) if ticket else ""
        ticker_oficial = st.text_input("Ticker oficial (Yahoo Finance)", value=sugestao, key="novo_ticker_oficial",
                                       help="Ex.: PETR4.SA, AAPL, BTC-USD. Pode ajustar depois na aba Movimentos.")

    if st.button("Salvar", key="novo_salvar"):
        valor_investido = float(preco) * float(quantidade)
        if compra_venda == "Venda":
            valor_investido = -valor_investido
        row = {
            "data": pd.to_datetime(data).strftime("%Y-%m-%d"),
            "ticket": (ticket or "").strip().upper(),
            "nome": (nome or "").strip(),
            "preco": float(preco),
            "quantidade": float(quantidade),
            "valor_investido": float(valor_investido),
            "compra_venda": compra_venda,
            "onde": (onde or "").strip(),
            "tipo": (tipo or "").strip(),
            "obs": (obs or "").strip(),
            "country": (country or "").strip(),
            "categoria": (categoria or "").strip(),
            "ticker_oficial": (ticker_oficial or "").strip() or suggest_official_from(ticket, country),
        }
        row["month"] = pd.to_datetime(row["data"]).month
        row["year"]  = pd.to_datetime(row["data"]).year

        if row["ticket"] == "" or row["preco"] <= 0 or row["quantidade"] <= 0:
            st.error("Preencha Ticker, Preço (> 0) e Quantidade (> 0).")
        else:
            insert_movimento(conn, row)
            st.success(f"Movimento salvo para {row['ticket']} em {row['data']}.")

# ---- Movimentos
with tab2:
    st.subheader("Histórico de movimentos")
    df = load_df(conn)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        filtro_ticket = st.text_input("Filtrar por Ticker", key="flt_ticker")
    with c2:
        filtro_ano = st.multiselect("Ano", sorted(df['year'].dropna().unique().tolist()), key="flt_ano")
    with c3:
        filtro_operacao = st.multiselect("Operação", ["Compra","Venda"], key="flt_op")
    with c4:
        filtro_onde = st.text_input("Filtrar por 'Onde'", key="flt_onde")

    fdf = df.copy()
    if filtro_ticket:
        fdf = fdf[fdf['ticket'].str.contains((filtro_ticket or "").strip().upper(), na=False)]
    if filtro_ano:
        fdf = fdf[fdf['year'].isin(filtro_ano)]
    if filtro_operacao:
        fdf = fdf[fdf['compra_venda'].isin(filtro_operacao)]
    if filtro_onde:
        fdf = fdf[fdf['onde'].str.contains(filtro_onde, na=False, case=False)]

    st.caption("✅ Edite as células (inclui **valor_investido** e **ticker_oficial**), marque 'Excluir' e use os botões.")

    onde_options = sorted(list(set(df['onde'].dropna().tolist() + ["XP","Rico","Nubank","Clear","Avenue","Biscoint"])))

    fdf_view = fdf[['id','data','ticket','nome','preco','quantidade','valor_investido',
                    'compra_venda','onde','tipo','country','categoria','ticker_oficial','obs']].copy()
    fdf_view.insert(1, "excluir", False)

    edited = st.data_editor(
        fdf_view,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "excluir": st.column_config.CheckboxColumn("Excluir"),
            "data": st.column_config.DateColumn("data"),
            "preco": st.column_config.NumberColumn("preco", format="%.6f"),
            "quantidade": st.column_config.NumberColumn("quantidade", format="%.6f"),
            "valor_investido": st.column_config.NumberColumn("valor_investido", format="%.6f", help="Pode editar manualmente."),
            "compra_venda": st.column_config.SelectboxColumn("compra_venda", options=["Compra","Venda"]),
            "onde": st.column_config.SelectboxColumn("onde", options=onde_options),
            "country": st.column_config.SelectboxColumn("País", options=["Brasil","USA","Crypto"]),
            "ticker_oficial": st.column_config.TextColumn("ticker_oficial", help="Ticker no Yahoo Finance (ex.: PETR4.SA, AAPL, BTC-USD)"),
        },
        use_container_width=True,
        key="mov_editor"
    )

    def to_float_safe(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(" ", "").replace(",", "")
            try:
                return float(s)
            except Exception:
                return np.nan
        return np.nan

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("💾 Salvar alterações", key="btn_save_inline"):
            merged = edited.merge(fdf[['id']], on='id', how='left')
            for _, row in merged.iterrows():
                orig = fdf[fdf['id']==row['id']].iloc[0]
                updates = {}
                for col in ['data','ticket','nome','preco','quantidade','valor_investido',
                            'compra_venda','onde','tipo','country','categoria','ticker_oficial','obs']:
                    new_val = row[col]
                    old_val = orig[col]
                    if col == 'data':
                        new_val = pd.to_datetime(new_val, errors='coerce')
                        if pd.isna(new_val):
                            continue
                        new_val = new_val.strftime("%Y-%m-%d")
                        old_val = pd.to_datetime(old_val).strftime("%Y-%m-%d")
                    changed = (pd.isna(new_val) != pd.isna(old_val)) or (str(new_val) != str(old_val))
                    if changed:
                        updates[col] = new_val

                changed_inputs = any(k in updates for k in ['preco','quantidade','compra_venda'])
                gave_value = ('valor_investido' in updates) and (not pd.isna(to_float_safe(updates['valor_investido'])))
                if changed_inputs and not gave_value:
                    preco = to_float_safe(updates.get('preco', row['preco']))
                    qtd   = to_float_safe(updates.get('quantidade', row['quantidade']))
                    oper  = updates.get('compra_venda', row['compra_venda'])
                    if pd.isna(preco): preco = 0.0
                    if pd.isna(qtd):   qtd   = 0.0
                    val = preco * qtd
                    if oper == 'Venda':
                        val = -val
                    updates['valor_investido'] = float(val)

                if 'data' in updates:
                    dt = pd.to_datetime(updates['data'])
                    updates['month'] = int(dt.month)
                    updates['year']  = int(dt.year)

                if updates:
                    for num_col in ['preco','quantidade','valor_investido']:
                        if num_col in updates and updates[num_col] is not None and not pd.isna(updates[num_col]):
                            updates[num_col] = to_float_safe(updates[num_col])
                    update_movimento(conn, int(row['id']), updates)
            st.success("Alterações salvas.")
            st.rerun()

    with colB:
        ids_excluir = edited.loc[edited['excluir']==True, 'id'].dropna().astype(int).tolist()
        if st.button(f"🗑️ Excluir selecionados ({len(ids_excluir)})", disabled=(len(ids_excluir)==0), key="btn_delete"):
            for rid in ids_excluir:
                delete_movimento(conn, rid)
            st.warning(f"{len(ids_excluir)} linha(s) excluída(s).")
            st.rerun()

    with colC:
        if st.button("☑️ Marcar/Desmarcar todos visíveis", key="btn_toggle_all"):
            edited['excluir'] = ~edited['excluir'].astype(bool)
            st.session_state['mov_editor'] = edited
            st.rerun()

    with st.expander("Edição em massa de 'onde' (aplica no filtro atual)"):
        novo_onde = st.selectbox("Definir 'onde' para TODAS as linhas filtradas:", options=onde_options, key="mass_onde")
        if st.button("Aplicar 'onde' nas linhas filtradas", key="btn_apply_onde"):
            for rid in fdf['id'].tolist():
                update_movimento(conn, int(rid), {"onde": novo_onde})
            st.success(f"'onde' atualizado em {len(fdf)} linha(s).")
            st.rerun()

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "⬇️ CSV filtrado",
            data=edited.drop(columns=['excluir']).to_csv(index=False).encode('utf-8'),
            file_name="movimentos_filtrado.csv", mime="text/csv", key="dl_csv_filtrado"
        )
    with exp2:
        st.download_button(
            "⬇️ Backup completo (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="movimentos_backup_completo.csv", mime="text/csv", key="dl_csv_full"
        )

# ---- Dashboards (sem gráficos + FX override) ----
with tab3:
    st.subheader("Dashboards")
    df = load_df(conn)
    if df.empty:
        st.info("Sem dados ainda.")
    else:
        ddf = df.copy()
        ddf['country_norm'] = ddf['country'].apply(norm_country)

        fx_auto = get_usd_brl()
        col_fx1, col_fx2 = st.columns([1,2])
        with col_fx1:
            use_manual_default = pd.isna(fx_auto)
            use_manual = st.toggle("Usar cotação manual", value=use_manual_default,
                                   help="Se o yfinance estiver fora, ative e digite a cotação.", key="fx_toggle")
        with col_fx2:
            fx_init = float(fx_auto) if pd.notna(fx_auto) else 5.00
            fx_manual = st.number_input("USD/BRL", value=fx_init, step=0.01, format="%.4f",
                                        disabled=not use_manual, key="fx_manual")
        usd_brl = fx_manual if use_manual else fx_auto

        if pd.notna(usd_brl):
            st.caption(f"USD/BRL em uso: {usd_brl:,.4f} ({'manual' if use_manual else 'yfinance'})")
        else:
            st.caption("USD/BRL indisponível no momento (defina manualmente para converter os valores de USA).")

        def split_values(row):
            country = row['country_norm']
            val = float(row['valor_investido']) if pd.notna(row['valor_investido']) else 0.0
            if country == 'USA':
                brl = val * usd_brl if pd.notna(usd_brl) and usd_brl != 0 else np.nan
                return pd.Series({'valor_local': val, 'moeda_local': 'USD', 'valor_brl': brl})
            else:
                return pd.Series({'valor_local': val, 'moeda_local': 'BRL', 'valor_brl': val})

        ddf = pd.concat([ddf, ddf.apply(split_values, axis=1)], axis=1)

        total_investido_brl = ddf['valor_brl'].sum(skipna=True)
        st.metric("Valor total investido (BRL)", f"{total_investido_brl:,.2f}")

        st.markdown("### Por corretora/plataforma (com conversão)")
        by_onde_brl = ddf.groupby('onde', dropna=False)['valor_brl'].sum(min_count=1).reset_index().rename(columns={'valor_brl':'Aporte (BRL)'})
        predominante = ddf.groupby(['onde','moeda_local'])['valor_local'].sum(min_count=1).reset_index()
        idx = predominante.groupby('onde')['valor_local'].idxmax()
        moeda_pred = predominante.loc[idx, ['onde','moeda_local']]
        by_onde_local = ddf.groupby('onde')['valor_local'].sum(min_count=1).reset_index().rename(columns={'valor_local':'Aporte (Local)'})
        table_onde = by_onde_brl.merge(by_onde_local, on='onde', how='left').merge(moeda_pred, on='onde', how='left')
        table_onde = table_onde.rename(columns={'moeda_local':'Moeda Local (predominante)'})
        table_onde = table_onde.sort_values('Aporte (BRL)', ascending=False, na_position='last')
        st.dataframe(table_onde, use_container_width=True)

        st.markdown("### Por país (Local vs BRL)")
        by_country = ddf.groupby('country_norm').agg(
            aporte_local=('valor_local','sum'),
            aporte_brl=('valor_brl','sum'),
            moeda=('moeda_local', lambda x: x.value_counts().index[0] if len(x)>0 else 'BRL')
        ).reset_index().rename(columns={'country_norm':'País','aporte_local':'Aporte (Local)','aporte_brl':'Aporte (BRL)','moeda':'Moeda'})
        by_country = by_country.sort_values('Aporte (BRL)', ascending=False, na_position='last')
        st.dataframe(by_country, use_container_width=True)

        st.markdown("### Preços atuais (yfinance) e comparação com preço médio")
        tmp_tickers = ddf.drop_duplicates('ticket')[['ticket','country','tipo','ticker_oficial']].copy()
        tmp_tickers['ticker_used'] = tmp_tickers.apply(
            lambda r: r['ticker_oficial'] if pd.notna(r.get('ticker_oficial')) and str(r.get('ticker_oficial')).strip() != "" 
            else guess_ticker_symbol(r.get('ticket',''), r.get('country',''), r.get('tipo','')), axis=1
        )
        tickers_unique = tmp_tickers['ticker_used'].dropna().unique().tolist()
        price_map, usd_brl_now = fetch_prices(tickers_unique)

        compras = ddf[ddf['compra_venda'] == 'Compra'].copy()
        if compras.empty:
            preco_medio = pd.DataFrame(columns=['ticket','preco_medio'])
        else:
            preco_medio = compras.groupby('ticket').apply(
                lambda g: (pd.to_numeric(g['preco'], errors='coerce') * pd.to_numeric(g['quantidade'], errors='coerce')).sum()
                          / max(pd.to_numeric(g['quantidade'], errors='coerce').sum(), 1e-9)
            ).rename('preco_medio').reset_index()

        tab = tmp_tickers.merge(preco_medio, on='ticket', how='left')
        tab['moeda'] = np.where(tab['country'].str.lower().eq('brasil'), 'BRL', 'USD')
        tab['preco_atual_local'] = pd.to_numeric(tab['ticker_used'].map(price_map), errors='coerce')

        def to_brl_price(row):
            if row['moeda'] == 'USD' and pd.notna(row['preco_atual_local']) and pd.notna(usd_brl_now):
                return row['preco_atual_local'] * usd_brl_now
            return row['preco_atual_local']

        tab['preco_atual_brl'] = tab.apply(to_brl_price, axis=1)

        tab['dif_pct'] = np.where(
            pd.notna(tab['preco_medio']) & (tab['preco_medio'] != 0),
            (tab['preco_atual_local'] - tab['preco_medio']) / tab['preco_medio'],
            np.nan
        )

        tab = tab.rename(columns={
            'ticket': 'Ticker',
            'country': 'País',
            'preco_medio': 'Preço médio (Local)',
            'preco_atual_local': 'Preço atual (Local)',
            'preco_atual_brl': 'Preço atual (BRL)',
            'dif_pct': 'Diferença % (vs. médio)'
        })
        tab = tab[['Ticker','País','moeda','Preço atual (Local)','Preço atual (BRL)','Preço médio (Local)','Diferença % (vs. médio)']]
        tab = tab.rename(columns={'moeda': 'Moeda'})
        tab = tab.sort_values('Ticker')

        st.dataframe(tab, use_container_width=True)

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
