# app.py — v3.6.2
# - Nova coluna 'ticker_oficial' (migrada automaticamente) e usada com prioridade para buscar preços no Yahoo Finance.
# - Editor salva 'ticker_oficial' corretamente.
# - Posições exibem preço atual, valor atual (BRL) e P&L.
# - Dashboards com USD/BRL automático (yfinance) e opção manual.
# - Movimentos: edição inline (inclui valor_investido e ticker_oficial), exclusão por checkbox e edição em massa de 'onde'.
# - Login por PIN via APP_PIN (default: 1234).

import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

DB_PATH   = "invest.db"
SEED_PATH = "seed_investimentos.csv"
REQUIRE_PIN = os.getenv("APP_PIN", "1234")

st.set_page_config(page_title="Controle de Investimentos – v3.6.2", page_icon="📊", layout="wide")

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

def ensure_column_ticker_oficial(conn):
    cur = conn.execute("PRAGMA table_info(movimentos)")
    cols = [r[1].lower() for r in cur.fetchall()]
    if "ticker_oficial" not in cols:
        conn.execute("ALTER TABLE movimentos ADD COLUMN ticker_oficial TEXT;")
        conn.commit()

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
        last = close.iloc[-1]
        if isinstance(last, pd.Series):
            return float(last.get('BRL=X', np.nan))
        return float(last)
    except Exception:
        return np.nan

def guess_ticker_symbol(row):
    t = str(row.get('ticket','')).upper().strip()
    country = str(row.get('country','')).strip().lower()
    tipo = str(row.get('tipo','')).strip().lower()
    # Crypto comuns em USD
    if 'crypto' in country or 'cripto' in country or 'crypto' in tipo or 'cripto' in tipo:
        if t.endswith('-USD'): return t
        return f"{t}-USD"
    # Brasil / B3
    if 'brasil' in country or 'b3' in tipo or 'fii' in tipo or 'brasil ações' in tipo:
        if t.endswith('.SA'): return t
        return t + '.SA'
    # USA default
    return t

def pick_fetch_symbol(row):
    tk = str(row.get('ticker_oficial','') or '').strip()
    if tk:
        return tk
    return guess_ticker_symbol(row)

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
    tmp['ticker_fetch'] = tmp.apply(pick_fetch_symbol, axis=1)
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
    positions['moeda'] = np.where(positions['country'].str.lower().eq('brasil'), 'BRL',
                          np.where(positions['country'].str.lower().eq('crypto'), 'USD', 'USD'))
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
ensure_column_ticker_oficial(conn)
seed_if_empty(conn)

st.title("Controle de Investimentos – v3.6.2")

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
    with col2:
        compra_venda = st.selectbox("Operação", ["Compra", "Venda"], key="novo_op")
        onde = st.text_input("Onde (corretora/plataforma)", key="novo_onde")
        tipo = st.text_input("Tipo (ex: Brasil Ações, EUA Ações, FII, ETF)", key="novo_tipo")
        country = st.selectbox("País", ["Brasil","USA","Crypto"], index=0, key="novo_pais")
        categoria = st.text_input("Categoria (ex: RV, RF, FII, ETF)", value="RV", key="novo_cat")
        obs = st.text_area("Observações", key="novo_obs")
    ticker_oficial = st.text_input("ticker_oficial (opcional, ex.: XPML11.SA, VOO, BTC-USD)", key="novo_tkoff")

    if st.button("Salvar", key="novo_salvar"):
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
            "ticker_oficial": ticker_oficial.strip() if ticker_oficial else None
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
        fdf = fdf[fdf['ticket'].str.contains(filtro_ticket.strip().upper(), na=False)]
    if filtro_ano:
        fdf = fdf[fdf['year'].isin(filtro_ano)]
    if filtro_operacao:
        fdf = fdf[fdf['compra_venda'].isin(filtro_operacao)]
    if filtro_onde:
        fdf = fdf[fdf['onde'].str.contains(filtro_onde, na=False, case=False)]

    st.caption("✅ Edite as células (inclui **valor_investido** e **ticker_oficial**), marque 'Excluir' para remover linhas e use os botões abaixo.")

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
            "ticker_oficial": st.column_config.TextColumn("ticker_oficial", help="Ticker do Yahoo Finance (ex.: PETR4.SA, VOO, BTC-USD)")
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
                            'compra_venda','onde','tipo','country','categoria','obs','ticker_oficial']:
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

# ---- Dashboards (FX manual/auto) ----
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
            if country == 'USA' or country == 'Crypto':
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

        st.markdown("---")
        st.markdown("### Preços atuais (yfinance) e comparação com preço médio")
        pos_tmp, usd_brl_tmp = build_positions(df)
        if not pos_tmp.empty:
            tbl = pos_tmp[['ticket','preco_medio','preco_atual','valor_atual_brl','aporte','pnl_brl','pnl_pct']].copy()
            st.dataframe(tbl, use_container_width=True)
        else:
            st.info("Não foi possível calcular posições ainda.")

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
            k1.metric("Valor total (BRL)", f"{pos['valor_atual_brl'].sum(skipna=True):,.2f}")
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
