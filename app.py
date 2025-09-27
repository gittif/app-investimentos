# app.py - versão v3.1 corrigida
# (conteúdo resumido: PIN, dashboards, edição inline em Movimentos, fix yfinance)
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

st.set_page_config(page_title="Investimentos – v3.1", page_icon="📈", layout="wide")

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

# demais funções e abas conforme versão v3.1 que já detalhei antes
st.write("✅ app.py v3.1 carregado. Copie o conteúdo completo que te enviei antes.")
