
# 📊 Controle de Investimentos – Streamlit

Um app simples para cadastrar **compras/vendas** de ativos pelo celular e visualizar um **dashboard** com seus movimentos.

## 🚀 Como usar (local ou na nuvem)
1. **Baixe este projeto** (zip) e extraia.
2. (Opcional) Coloque seu arquivo `seed_investimentos.csv` na raiz do projeto se quiser começar com seus dados históricos.
   - Neste pacote já incluí um `seed_investimentos.csv` gerado a partir dos arquivos que você enviou.
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Rode o app:
   ```bash
   streamlit run app.py
   ```
5. Abra no celular: copie o link local que o Streamlit mostrar (ex: `http://192.168.0.10:8501`) e acesse a partir do celular conectado à mesma rede Wi‑Fi.
   - **Na nuvem (recomendado):** Faça upload deste projeto para um repositório no GitHub e publique no **Streamlit Community Cloud**. O banco `invest.db` (SQLite) ficará no diretório do app e tende a **persistir** entre execuções. Em uma **reimplantação** (deploy novo), ele pode ser recriado a partir do `seed_investimentos.csv` automaticamente.

## 🧠 Como funciona a persistência
- O app usa **SQLite** (`invest.db`) para guardar os movimentos.
- Se o banco estiver vazio no primeiro boot, ele carrega os dados iniciais do `seed_investimentos.csv`.
- Você pode fazer **backup** a qualquer momento pela aba **Movimentos**.

## 📝 Campos do cadastro
- Data, Ticker, Nome, Preço, Quantidade, Operação (Compra/Venda), Onde (corretora), Tipo (ex: Brasil Ações, FII, ETF), País, Categoria, Observações.
- O campo **Valor Investido** é calculado automaticamente como `preço × quantidade` (negativo para **Venda**).

## 🧩 Extensões futuras (ideias)
- Autenticação por senha (ex: `stauth`).
- Integração com Google Sheets ou Supabase para persistência 100% gerenciada.
- Cálculo de posição atual e preço médio por ticker.
- Importação de notas de corretagem.
- Projeção de dividendos e IR.

---

Feito para uso rápido por mobile. Qualquer ajuste, me fale que eu edito o app.
