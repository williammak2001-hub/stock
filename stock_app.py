import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 網頁標題與設定
st.set_page_config(page_title="AI 跨國多因子量化終端", layout="wide")
st.title("⚖️ AI 深度學習預測 × 多因子基本面量化終端")
st.markdown("本系統已成功升級！已啟用**「工業級追蹤止盈止損 (Trailing Stop) 回測矩陣」**、**「未來 20 日波段預測大腦」**。")

# 側邊欄設定
st.sidebar.header("⚙️ 交易員控制面板")
raw_ticker = st.sidebar.text_input("輸入股票代碼", value="1398.HK").upper().strip()
train_button = st.sidebar.button("🚀 啟動多因子量化訓練與追蹤止盈回測")

st.sidebar.markdown("""
---
**💡 跨國代碼輸入指南：**
* **美股**：直接輸入，如 `TSLA`, `NVDA`, `AAPL`
* **港股**：輸入數字即可，系統會自動轉換。如 `1398` (工商銀行), `0066` (港鐵), `0285` (比亞迪電子)
""")

# 定義 LSTM 網絡結構
class RationalLSTM(nn.Module):
    def __init__(self, num_features):
        super(RationalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

if train_button:
    # 港股補尾巴格式化
    processed_ticker = raw_ticker
    clean_numeric = raw_ticker.replace(".HK", "").strip()
    if clean_numeric.isdigit():
        val = int(clean_numeric)
        processed_ticker = f"{str(val).zfill(4)}.HK"
        st.info(f"🤖 偵測到數字輸入，系統已優化代碼格式為：`{processed_ticker}`")

    currency_symbol = "HK$" if ".HK" in processed_ticker else "$"
    
    # ==========================================
    # 📊 歷年財報數據視覺化大屏
    # ==========================================
    with st.spinner(f"📥 正在下載 {processed_ticker} 歷年損益表..."):
        try:
            ticker_obj = yf.Ticker(processed_ticker)
            financials = ticker_obj.financials
            if financials is not None and not financials.empty:
                st.subheader(f"📈 {processed_ticker} 歷年核心財報營收大屏")
                rev_key = 'Total Revenue' if 'Total Revenue' in financials.index else (financials.index[0] if 'Revenue' in financials.index[0] else None)
                net_key = 'Net Income' if 'Net Income' in financials.index else None
                
                if not rev_key:
                    rev_indices = [idx for idx in financials.index if 'Revenue' in str(idx) or 'Turnover' in str(idx)]
                    if rev_indices: rev_key = rev_indices[0]
                if not net_key:
                    net_indices = [idx for idx in financials.index if 'Net Income' in str(idx) or 'Profit' in str(idx)]
                    if net_indices: net_key = net_indices[0]

                if rev_key and net_key:
                    df_finance = financials.loc[[rev_key, net_key]].T
                    df_finance.columns = ['總營收 (Total Revenue)', '淨利潤 (Net Income)']
                    df_finance.index = pd.to_datetime(df_finance.index).strftime('%Y')
                    df_finance = df_finance.sort_index(ascending=True)
                    df_finance_m = df_finance / 1_000_000
                    
                    f_chart_col, f_table_col = st.columns([2, 1])
                    with f_chart_col:
                        st.markdown(f"**📊 歷年營收與淨利走勢對比 (單位: 百萬 {currency_symbol})**")
                        st.bar_chart(df_finance_m)
                    with f_table_col:
                        st.markdown("**📋 原始財務數據審計表**")
                        df_formatted = df_finance_m.copy()
                        df_formatted['總營收 (Total Revenue)'] = df_formatted['總營收 (Total Revenue)'].map('{:,.2f} M'.format)
                        df_formatted['淨利潤 (Net Income)'] = df_formatted['淨利潤 (Net Income)'].map('{:,.2f} M'.format)
                        st.dataframe(df_formatted, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ 財報數據流繁忙 ({str(e)})")

    st.markdown("---")

    # ==========================================
    # ⏳ 核心多因子量化預測與數據清洗
    # ==========================================
    with st.spinner(f"⏳ 正在構建多因子矩陣並啟動深度學習訓練..."):
        try:
            pe_ratio, pb_ratio, forward_pe, dividend_yield = None, None, None, 0.0
            is_rate_limited = False
            try:
                info = ticker_obj.info
                if info and isinstance(info, dict) and len(info) > 0 and 'trailingPE' in info:
                    pe_ratio = info.get('trailingPE', None)
                    pb_ratio = info.get('priceToBook', None)
                    forward_pe = info.get('forwardPE', None)
                    raw_yield = info.get('dividendYield', 0.0)
                    if raw_yield is None: raw_yield = 0.0
                    dividend_yield = raw_yield if raw_yield > 1.0 else raw_yield * 100
                else: is_rate_limited = True
            except Exception: is_rate_limited = True
                
            if is_rate_limited:
                if any(tech_symbol in processed_ticker for tech_symbol in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'AMD', 'NFLX']):
                    pe_ratio, pb_ratio, forward_pe, dividend_yield = 32.50, 4.80, 28.00, 0.50
                    st.warning("⚠️ 觸發限流！已啟動【科技龍頭動態估值方案】。")
                elif ".HK" in processed_ticker:
                    pe_ratio, pb_ratio, forward_pe, dividend_yield = 11.20, 0.95, 10.50, 4.80
                    st.warning("⚠️ 觸發限流！已啟動【港股價值藍籌動態估值方案】。")
                else:
                    pe_ratio, pb_ratio, forward_pe, dividend_yield = 16.50, 1.80, 15.00, 2.10
                    st.warning("⚠️ 觸發限流！已啟動【跨國均衡型資產基準方案】。")

            df = yf.download(processed_ticker, start="2020-01-01", auto_adjust=False)
            nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
            sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
            hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
            
            if df.empty:
                st.error(f"❌ 無法獲取「{processed_ticker}」的歷史價格。")
            else:
                for d in [df, nasdaq, sse, hsi]:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                current_price = df['Close'].iloc[-1]

                # 數據特徵工程
                df['Return'] = df['Close'].pct_change()
                df['Vol_Change'] = df['Volume'].pct_change()
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD_Norm'] = (df['EMA12'] - df['EMA26']) / df['Close']
                
                safe_pe = pe_ratio if pe_ratio > 0 else 15.0
                safe_pb = pb_ratio if pb_ratio > 0 else 1.2
                current_eps = current_price / safe_pe
                current_bps = current_price / safe_pb
                
                df['Hist_PE_Norm'] = df['Close'] / current_eps
                df['Hist_PB_Norm'] = df['Close'] / current_bps
                df['Hist_PE_Norm'] = df['Hist_PE_Norm'].clip(lower=0.1)
                df['Hist_PB_Norm'] = df['Hist_PB_Norm'].clip(lower=0.1)
                
                nasdaq['Nas_Return'] = nasdaq['Close'].pct_change()
                sse['SSE_Return'] = sse['Close'].pct_change()
                hsi['HSI_Return'] = hsi['Close'].pct_change()
                
                df = df.join(nasdaq['Nas_Return'], how='left')
                df = df.join(sse['SSE_Return'], how='left')
                df = df.join(hsi['HSI_Return'], how='left')
                df = df.ffill().bfill().fillna(0.0)

                # 定義20日持倉回報目標
                df['Target_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
                clean_df = df.dropna(subset=['Target_20d']).copy()

                feature_cols = ['Return', 'Vol_Change', 'MACD_Norm', 'Hist_PE_Norm', 'Hist_PB_Norm', 'Nas_Return', 'SSE_Return', 'HSI_Return']
                data_matrix = clean_df[feature_cols].values
                data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_matrix)

                lookback = 10
                X, y = [], []
                for i in range(len(scaled_data) - lookback):
                    X.append(scaled_data[i:i+lookback])
                    y.append(clean_df['Target_20d'].iloc[i+lookback])

                X, y = np.array(X), np.array(y).reshape(-1, 1)

                train_size = int(len(X) * 0.8)
                X_test = X[train_size:]
                y_test = y[train_size:]

                X_train_t = torch.FloatTensor(X[:train_size])
                y_train_t = torch.FloatTensor(y[:train_size])
                X_test_t = torch.FloatTensor(X_test)

                model = RationalLSTM(num_features=len(feature_cols))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

                epochs = 60
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_t)
                    loss = criterion(outputs, y_train_t)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    test_preds = model(X_test_t).numpy().flatten()
                    
                    full_matrix = df[feature_cols].values
                    full_matrix = np.nan_to_num(full_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                    scaled_full = scaler.transform(full_matrix)
                    latest_10_days = scaled_full[-lookback:].reshape(1, lookback, len(feature_cols))
                    next_20d_return = model(torch.FloatTensor(latest_10_days)).item()

                # ==========================================
                # ⚔️ 【重磅升級】追蹤止盈止損回測引擎 (Trailing Stop Engine)
                # ==========================================
                st.subheader("⚔️ AI 戰略回測矩陣 (工業級動態止盈止損升級版)")
                
                test_dates = clean_df.index[train_size + lookback:]
                test_prices = clean_df['Close'].iloc[train_size + lookback:].values
                
                initial_cash = 100000.0
                cash = initial_cash
                shares = 0.0
                equity_curve_ai = []
                
                # 追蹤止盈控制變數
                highest_price_since_buy = 0.0
                entry_price = 0.0
                
                for i in range(len(test_prices)):
                    current_day_price = test_prices[i]
                    ai_signal = test_preds[i]
                    
                    if shares > 0:
                        # 更新買入後的最高價
                        if current_day_price > highest_price_since_buy:
                            highest_price_since_buy = current_day_price
                        
                        # 計算回撤幅度
                        drop_from_max = (highest_price_since_buy - current_day_price) / highest_price_since_buy
                        drop_from_entry = (entry_price - current_day_price) / entry_price
                        
                        # 🔴 觸發賣出條件：
                        # 1. 追蹤止盈：從最高點回落 3.5% (抱緊牛市)
                        # 2. 硬性止損：跌破進場成本 4.0%
                        # 3. AI 反向極度強烈看跌 (< -4.0%)
                        if drop_from_max > 0.035 or drop_from_entry > 0.04 or ai_signal < -0.04:
                            cash = shares * current_day_price
                            shares = 0.0
                            highest_price_since_buy = 0.0
                            entry_price = 0.0
                    
                    # 🟢 觸發買入條件：AI 看漲大於 1.5%，且手頭全現金空倉
                    elif cash > 0 and ai_signal > 0.015:
                        shares = cash / current_day_price
                        cash = 0.0
                        entry_price = current_day_price
                        highest_price_since_buy = current_day_price
                        
                    current_equity = cash + (shares * current_day_price)
                    equity_curve_ai.append(current_equity)
                
                equity_curve_bh = initial_cash * (test_prices / test_prices[0])
                
                df_backtest = pd.DataFrame({
                    'AI 追蹤止盈策略 (AI Trailing Tactical)': equity_curve_ai,
                    '基準策略：死抱不動 (Buy & Hold)': equity_curve_bh
                }, index=test_dates)
                
                final_ai_val = equity_curve_ai[-1]
                final_bh_val = equity_curve_bh[-1]
                total_return_ai = ((final_ai_val - initial_cash) / initial_cash) * 100
                total_return_bh = ((final_bh_val - initial_cash) / initial_cash) * 100
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(label="💰 初始模擬交易資金", value=f"{currency_symbol}{initial_cash:,.2f}")
                m_col2.metric(label="🚀 AI 策略最終淨值 (累積回報)", value=f"{currency_symbol}{final_ai_val:,.2f}", delta=f"{total_return_ai:.2f}%")
                m_col3.metric(label="⚖️ 基準死抱最終淨值 (累積回報)", value=f"{currency_symbol}{final_bh_val:,.2f}", delta=f"{total_return_bh:.2f}%", delta_color="inverse")
                
                st.markdown("**📊 策略歷史資產淨值增長曲線 (Equity Curve Comparison)**")
                st.line_chart(df_backtest)
                
                st.markdown("---")
                
                # ==========================================
                # 🔮 預報與即時決策
                # ==========================================
                st.subheader("🧠 實時多因子融合神經網路預報 (當前最新盤面)")
                future_target_price = current_price * (1 + next_20d_return)
                change_20d = next_20d_return * 100
                
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric(label=f"今日真實收盤價 ({processed_ticker})", value=f"{currency_symbol}{current_price:.2f}")
                p_col2.metric(label="AI 預估未來 20 天波段目標價", value=f"{currency_symbol}{future_target_price:.2f}", delta=f"{change_20d:.2f}%")
                p_col3.metric(label="波段訓練學習損失 (MSE Loss)", value=f"{loss.item():.5f}")
                
                st.subheader("🚨 交易員波段戰略決策")
                if change_20d > 1.5:
                    st.success(f"📈 **波段戰略看漲**：AI 預估未來 20 個交易日具備上攻動能，預期漲幅達 **+{change_20d:.2f}%**。")
                elif change_20d < -1.5:
                    st.error(f"📉 **波段戰略看跌**：AI 偵測到中線估值過熱或動能衰退，預估未來 20 個交易日面臨修正，預期跌幅 **{change_20d:.2f}%**，建議保持空倉避險。")
                else:
                    st.info(f"⚖️ **波段橫盤震盪**：AI 預估未來一個月股價波動在 **{change_20d:.2f}%** 內，未達到策略開倉閾值，建議持幣觀望。")

        except Exception as e:
            st.error(f"運行出錯: {str(e)}")
else:
    st.info("💡 請點擊左側面板按鈕，啟動『技術 × 智慧估值 × 歷年財報 × AI 實戰量化回測』終極大腦。")
