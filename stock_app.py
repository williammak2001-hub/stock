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
st.markdown("本系統已成功升級！已融合**「動態基準降級策略」**、**「港股動態長度校正」**與最新**「歷年財報營收動態視覺化大屏」**。")

# 側邊欄設定
st.sidebar.header("⚙️ 交易員控制面板")
raw_ticker = st.sidebar.text_input("輸入股票代碼", value="0066.HK").upper().strip()
train_button = st.sidebar.button("🚀 啟動多因子量化訓練")

st.sidebar.markdown("""
---
**💡 跨國代碼輸入指南：**
* **美股**：直接輸入，如 `TSLA`, `NVDA`, `AAPL`
* **港股**：輸入數字即可，系統會自動轉換。如 `0066` (港鐵), `0285` 或 `00285` (比亞迪電子), `0700` (騰訊)
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
    # 港股 4 位數/5 位數長度校正防呆
    processed_ticker = raw_ticker
    clean_numeric = raw_ticker.replace(".HK", "").strip()
    if clean_numeric.isdigit():
        val = int(clean_numeric)
        processed_ticker = f"{str(val).zfill(4)}.HK"
        st.info(f"🤖 偵測到數字輸入，系統已啟動長度校正，優化代碼格式為標準 Yahoo 港股：`{processed_ticker}`")

    currency_symbol = "HK$" if ".HK" in processed_ticker else "$"
    
    # ==========================================
    # 📊 【全新功能】歷年財報數據動態拉取與視覺化
    # ==========================================
    with st.spinner(f"📥 正在穿透財務數據庫，下載 {processed_ticker} 歷年損益表..."):
        try:
            ticker_obj = yf.Ticker(processed_ticker)
            # 獲取年度損益表 (Annual Financials)
            financials = ticker_obj.financials
            
            if financials is not None and not financials.empty:
                st.subheader(f"📈 {processed_ticker} 歷年核心財報營收大屏")
                
                # 提取「總營收」與「淨利潤」
                # yfinance 的欄位名稱有時會因版本而異，做個安全相容防護
                rev_key = 'Total Revenue' if 'Total Revenue' in financials.index else (financials.index[0] if 'Revenue' in financials.index[0] else None)
                net_key = 'Net Income' if 'Net Income' in financials.index else None
                
                # 自動搜尋相似欄位
                if not rev_key:
                    rev_indices = [idx for idx in financials.index if 'Revenue' in str(idx) or 'Turnover' in str(idx)]
                    if rev_indices: rev_key = rev_indices[0]
                if not net_key:
                    net_indices = [idx for idx in financials.index if 'Net Income' in str(idx) or 'Profit' in str(idx)]
                    if net_indices: net_key = net_indices[0]

                if rev_key and net_key:
                    # 提取數據並轉置，以年份為 Index
                    df_finance = financials.loc[[rev_key, net_key]].T
                    df_finance.columns = ['總營收 (Total Revenue)', '淨利潤 (Net Income)']
                    
                    # 將日期格式化為年份字串 (如 2023, 2024)
                    df_finance.index = pd.to_datetime(df_finance.index).strftime('%Y')
                    df_finance = df_finance.sort_index(ascending=True) # 依年份正序排列
                    
                    # 轉換單位為「百萬元 (Millions)」讓圖表更美觀
                    df_finance_m = df_finance / 1_000_000
                    
                    # 建立 Streamlit 兩欄佈局展示
                    f_chart_col, f_table_col = st.columns([2, 1])
                    
                    with f_chart_col:
                        st.markdown(f"**📊 歷年營收與淨利走勢對比 (單位: 百萬 {currency_symbol})**")
                        # 繪製並排長條圖
                        st.bar_chart(df_finance_m)
                        
                    with f_table_col:
                        st.markdown("**📋 原始財務數據審計表**")
                        # 格式化表格數字加千分位
                        df_formatted = df_finance_m.copy()
                        df_formatted['總營收 (Total Revenue)'] = df_formatted['總營收 (Total Revenue)'].map('{:,.2f} M'.format)
                        df_formatted['淨利潤 (Net Income)'] = df_formatted['淨利潤 (Net Income)'].map('{:,.2f} M'.format)
                        st.dataframe(df_formatted, use_container_width=True)
                else:
                    st.warning("⚠️ 財報解析成功，但未找到標準的 Revenue 或 Net Income 欄位項目。")
            else:
                st.warning("⚠️ 該資產未在交易所披露標準的年度損益表，跳過財報視覺化面板。")
        except Exception as e:
            st.warning(f"⚠️ 財報數據流暫時繁忙 ({str(e)})，優先保護底層 AI 核心運作。")

    st.markdown("---")

    # ==========================================
    # ⏳ 核心多因子量化預測大腦
    # ==========================================
    with st.spinner(f"⏳ 正在同步全球大盤，構建『技術 × 智慧估值』多因子矩陣並訓練 AI..."):
        try:
            # 1. 嘗試抓取即時基本面估值快照
            pe_ratio = None
            pb_ratio = None
            forward_pe = None
            dividend_yield = 0.0
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
                else:
                    is_rate_limited = True
            except Exception:
                is_rate_limited = True
                
            # 動態防禦基準
            if is_rate_limited:
                if any(tech_symbol in processed_ticker for tech_symbol in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'AMD', 'NFLX']):
                    st.warning("⚠️ 觸發流量限制！已為美股高成長資產啟動【科技龍頭動態估值錨定方案】。")
                    pe_ratio = 32.50
                    pb_ratio = 4.80
                    forward_pe = 28.00
                    dividend_yield = 0.50
                elif ".HK" in processed_ticker:
                    st.warning("⚠️ 觸發流量限制！已為香港本地資產啟動【港股價值藍籌動態估值錨定方案】。")
                    pe_ratio = 11.20
                    pb_ratio = 0.95
                    forward_pe = 10.50
                    dividend_yield = 4.80
                else:
                    st.warning("⚠️ 觸發流量限制！已啟動【跨國均衡型資產基準估值錨定方案】。")
                    pe_ratio = 16.50
                    pb_ratio = 1.80
                    forward_pe = 15.00
                    dividend_yield = 2.10

            # 2. 歷史數據抓取
            df = yf.download(processed_ticker, start="2020-01-01", auto_adjust=False)
            nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
            sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
            hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
            
            if df.empty:
                st.error(f"❌ 無法獲取「{processed_ticker}」的歷史價格，請確認代碼是否正確。")
            else:
                for d in [df, nasdaq, sse, hsi]:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                current_price = df['Close'].iloc[-1]

                # 估值快照顯示
                status_suffix = " (🤖 智慧動態錨定值)" if is_rate_limited else " (📊 實時市場數據)"
                st.subheader(f"📊 {processed_ticker} 當前基本面估值快照{status_suffix}")
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                
                with f_col1:
                    pe_str = f"{pe_ratio:.2f} 倍" if pe_ratio else "N/A"
                    st.metric(label="📊 歷史滾動本益比 (Trailing P/E)", value=pe_str)
                with f_col2:
                    pb_str = f"{pb_ratio:.2f} 倍" if pb_ratio else "N/A"
                    st.metric(label="📕 股價淨值比 (P/B Ratio)", value=pb_str)
                with f_col3:
                    f_pe_str = f"{forward_pe:.2f} 倍" if forward_pe else "N/A"
                    st.metric(label="🔮 預期本益比 (Forward P/E)", value=f_pe_str)
                with f_col4:
                    st.metric(label="💰 歷史股息率 (Dividend Yield)", value=f"{dividend_yield:.2f}%")

                st.markdown("---")

                # ==========================================
                # 🧮 數據清洗與特徵矩陣建構
                # ==========================================
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

                # ==========================================
                # 🤖 AI 深度學習大腦訓練與預測
                # ==========================================
                st.subheader("🧠 LSTM 多因子融合神經網路預報")
                
                feature_cols = [
                    'Return', 'Vol_Change', 'MACD_Norm', 
                    'Hist_PE_Norm', 'Hist_PB_Norm', 
                    'Nas_Return', 'SSE_Return', 'HSI_Return'
                ]
                
                data_matrix = df[feature_cols].values
                data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_matrix)

                lookback = 10
                X, y = [], []
                for i in range(len(scaled_data) - lookback):
                    X.append(scaled_data[i:i+lookback])
                    y.append(scaled_data[i+lookback, 0])

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
                    test_preds = model(X_test_t).numpy()
                    latest_10_days = scaled_data[-lookback:].reshape(1, lookback, len(feature_cols))
                    next_return_scaled = model(torch.FloatTensor(latest_10_days)).item()

                dummy_pred = np.zeros((len(test_preds), len(feature_cols)))
                dummy_pred[:, 0] = test_preds.flatten()
                pred_returns = scaler.inverse_transform(dummy_pred)[:, 0]

                dummy_real = np.zeros((len(y_test), len(feature_cols)))
                dummy_real[:, 0] = y_test.flatten()
                real_returns = scaler.inverse_transform(dummy_real)[:, 0]
                
                dummy_next = np.zeros((1, len(feature_cols)))
                dummy_next[0, 0] = next_return_scaled
                next_return_real = scaler.inverse_transform(dummy_next)[0, 0]

                test_dates = df.index[train_size + lookback:]
                base_prices = df['Close'].iloc[train_size + lookback - 1 : -1].values
                
                predictions_real = base_prices * (1 + pred_returns)
                y_test_real = base_prices * (1 + real_returns)
                
                next_price = current_price * (1 + next_return_real)
                change = next_return_real * 100

                st.success(f"🎉 跨國『技術 × 智慧估值』多因子大腦對 {processed_ticker} 訓練完畢！")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label=f"今日真實收盤價 ({processed_ticker})", value=f"{currency_symbol}{current_price:.2f}")
                col2.metric(label="AI 預測下一個交易日收盤價", value=f"{currency_symbol}{next_price:.2f}", delta=f"{change:.2f}%")
                col3.metric(label="神經網路學習損失 (MSE Loss)", value=f"{loss.item():.5f}")

                predictions_real = np.clip(predictions_real, a_min=current_price*0.3, a_max=current_price*3.0)
                y_test_real = np.clip(y_test_real, a_min=current_price*0.3, a_max=current_price*3.0)

                chart_data = pd.DataFrame({
                    '真實價格 (Real)': y_test_real,
                    'AI 多因子預測 (Multi-Factor Predicted)': predictions_real
                }, index=test_dates)

                st.subheader("📊 多因子整合 - 價格趨勢還原對比圖")
                st.line_chart(chart_data)

                st.subheader("🚨 交易員決策建議")
                if next_price > current_price:
                    st.success(f"📈 **多因子理性看漲**：AI 綜合評估該資產之技術面動能與動態錨定估值後，預估下一個交易日穩健上漲 **+{change:.2f}%**。")
                else:
                    st.error(f"📉 **多因子理性看跌**：AI 偵測到當前動態估值溢價或技術面動能衰退，預估下一個交易日回落 **{change:.2f}%**。")

        except Exception as e:
            st.error(f"運行出錯: {str(e)}")
else:
    st.info("💡 請點擊左側面板按鈕，啟動『技術 × 智慧動態估值 × 歷年財報大屏』三位一體量化分析。")
