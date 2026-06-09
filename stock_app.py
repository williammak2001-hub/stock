import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 網頁標題與設定
st.set_page_config(page_title="AI 跨國量化交易終端機", layout="wide")
st.title("🎛️ AI 深度學習預測 × 全球多維度量化分析終端")
st.markdown("本終端全面打通美港股市場，融合 **LSTM 神經網路漲跌幅預測**與**華爾街多維技術特徵矩陣**，拒絕過度擬合與數據污染。")

# 側邊欄設定
st.sidebar.header("⚙️ 交易員控制面板")
raw_ticker = st.sidebar.text_input("輸入股票代碼", value="0066.HK").upper().strip()
train_button = st.sidebar.button("🚀 執行全方位量化分析與訓練")

st.sidebar.markdown("""
---
**💡 跨國代碼輸入指南：**
* **美股**：直接輸入，如 `TSLA`, `NVDA`, `AAPL`
* **港股**：輸入4位數字加 `.HK`，如 `0066.HK` (港鐵), `0700.HK` (騰訊), `0005.HK` (匯豐)
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
    currency_symbol = "HK$" if ".HK" in raw_ticker else "$"
    
    with st.spinner(f"⏳ 正在拉取全球聯動數據，構建量化特徵矩陣..."):
        try:
            # 1. 安全隔離抓取個股與大盤數據
            df = yf.download(raw_ticker, start="2020-01-01", auto_adjust=False)
            nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
            sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
            hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
            
            if df.empty:
                st.error(f"❌ Yahoo Finance 拒絕返回「{raw_ticker}」的數據，請確認格式。")
            else:
                # 熨平 MultiIndex 欄位
                for d in [df, nasdaq, sse, hsi]:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                current_price = df['Close'].iloc[-1]

                # 2. 【核心擴充】計算多維度量化分析特徵
                # A. 動量特徵 (MACD)
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD_Line'] = df['EMA12'] - df['EMA26']
                df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
                df['MACD_Norm'] = df['MACD_Line'] / df['Close']
                
                # B. 波動率特徵 (布林通道 Bollinger Bands)
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['STD20'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
                df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
                
                # C. 回報率計算
                df['Return'] = df['Close'].pct_change()
                df['Vol_Change'] = df['Volume'].pct_change()
                
                nasdaq['Nas_Return'] = nasdaq['Close'].pct_change()
                sse['SSE_Return'] = sse['Close'].pct_change()
                hsi['HSI_Return'] = hsi['Close'].pct_change()
                
                # 合併大盤特徵
                df = df.join(nasdaq['Nas_Return'], how='left')
                df = df.join(sse['SSE_Return'], how='left')
                df = df.join(hsi['HSI_Return'], how='left')
                df = df.ffill().bfill().fillna(0.0)

                # ==========================================
                # 📊 模組一：技術指標分析面板 (置於預測之上)
                # ==========================================
                st.subheader(f"📊 {raw_ticker} 多維量化分析視覺化看板")
                
                tab1, tab2, tab3 = st.tabs(["📈 價格通道與波動率", "📉 趨勢動量 (MACD)", "🌐 全球資產聯動相關性"])
                
                with tab1:
                    st.markdown("**布林通道 (Bollinger Bands) 軌道動態**：觀測當前真實價格是否觸及超買(上軌)或超賣(下軌)邊界。")
                    bb_chart_data = df[['Close', 'BB_Upper', 'MA20', 'BB_Lower']].iloc[-120:] # 看最近大約半年
                    st.line_chart(bb_chart_data)
                    
                with tab2:
                    st.markdown("**MACD 趨勢多空強度**：當快線(MACD)穿過慢線(Signal)時為潛在趨勢反轉點。")
                    macd_chart_data = df[['MACD_Line', 'Signal_Line']].iloc[-120:]
                    st.line_chart(macd_chart_data)
                    
                with tab3:
                    st.markdown("**全球資本聯動矩陣**：展示本股票與全球三大核心指數的**回報率相關係數 (Correlation)**。常規情況下，港股受恆指與上證影響深，美股受那指影響深。")
                    corr_cols = ['Return', 'Nas_Return', 'SSE_Return', 'HSI_Return']
                    corr_matrix = df[corr_cols].corr()
                    # 重新命名更易讀
                    corr_matrix.columns = [f'{raw_ticker}', '美股那指', '上證指數', '恆生指數']
                    corr_matrix.index = [f'{raw_ticker}', '美股那指', '上證指數', '恆生指數']
                    st.table(corr_matrix.style.format("{:.2f}").background_gradient(cmap='coolwarm', axis=None))

                st.markdown("---")

                # ==========================================
                # 🤖 模組二：AI 深度學習預測大腦
                # ==========================================
                st.subheader("🧠 LSTM 神經網路理性回報預報")
                
                # 特徵矩陣送入神經網路
                feature_cols = ['Return', 'Vol_Change', 'MACD_Norm', 'Nas_Return', 'SSE_Return', 'HSI_Return']
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

                st.success(f"🎉 跨國理性大腦對 {raw_ticker} 訓練完畢！")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label=f"今日真實收盤價 ({raw_ticker})", value=f"{currency_symbol}{current_price:.2f}")
                col2.metric(label="AI 預測下一個交易日收盤價", value=f"{currency_symbol}{next_price:.2f}", delta=f"{change:.2f}%")
                col3.metric(label="神經網路學習損失 (MSE Loss)", value=f"{loss.item():.5f}")

                chart_data = pd.DataFrame({
                    '真實價格 (Real)': y_test_real,
                    'AI 理性預測 (Rational Predicted)': predictions_real
                }, index=test_dates)

                st.line_chart(chart_data)

                # 決策建議
                st.subheader("🚨 交易員決策建議")
                if next_price > current_price:
                    st.success(f"📈 **理性看漲**：AI 參考全球大盤回報率後，預估下一個交易日穩健上漲 **+{change:.2f}%**。")
                else:
                    st.error(f"📉 **理性看跌**：AI 偵測到全球資產動態有下行風險，預估下一個交易日回落 **{change:.2f}%**。")

        except Exception as e:
            st.error(f"運行出錯: {str(e)}")
else:
    st.info("💡 請點擊左側面板按鈕，啟動全方位量化分析。")
