import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 網頁標題與設定
st.set_page_config(page_title="AI 量化跨國理性預報台", layout="wide")
st.title("⚖️ 跨國通用平穩化 Returns-based LSTM 預報台")
st.markdown("本系統採用**每日漲跌幅百分比 (Daily Returns)** 進行深度學習，全面支援**美股與港股**，完美消除跨國資產價格不對稱的數學污染。")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定面板")
raw_ticker = st.sidebar.text_input("輸入股票代碼", value="0066.HK").upper().strip()
train_button = st.sidebar.button("🚀 啟動跨國理性訓練")

st.sidebar.markdown("""
---
**💡 跨國代碼輸入指南：**
* **美股**：直接輸入，如 `TSLA`, `NVDA`, `NKE`
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
    
    with st.spinner(f"⏳ 正在拉取全球聯動數據，並為 {raw_ticker} 建立漲跌幅矩陣..."):
        try:
            # 1. 隔離抓取個股數據 (關閉自動復權以獲得真實市價)
            df = yf.download(raw_ticker, start="2020-01-01", auto_adjust=False)
            
            # 2. 安全抓取全球大盤數據 (指數類開啟 auto_adjust 避免格式錯亂)
            nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
            sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
            hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
            
            if df.empty:
                st.error(f"❌ Yahoo Finance 拒絕返回「{raw_ticker}」的數據，請確認該股票當前是否正常交易。")
            else:
                # 熨平所有 MultiIndex 欄位
                for d in [df, nasdaq, sse, hsi]:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                # 保存最後一天的真實市場價格
                current_price = df['Close'].iloc[-1]

                # 3. 全面轉化為每日漲跌幅 (Daily Returns)
                df['Return'] = df['Close'].pct_change()
                df['Vol_Change'] = df['Volume'].pct_change()
                df['MACD_Norm'] = (df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()) / df['Close']
                
                nasdaq['Nas_Return'] = nasdaq['Close'].pct_change()
                sse['SSE_Return'] = sse['Close'].pct_change()
                hsi['HSI_Return'] = hsi['Close'].pct_change()
                
                # 合併特徵並處理跨國時差與休市空值
                df = df.join(nasdaq['Nas_Return'], how='left')
                df = df.join(sse['SSE_Return'], how='left')
                df = df.join(hsi['HSI_Return'], how='left')
                df = df.ffill().bfill().fillna(0.0)

                # 特徵矩陣
                feature_cols = ['Return', 'Vol_Change', 'MACD_Norm', 'Nas_Return', 'SSE_Return', 'HSI_Return']
                data_matrix = df[feature_cols].values
                
                # 強力修復：防止 inf 破壞 StandardScaler
                data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                # 4. 數據標準化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_matrix)

                # 5. 製作時間窗口
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

                # 6. 訓練模型
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

                # 7. 預測
                model.eval()
                with torch.no_grad():
                    test_preds = model(X_test_t).numpy()
                    latest_10_days = scaled_data[-lookback:].reshape(1, lookback, len(feature_cols))
                    next_return_scaled = model(torch.FloatTensor(latest_10_days)).item()

                # 8. 還原絕對股價
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

                # 9. 網頁展示
                st.success(f"🎉 跨國理性大腦對 {raw_ticker} 訓練完畢！")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label=f"今日真實收盤價 ({raw_ticker})", value=f"{currency_symbol}{current_price:.2f}")
                col2.metric(label="AI 預測下一個交易日收盤價", value=f"{currency_symbol}{next_price:.2f}", delta=f"{change:.2f}%")
                col3.metric(label="神經網路學習損失 (MSE Loss)", value=f"{loss.item():.5f}")

                # 互動式圖表
                chart_data = pd.DataFrame({
                    '真實價格 (Real)': y_test_real,
                    'AI 理性預測 (Rational Predicted)': predictions_real
                }, index=test_dates)

                st.subheader("📊 跨國通用版 - 價格趨勢還原對比圖")
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
    st.info("💡 請輸入美股代碼（如 TSLA）或港股代碼（如 0066.HK），並點擊按鈕進行分析。")
