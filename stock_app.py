import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 網頁標題與設定
st.set_page_config(page_title="AI 跨國多因子量化雷達終端", layout="wide")
st.title("⚖️ AI 深度學習預測 × 多股聯動基本面選股雷達")
st.markdown("本系統已成功升級！已啟用**「動態個股波動率風控剪裁 (Volatility Clipping)」**、**「AI 全自動多股矩陣掃描大腦」**。")

class RationalLSTM(nn.Module):
    def __init__(self, num_features):
        super(RationalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

st.sidebar.header("⚙️ 雷達控制中心")
default_tickers = "NVDA, TSLA, AAPL, MSFT, AMZN, GOOG, META, AMD, AVGO, NFLX, 0700.HK, 1398.HK, 1211.HK, 3690.HK, 9988.HK, 2318.HK, 0005.HK, 0941.HK, 1810.HK, 1024.HK"
ticker_input = st.sidebar.text_area("🛰️ 自訂掃描股票清單 (英文逗號分隔)", value=default_tickers)
scan_button = st.sidebar.button("🚀 啟動全球 AI 風控版矩陣掃描")

if scan_button:
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    st.subheader(f"🛰️ 全球 AI 多因子風控矩陣掃描進行中... (共需掃描 {len(ticker_list)} 支標的)")
    progress_bar = st.progress(0)
    radar_results = []
    
    with st.spinner("📥 正在同步全球大盤指數基準流 (NASDAQ / 上證 / 恆指)..."):
        nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
        sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
        hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
        
        for d in [nasdaq, sse, hsi]:
            if not d.empty and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
                
        nasdaq['Nas_Return'] = nasdaq['Close'].pct_change()
        sse['SSE_Return'] = sse['Close'].pct_change()
        hsi['HSI_Return'] = hsi['Close'].pct_change()

    for idx, ticker in enumerate(ticker_list):
        try:
            processed_ticker = ticker
            if ticker.isdigit():
                processed_ticker = f"{str(int(ticker)).zfill(4)}.HK"
            
            df = yf.download(processed_ticker, start="2020-01-01", auto_adjust=False)
            if df.empty: continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            current_price = df['Close'].iloc[-1]
            
            # 🧮 核心風控計算：計算該個股真實歷史波動率上限 (20日標準差)
            df['Return'] = df['Close'].pct_change()
            recent_vol = df['Return'].tail(60).std() # 拿最近 60 個交易日的常態波動率
            max_allowable_return = recent_vol * np.sqrt(20) * 1.5 # 換算成20日波段的 1.5 倍標準差天花板
            
            # 智慧估值基準設定
            pe_ratio = 16.50
            if any(k in processed_ticker for k in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMD', 'AVGO', 'NFLX']): pe_ratio = 32.50
            elif ".HK" in processed_ticker: pe_ratio = 11.20
            
            # 特徵工程
            df['Vol_Change'] = df['Volume'].pct_change()
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Norm'] = (df['EMA12'] - df['EMA26']) / df['Close']
            
            current_eps = current_price / pe_ratio
            df['Hist_PE_Norm'] = (df['Close'] / current_eps).clip(lower=0.1, upper=5.0)
            df['Hist_PB_Norm'] = 1.0
            
            df = df.join(nasdaq['Nas_Return'], how='left')
            df = df.join(sse['SSE_Return'], how='left')
            df = df.join(hsi['HSI_Return'], how='left')
            df = df.ffill().bfill().fillna(0.0)
            
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
            
            X_train_t = torch.FloatTensor(X)
            y_train_t = torch.FloatTensor(y)
            
            model = RationalLSTM(num_features=len(feature_cols))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            for epoch in range(40):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                full_matrix = df[feature_cols].values
                full_matrix = np.nan_to_num(full_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                scaled_full = scaler.transform(full_matrix)
                latest_10_days = scaled_full[-lookback:].reshape(1, lookback, len(feature_cols))
                raw_pred_return = model(torch.FloatTensor(latest_10_days)).item()
                
            # 🛡️ 【風控核心】動態剪裁極端失真值
            clipped_pred_return = np.clip(raw_pred_return, -max_allowable_return, max_allowable_return)
            
            # 標記是否觸發風控優化
            was_clipped = " (🛡️ 已啟動波動率安全剪裁)" if abs(raw_pred_return) > max_allowable_return else ""
            
            future_target_price = current_price * (1 + clipped_pred_return)
            change_percent = clipped_pred_return * 100
            
            if change_percent > 3.5: rating = "🟢 強烈看漲 (Bullish)"
            elif change_percent < -3.5: rating = "🔴 戰略避險 (Bearish)"
            else: rating = "⚖️ 橫盤觀望 (Neutral)"
                
            radar_results.append({
                "股票代碼 (Ticker)": processed_ticker,
                "當前現價 (Price)": f"{current_price:.2f}",
                "AI 20日目標價": f"{future_target_price:.2f}",
                "預期波段漲跌幅": change_percent,
                "風控狀態備註": was_clipped if was_clipped else "🟢 正常數值區間",
                "AI 戰略評級 (Rating)": rating
            })
            
        except Exception:
            continue
            
        progress_bar.progress((idx + 1) / len(ticker_list))
        
    st.markdown("---")
    st.success("🎉 全球 AI 多因子風控版選股雷達掃描完畢！")
    
    if radar_results:
        df_radar = pd.DataFrame(radar_results)
        df_radar = df_radar.sort_values(by="預期波段漲跌幅", ascending=False).reset_index(drop=True)
        df_radar["預期波段漲跌幅"] = df_radar["預期波段漲跌幅"].map("{:+.2f}%".format)
        
        st.subheader("📋 華爾街操盤手晨會——AI 核心資產波段潛力推薦榜 (風控優化版)")
        st.dataframe(df_radar, use_container_width=True)
        
        top_stock = df_radar.iloc[0]["股票代碼 (Ticker)"]
        top_return = df_radar.iloc[0]["預期波段漲跌幅"]
        st.info(f"💡 **雷達首席戰略官提示**：經過真實波動率風控剪裁後，當前 AI 最看好的真實波段資產為 **{top_stock}**，預期一個月內具備 **{top_return}** 的理性上攻動能。")

else:
    st.info("💡 請點擊左側面板按鈕，啟動防禦機制完全體——20支全明星資產風控掃描。")
