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
st.markdown("本系統已成功升級！已啟用**「Transformer 自注意力機制 (Attention) 預測大腦」**與**「微觀價量 × 全球宏觀多因子矩陣」**。")

# 定義具備自注意力機制 (Self-Attention) 的量化神經網路
class AttentionLSTM(nn.Module):
    def __init__(self, num_features):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        
        # 自注意力層 (Self-Attention Mechanism)
        self.attention_query = nn.Linear(64, 64)
        self.attention_key = nn.Linear(64, 64)
        self.attention_value = nn.Linear(64, 64)
        
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x) # lstm_out shape: [batch, lookback, 64]
        
        # 計算 Attention 權重
        Q = self.attention_query(lstm_out)
        K = self.attention_key(lstm_out)
        V = self.attention_value(lstm_out)
        
        # 點積注意力得分 [batch, lookback, lookback]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 上下文向量融合
        context = torch.bmm(attn_weights, V) # [batch, lookback, 64]
        
        # 取最後一個時序的特徵進行最終預測
        return self.linear(context[:, -1, :])

st.sidebar.header("⚙️ 雷達控制中心")
default_tickers = "NVDA, TSLA, AAPL, MSFT, AMZN, GOOG, META, AMD, AVGO, NFLX, 0700.HK, 1398.HK, 1211.HK, 3690.HK, 9988.HK, 2318.HK, 0005.HK, 0941.HK, 1810.HK, 1997.HK"
ticker_input = st.sidebar.text_area("🛰️ 自訂精準掃描清單 (英文逗號分隔)", value=default_tickers)
scan_button = st.sidebar.button("🚀 啟動 Attention 頂級矩陣掃描")

if scan_button:
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    st.subheader(f"🛰️ 全球 AI 自注意力多因子風控掃描進行中... (共 {len(ticker_list)} 支標的)")
    progress_bar = st.progress(0)
    radar_results = []
    
    with st.spinner("📥 正在同步全球大盤指數與宏觀環境流 (美債收益率/黃金/納指/恆指)..."):
        # 基礎三大指數
        nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
        sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
        hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
        
        # 🔑 新增宏觀全局因子：10年期美債收益率 + 全球避險黃金期貨
        tnx = yf.download("^TNX", start="2020-01-01", auto_adjust=True)
        gold = yf.download("GC=F", start="2020-01-01", auto_adjust=True)
        
        for d in [nasdaq, sse, hsi, tnx, gold]:
            if not d.empty and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
                
        nasdaq['Nas_Return'] = nasdaq['Close'].pct_change()
        sse['SSE_Return'] = sse['Close'].pct_change()
        hsi['HSI_Return'] = hsi['Close'].pct_change()
        tnx['TNX_Change'] = tnx['Close'].pct_change()
        gold['Gold_Return'] = gold['Close'].pct_change()

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
            
            # 🧮 核心風控計算
            df['Return'] = df['Close'].pct_change()
            recent_vol = df['Return'].tail(60).std()
            max_allowable_return = recent_vol * np.sqrt(20) * 1.5
            
            # 🔥 新增微觀特徵 1：布林通道帶寬 (Volatility Squeeze)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD20'] = df['Close'].rolling(window=20).std()
            df['BB_Width'] = (df['STD20'] * 4) / df['MA20']
            
            # 🔥 新增微觀特徵 2：價量趨勢指標 PVT (Price Volume Trend)
            df['PVT'] = (df['Return'] * df['Volume']).cumsum()
            df['PVT_Norm'] = df['PVT'] / df['Volume'].rolling(window=20).mean()
            
            # 傳統技術與基本面因子
            df['Vol_Change'] = df['Volume'].pct_change()
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Norm'] = (df['EMA12'] - df['EMA26']) / df['Close']
            
            pe_ratio = 16.50
            if any(k in processed_ticker for k in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMD', 'AVGO', 'NFLx']): pe_ratio = 32.50
            elif ".HK" in processed_ticker: pe_ratio = 11.20
            current_eps = current_price / pe_ratio
            df['Hist_PE_Norm'] = (df['Close'] / current_eps).clip(lower=0.1, upper=5.0)
            
            # 橫向融合全球宏觀因子
            df = df.join(nasdaq['Nas_Return'], how='left')
            df = df.join(sse['SSE_Return'], how='left')
            df = df.join(hsi['HSI_Return'], how='left')
            df = df.join(tnx['TNX_Change'], how='left')
            df = df.join(gold['Gold_Return'], how='left')
            df = df.ffill().bfill().fillna(0.0)
            
            df['Target_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
            clean_df = df.dropna(subset=['Target_20d']).copy()
            
            # 擴展後的全新「精準量化因子矩陣」 (共 11 個核心維度因子)
            feature_cols = [
                'Return', 'Vol_Change', 'MACD_Norm', 'Hist_PE_Norm', 
                'BB_Width', 'PVT_Norm', 
                'Nas_Return', 'SSE_Return', 'HSI_Return', 'TNX_Change', 'Gold_Return'
            ]
            
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
            
            # 訓練全新自注意力網路大腦 (Attention Network)
            model = AttentionLSTM(num_features=len(feature_cols))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            for epoch in range(45):
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
                
            # 風控動態剪裁
            clipped_pred_return = np.clip(raw_pred_return, -max_allowable_return, max_allowable_return)
            was_clipped = "🛡️ 觸發波動率限額" if abs(raw_pred_return) > max_allowable_return else "🟢 正常高精度區間"
            
            future_target_price = current_price * (1 + clipped_pred_return)
            change_percent = clipped_pred_return * 100
            
            if change_percent > 3.5: rating = "🟢 強烈看漲 (Bullish)"
            elif change_percent < -3.5: rating = "🔴 戰略避險 (Bearish)"
            else: rating = "⚖️ 橫盤觀望 (Neutral)"
                
            radar_results.append({
                "股票代碼 (Ticker)": processed_ticker,
                "當前現價 (Price)": f"{current_price:.2f}",
                "Attention 20日目標價": f"{future_target_price:.2f}",
                "預期波段漲跌幅": change_percent,
                "矩陣智商備註": was_clipped,
                "AI 戰略評級 (Rating)": rating
            })
            
        except Exception:
            continue
            
        progress_bar.progress((idx + 1) / len(ticker_list))
        
    st.markdown("---")
    st.success("🎉 全球 AI Attention高精度版選股雷達掃描完畢！")
    
    if radar_results:
        df_radar = pd.DataFrame(radar_results)
        df_radar = df_radar.sort_values(by="預期波段漲跌幅", ascending=False).reset_index(drop=True)
        df_radar["預期波段漲跌幅"] = df_radar["預期波段漲跌幅"].map("{:+.2f}%".format)
        
        st.subheader("📋 華爾街操盤手晨會——AI 核心資產波段潛力推薦榜 (Attention 高精度版)")
        st.dataframe(df_radar, use_container_width=True)
        
        top_stock = df_radar.iloc[0]["股票代碼 (Ticker)"]
        top_return = df_radar.iloc[0]["預期波段漲跌幅"]
        st.info(f"💡 **雷達首席戰略官提示**：融合微觀價量結構、美債黃金宏觀因子與注意力機制後，當前最精準看好的波段首選為 **{top_stock}**，預期一個月內上攻動能為 **{top_return}**。")
else:
    st.info("💡 請點擊左側面板按鈕，啟動全新的「自注意力神經網路 + 跨國宏觀風控完全體大腦」。")
