import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import time

# 網頁標題與設定
st.set_page_config(page_title="AI 跨國多因子量化雷達終端", layout="wide")
st.title("⚖️ AI 深度學習預測 × 多股聯動基本面選股雷達")
st.markdown("本系統已成功升級！已啟用**「AI 全自動多股矩陣掃描大腦」**，一鍵穿透全球核心資產，篩選最佳波段標的。")

# 定義 LSTM 網絡結構
class RationalLSTM(nn.Module):
    def __init__(self, num_features):
        super(RationalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

# 側邊欄設定
st.sidebar.header("⚙️ 雷達控制中心")

# 預設一組極具代表性的全球核心跨國資產清單
default_tickers = "NVDA, TSLA, AAPL, 0700.HK, 1398.HK, 1211.HK"
ticker_input = st.sidebar.text_area("🛰️ 自訂掃描股票清單 (英文逗號分隔)", value=default_tickers)
scan_button = st.sidebar.button("🚀 啟動全球 AI 多因子聯動掃描")

st.sidebar.markdown("""
---
**💡 掃描名單池推薦：**
* 科技巨頭：`NVDA`, `TSLA`, `AAPL`, `MSFT`
* 港股核心：`0700.HK` (騰訊), `1398.HK` (工行), `1211.HK` (比亞迪), `3690.HK` (美團)
""")

if scan_button:
    # 解析輸入代碼池
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    st.subheader(f"🛰️ 全球 AI 多因子矩陣掃描進行中... (共需掃描 {len(ticker_list)} 支標的)")
    progress_bar = st.progress(0)
    
    # 用於儲存最終所有股票分析結果的容器
    radar_results = []
    
    # 預抓全球大盤基準數據 (優化效率，只抓一次)
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

    # 開始遍歷股票池進行量化運算
    for idx, ticker in enumerate(ticker_list):
        st.write(f"⏳ [`{idx+1}/{len(ticker_list)}`] 正在穿透神經網路訓練資產：**{ticker}** ...")
        
        try:
            # 港股格式防呆校正
            processed_ticker = ticker
            if ticker.isdigit():
                processed_ticker = f"{str(int(ticker)).zfill(4)}.HK"
            
            # 抓取個股
            df = yf.download(processed_ticker, start="2020-01-01", auto_adjust=False)
            if df.empty:
                st.warning(f"⚠️ 無法獲取 {processed_ticker} 數據，跳過該標的。")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            current_price = df['Close'].iloc[-1]
            
            # 智慧估值降級策略 (雷達批量版)
            pe_ratio = 16.50  # 預設均衡型
            if any(k in processed_ticker for k in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMD']):
                pe_ratio = 32.50
            elif ".HK" in processed_ticker:
                pe_ratio = 11.20
            
            # 特徵工程
            df['Return'] = df['Close'].pct_change()
            df['Vol_Change'] = df['Volume'].pct_change()
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Norm'] = (df['EMA12'] - df['EMA26']) / df['Close']
            
            current_eps = current_price / pe_ratio
            df['Hist_PE_Norm'] = (df['Close'] / current_eps).clip(lower=0.1)
            df['Hist_PB_Norm'] = 1.0  # 雷達版簡化次要因子提升批量運算速度
            
            # 結合大盤
            df = df.join(nasdaq['Nas_Return'], how='left')
            df = df.join(sse['SSE_Return'], how='left')
            df = df.join(hsi['HSI_Return'], how='left')
            df = df.ffill().bfill().fillna(0.0)
            
            # 波段目標
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
            
            # 快速批量訓練 (雷達模式設定 40 epochs 兼顧速度與準確度)
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
                
            # 推算最新未來20日預測
            model.eval()
            with torch.no_grad():
                full_matrix = df[feature_cols].values
                full_matrix = np.nan_to_num(full_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                scaled_full = scaler.transform(full_matrix)
                latest_10_days = scaled_full[-lookback:].reshape(1, lookback, len(feature_cols))
                next_20d_return = model(torch.FloatTensor(latest_10_days)).item()
                
            future_target_price = current_price * (1 + next_20d_return)
            change_percent = next_20d_return * 100
            
            # 建立戰略評級
            if change_percent > 3.0:
                rating = "🟢 強烈看漲 (Bullish)"
            elif change_percent < -3.0:
                rating = "🔴 戰略避險 (Bearish)"
            else:
                rating = "⚖️ 橫盤觀望 (Neutral)"
                
            # 將結果打包存入大表
            radar_results.append({
                "股票代碼 (Ticker)": processed_ticker,
                "當前現價 (Price)": f"{current_price:.2f}",
                "AI 20日目標價": f"{future_target_price:.2f}",
                "預期波段漲跌幅": change_percent,
                "AI 戰略評級 (Rating)": rating,
                "模型訓練損失 (MSE)": f"{loss.item():.5f}"
            })
            
        except Exception as e:
            st.error(f"❌ 標的 {ticker} 運算失敗: {str(e)}")
            
        # 更新進度條
        progress_bar.progress((idx + 1) / len(ticker_list))
        
    # ==========================================
    # 🏁 渲染華爾街頂級選股雷達看板
    # ==========================================
    st.markdown("---")
    st.success("🎉 全球 AI 多因子聯動選股雷達掃描完畢！")
    
    if radar_results:
        df_radar = pd.DataFrame(radar_results)
        
        # 依照預期波段漲跌幅由高到低進行「戰略排序」
        df_radar = df_radar.sort_values(by="預期波段漲跌幅", ascending=False).reset_index(drop=True)
        
        # 美化格式
        df_radar["預期波段漲跌幅"] = df_radar["預期波段漲跌幅"].map("{:+.2f}%".format)
        
        st.subheader("📋 華爾街操盤手晨會——AI 核心資產波段潛力推薦榜")
        st.dataframe(df_radar, use_container_width=True)
        
        # 戰略小結提示
        top_stock = df_radar.iloc[0]["股票代碼 (Ticker)"]
        top_return = df_radar.iloc[0]["預期波段漲跌幅"]
        st.info(f"💡 **雷達首席戰略官提示**：本次掃描中，AI 大腦最看好的波段資產為 **{top_stock}**，預期一個月內具備 **{top_return}** 的上攻動能。建議多加關注！")
    else:
        st.error("❌ 掃描池內無有效資產數據。")
        
else:
    st.info("💡 請在左側輸入您想監控的全球股票代碼池（美股、港股皆可），並點擊按鈕啟動全自動 AI 雷達矩陣。")
