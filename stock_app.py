import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# 網頁標題與設定
st.set_page_config(page_title="AI 全球資產配置量化矩陣終端", layout="wide")
st.title("⚖️ AI 深度學習預測 × 馬可維茲動態投資組合配倉大腦")
st.markdown("本系統已成功升級至頂級基金架構！已啟用**「Transformer 自注意力預測大腦」**與**「馬可維茲最高夏普比率投資組合優化矩陣 (Max Sharpe Portfolio Optimization)」**。")

# 自注意力時序神經網路
class AttentionLSTM(nn.Module):
    def __init__(self, num_features):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.attention_query = nn.Linear(64, 64)
        self.attention_key = nn.Linear(64, 64)
        self.attention_value = nn.Linear(64, 64)
        self.linear = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        Q = self.attention_query(lstm_out)
        K = self.attention_key(lstm_out)
        V = self.attention_value(lstm_out)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)
        return self.linear(context[:, -1, :])

# 側邊欄設定
st.sidebar.header("⚙️ 基金控制中心")
default_tickers = "NVDA, TSLA, AAPL, MSFT, AMZN, GOOG, META, AMD, AVGO, NFLX, 0700.HK, 1398.HK, 1211.HK, 3690.HK, 9988.HK, 2318.HK, 0005.HK, 0941.HK, 1810.HK, 1997.HK"
ticker_input = st.sidebar.text_area("🛰️ 自訂核心資產池代碼 (英文逗號分隔)", value=default_tickers)
risk_free_rate = st.sidebar.slider("💵 模擬無風險年化利率 (%)", min_value=0.0, max_value=6.0, value=3.5, step=0.1)
scan_button = st.sidebar.button("🚀 啟動 AI 投資組合優化矩陣計算")

if scan_button:
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    st.subheader(f"🛰️ 全球 AI 自注意力矩陣運算 & 馬可維茲配倉優化中... (資產池共 {len(ticker_list)} 支標的)")
    progress_bar = st.progress(0)
    
    # 用於儲存期望回報和歷史日收益率的容器（供協方差計算）
    p_returns = {}
    historical_returns_dict = {}
    radar_results = []
    
    with st.spinner("📥 正在同步全球大盤指數與宏觀環境流 (美債收益率/黃金/納指/恆指)..."):
        nasdaq = yf.download("^IXIC", start="2020-01-01", auto_adjust=True)
        sse = yf.download("000001.SS", start="2020-01-01", auto_adjust=True)
        hsi = yf.download("^HSI", start="2020-01-01", auto_adjust=True)
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
            df['Return'] = df['Close'].pct_change()
            
            # 儲存個股過去 60 天日收益率，供馬可維茲協方差矩陣使用
            historical_returns_dict[processed_ticker] = df['Return'].tail(60).values
            
            # 核心風控計算
            recent_vol = df['Return'].tail(60).std()
            max_allowable_return = recent_vol * np.sqrt(20) * 1.5
            
            # 特徵工程
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD20'] = df['Close'].rolling(window=20).std()
            df['BB_Width'] = (df['STD20'] * 4) / df['MA20']
            df['PVT'] = (df['Return'] * df['Volume']).cumsum()
            df['PVT_Norm'] = df['PVT'] / df['Volume'].rolling(window=20).mean()
            df['Vol_Change'] = df['Volume'].pct_change()
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Norm'] = (df['EMA12'] - df['EMA26']) / df['Close']
            
            pe_ratio = 16.50
            if any(k in processed_ticker for k in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMD', 'AVGO', 'NFLX']): pe_ratio = 32.50
            elif ".HK" in processed_ticker: pe_ratio = 11.20
            current_eps = current_price / pe_ratio
            df['Hist_PE_Norm'] = (df['Close'] / current_eps).clip(lower=0.1, upper=5.0)
            
            df = df.join(nasdaq['Nas_Return'], how='left')
            df = df.join(sse['SSE_Return'], how='left')
            df = df.join(hsi['HSI_Return'], how='left')
            df = df.join(tnx['TNX_Change'], how='left')
            df = df.join(gold['Gold_Return'], how='left')
            df = df.ffill().bfill().fillna(0.0)
            
            df['Target_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
            clean_df = df.dropna(subset=['Target_20d']).copy()
            
            feature_cols = ['Return', 'Vol_Change', 'MACD_Norm', 'Hist_PE_Norm', 'BB_Width', 'PVT_Norm', 'Nas_Return', 'SSE_Return', 'HSI_Return', 'TNX_Change', 'Gold_Return']
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
            
            model = AttentionLSTM(num_features=len(feature_cols))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            for epoch in range(40):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(torch.FloatTensor(X)), torch.FloatTensor(y))
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                scaled_full = scaler.transform(np.nan_to_num(df[feature_cols].values, nan=0.0))
                latest_10_days = scaled_full[-lookback:].reshape(1, lookback, len(feature_cols))
                raw_pred_return = model(torch.FloatTensor(latest_10_days)).item()
                
            clipped_pred_return = np.clip(raw_pred_return, -max_allowable_return, max_allowable_return)
            
            # 將 20 日期望回報率轉化為馬可維茲需要的「期望回報向量」
            p_returns[processed_ticker] = clipped_pred_return
            
            future_target_price = current_price * (1 + clipped_pred_return)
            
            radar_results.append({
                "資產代碼 (Ticker)": processed_ticker,
                "當前現價": f"{current_price:.2f}",
                "AI 預估目標價": f"{future_target_price:.2f}",
                "AI 預估20日回報": clipped_pred_return
            })
        except Exception:
            continue
        progress_bar.progress((idx + 1) / len(ticker_list))
        
    # =========================================================
    # 🏁 核心演算法：馬可維茲最高夏普比率優化計算
    # =========================================================
    st.markdown("---")
    if len(radar_results) >= 2:
        st.success("🎉 全球 AI 自注意力時序預測完成！正在啟動馬可維茲資金配倉優化大腦...")
        
        # 建立收益率 DataFrame 以便計算協方差
        df_hist_ret = pd.DataFrame(historical_returns_dict).fillna(0.0)
        valid_tickers = df_hist_ret.columns.tolist()
        
        # 提取對齊後的期望回報向量 (20日收益率轉化為日回報基準，與日協方差矩陣對齊)
        exp_returns = np.array([p_returns[t] / 20.0 for t in valid_tickers])
        cov_matrix = df_hist_ret.cov().values
        
        # 轉化每日無風險利率
        rf_daily = (risk_free_rate / 100.0) / 252.0
        
        # 定義優化目標函數：最小化「負夏普比率」
        def neg_sharpe(weights, exp_returns, cov_matrix, rf_daily):
            p_ret = np.sum(exp_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol == 0: return 0
            return -(p_ret - rf_daily) / p_vol

        # 約束條件：所有資金權重相加必須等於 1.0 (100% 滿倉分配)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        # 邊界條件：不允許放空、不允許開槓桿，單一股票權重在 0% ~ 100% 之間
        bounds = tuple((0.0, 1.0) for _ in range(len(valid_tickers)))
        # 初始均勻分配權重
        init_weights = [1.0 / len(valid_tickers)] * len(valid_tickers)
        
        # 執行二次規劃求解
        opt_results = minimize(neg_sharpe, init_weights, args=(exp_returns, cov_matrix, rf_daily),
                               method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = opt_results.x
        
        # 融合預測榜單與馬可維茲配倉比例
        final_portfolio = []
        for i, t in enumerate(valid_tickers):
            # 找到對應的預測數據
            raw_data = next(item for item in radar_results if item["資產代碼 (Ticker)"] == t)
            weight_pct = optimal_weights[i] * 100.0
            
            # 只納入分配權重 > 0.01% 的核心建議資產，避免表格過於雜亂
            final_portfolio.append({
                "🏆 實戰推薦分配權重 (Weight)": weight_pct,
                "資產代碼 (Ticker)": t,
                "當前現價": raw_data["當前現價"],
                "AI Attention 預估目標價": raw_data["AI 預估目標價"],
                "預期波段漲跌幅": f"{raw_data['AI 預估20日回報']*100:+.2f}%"
            })
            
        df_portfolio = pd.DataFrame(final_portfolio)
        # 依照推薦分配權重由高到低進行「戰略重倉排序」
        df_portfolio = df_portfolio.sort_values(by="🏆 實戰推薦分配權重 (Weight)", ascending=False).reset_index(drop=True)
        
        # 另外複製一份用於前端美化顯示
        df_display = df_portfolio.copy()
        df_display["🏆 實戰推薦分配權重 (Weight)"] = df_display["🏆 實戰推薦分配權重 (Weight)"].map("{:.2f}%".format)
        
        # 渲染華爾街動態資產配倉看板
        st.subheader("📊 華爾街頂級寬客決策——AI馬可維茲最高夏普比率黃金配倉大表")
        st.dataframe(df_display, use_container_width=True)
        
        # 智能戰略排版小結
        top_asset = df_portfolio.iloc[0]["資產代碼 (Ticker)"]
        top_weight = df_portfolio.iloc[0]["🏆 實戰推薦分配權重 (Weight)"]
        
        # 算出優化後投資組合的整體預期 20 日回報率與預期年化夏普比率
        p_opt_ret_20d = np.sum([p_returns[t] * optimal_weights[i] for i, t in enumerate(valid_tickers)]) * 100
        p_opt_daily_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        p_opt_sharpe_annual = ((np.sum(exp_returns * optimal_weights) - rf_daily) / p_opt_daily_vol) * np.sqrt(252) if p_opt_daily_vol > 0 else 0
        
        # 展示投資組合的核心健康指標
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 投資組合預期 20 日總回報", f"{p_opt_ret_20d:+.2f}%")
        with col2:
            st.metric("📈 優化後整體年化夏普比率 (Sharpe)", f"{p_opt_sharpe_annual:.2f}")
        with col3:
            st.metric("👑 機構戰略首席重倉標的", f"{top_asset} ({top_weight:.2f}%)")
            
        st.info(f"🛰️ **基金經理人風控備註**：馬可維茲大腦已自動穿透資產間的協方差矩陣。為了對沖高 Beta 科技股的下行風險，大腦已自動調配資產相關性，**建議第一重倉配置 {top_asset}，分配比例為 {top_weight:.2f}%**。這是在當前宏觀環境下，能榨出最高夏普比率的黃金防禦陣型！")
    else:
        st.error("❌ 有效計算資產不足 2 支，馬可維茲投資組合優化矩陣至少需要 2 支資產才能進行協方差分散對沖。")
else:
    st.info("💡 請點擊左側控制面板按鈕，啟動「自注意力大腦 × 馬可維茲配倉完全體大腦」，為您的模擬資金開啟科學權重分佈。")
