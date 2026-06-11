import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 網頁標題與設定
st.set_page_config(page_title="AI 全球資產配置與勝率審計終端", layout="wide")
st.title("⚖️ AI 深度學習預測 × 馬可維茲配倉 × 成功率審計大屏")
st.markdown("本系統已全面封頂！已啟用**「Transformer 自注意力預測大腦」**、**「馬可維茲優化矩陣」**與**「動態自選股票成功率透視面板」**。")

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
scan_button = st.sidebar.button("🚀 啟動 AI 投資組合優化與勝率審計")

if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
    st.session_state.df_display = None
    st.session_state.audit_data = {}
    st.session_state.metrics = {}

if scan_button:
    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    st.subheader(f"🛰️ 全球 AI 矩陣運算、馬可維茲優化與成功率審計中... (資產池共 {len(ticker_list)} 支標的)")
    progress_bar = st.progress(0)
    
    p_returns = {}
    historical_returns_dict = {}
    radar_results = []
    audit_data_temp = {}
    
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
            
            # 💡 【核心修復點 1】保留 Series 結構與 Datetime 索引，拒絕使用純 values 陣列
            historical_returns_dict[processed_ticker] = df['Return'].tail(60)
            
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
                p_returns[processed_ticker] = clipped_pred_return
                future_target_price = current_price * (1 + clipped_pred_return)
                
                # 滾動窗口審計引擎
                hist_preds = []
                hist_actuals = []
                hist_dates = []
                
                for j in range(len(scaled_data) - lookback - 60, len(scaled_data) - lookback):
                    if j < 0: continue
                    window = scaled_data[j:j+lookback].reshape(1, lookback, len(feature_cols))
                    pred_ret = model(torch.FloatTensor(window)).item()
                    pred_ret_clipped = np.clip(pred_ret, -max_allowable_return, max_allowable_return)
                    
                    actual_ret = clean_df['Target_20d'].iloc[j+lookback]
                    current_close = clean_df['Close'].iloc[j+lookback]
                    
                    hist_preds.append(current_close * (1 + pred_ret_clipped))
                    hist_actuals.append(current_close * (1 + actual_ret))
                    hist_dates.append(clean_df.index[j+lookback])
                
                preds_arr = np.array(hist_preds)
                actuals_arr = np.array(hist_actuals)
                
                pred_dir = preds_arr > clean_df['Close'].iloc[len(scaled_data)-len(preds_arr):].values
                actual_dir = actuals_arr > clean_df['Close'].iloc[len(scaled_data)-len(actuals_arr):].values
                directional_win_rate = np.mean(pred_dir == actual_dir) * 100.0
                mape = np.mean(np.abs((actuals_arr - preds_arr) / actuals_arr)) * 100.0
                price_accuracy = max(0.0, 100.0 - mape)
                
                audit_data_temp[processed_ticker] = {
                    "dates": hist_dates,
                    "preds": hist_preds,
                    "actuals": hist_actuals,
                    "win_rate": directional_win_rate,
                    "accuracy": price_accuracy
                }
            
            radar_results.append({
                "資產代碼 (Ticker)": processed_ticker,
                "當前現價": f"{current_price:.2f}",
                "AI 預估目標價": f"{future_target_price:.2f}",
                "AI 預估20日回報": clipped_pred_return,
                "方向預測勝率 (Win Rate)": directional_win_rate,
                "目標價精準度 (Accuracy)": price_accuracy
            })
        except Exception:
            continue
        progress_bar.progress((idx + 1) / len(ticker_list))
        
    if len(radar_results) >= 2:
        # 💡 【核心修復點 2】使用 pd.concat(..., axis=1) 安全合併不同交易日長度的 Series，並自動補齊缺漏值
        df_hist_ret = pd.concat(historical_returns_dict, axis=1).fillna(0.0)
        valid_tickers = df_hist_ret.columns.tolist()
        
        exp_returns = np.array([p_returns[t] / 20.0 for t in valid_tickers])
        cov_matrix = df_hist_ret.cov().values
        rf_daily = (risk_free_rate / 100.0) / 252.0
        
        def neg_sharpe(weights, exp_returns, cov_matrix, rf_daily):
            p_ret = np.sum(exp_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol == 0: return 0
            return -(p_ret - rf_daily) / p_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(len(valid_tickers)))
        init_weights = [1.0 / len(valid_tickers)] * len(valid_tickers)
        
        opt_results = minimize(neg_sharpe, init_weights, args=(exp_returns, cov_matrix, rf_daily), method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = opt_results.x
        
        final_portfolio = []
        for i, t in enumerate(valid_tickers):
            raw_data = next(item for item in radar_results if item["資產代碼 (Ticker)"] == t)
            weight_pct = optimal_weights[i] * 100.0
            
            final_portfolio.append({
                "🏆 實戰推薦分配權重 (Weight)": weight_pct,
                "資產代碼 (Ticker)": t,
                "當前現價": raw_data["當前現價"],
                "AI 預估目標價": raw_data["AI 預估目標價"],
                "預期波段漲跌幅": f"{raw_data['AI 預估20日回報']*100:+.2f}%",
                "🎯 歷史方向勝率": f"{raw_data['方向預測勝率 (Win Rate)']:.1f}%",
                "📐 價格預測精準度": f"{raw_data['目標價精準度 (Accuracy)']:.1f}%"
            })
            
        df_portfolio = pd.DataFrame(final_portfolio)
        df_portfolio = df_portfolio.sort_values(by="🏆 實戰推薦分配權重 (Weight)", ascending=False).reset_index(drop=True)
        
        df_display = df_portfolio.copy()
        df_display["🏆 實戰推薦分配權重 (Weight)"] = df_display["🏆 實戰推薦分配權重 (Weight)"].map("{:.2f}%".format)
        
        p_opt_ret_20d = np.sum([p_returns[t] * optimal_weights[i] for i, t in enumerate(valid_tickers)]) * 100
        p_opt_daily_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        p_opt_sharpe_annual = ((np.sum(exp_returns * optimal_weights) - rf_daily) / p_opt_daily_vol) * np.sqrt(252) if p_opt_daily_vol > 0 else 0
        avg_win_rate = np.mean([radar_results[k]["方向預測勝率 (Win Rate)"] for k in range(len(radar_results))])
        avg_price_acc = np.mean([radar_results[k]["目標價精準度 (Accuracy)"] for k in range(len(radar_results))])
        
        st.session_state.df_display = df_display
        st.session_state.audit_data = audit_data_temp
        st.session_state.valid_tickers = valid_tickers
        st.session_state.metrics = {
            "ret_20d": p_opt_ret_20d,
            "sharpe": p_opt_sharpe_annual,
            "avg_win": avg_win_rate,
            "avg_acc": avg_price_acc,
            "top_asset": df_portfolio.iloc[0]["資產代碼 (Ticker)"],
            "top_weight": df_portfolio.iloc[0]["🏆 實戰推薦分配權重 (Weight)"]
        }
        st.session_state.scan_done = True

# 前端動態渲染面板
if st.session_state.scan_done:
    st.subheader("📊 華爾街頂級寬客決策——AI馬可維茲黃金配倉與勝率審計大表")
    st.dataframe(st.session_state.df_display, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔮 ⚙️ 核心資產歷史成功率動態透視面板")
    
    selected_stock = st.selectbox(
        "🔎 請選擇您想要透視審計的股票代碼 (Ticker)：", 
        options=st.session_state.valid_tickers,
        index=0
    )
    
    if selected_stock in st.session_state.audit_data:
        data_plot = st.session_state.audit_data[selected_stock]
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"🎯 {selected_stock} 歷史方向預測勝率", f"{data_plot['win_rate']:.1f}%")
        with c2:
            st.metric(f"📐 {selected_stock} 價格預測絕對精準度", f"{data_plot['accuracy']:.1f}%")
        
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(data_plot["dates"], data_plot["actuals"], label="Realized 20d Later Price (真實走勢)", color="#2ca02c", linewidth=2.5, linestyle='-')
        ax.plot(data_plot["dates"], data_plot["preds"], label="AI Attention Predicted Price (AI預估價格)", color="#ff7f0e", linewidth=2.0, linestyle='--')
        
        ax.set_title(f"🔍 {selected_stock} 歷史預測與未來貼合度精準審計藍圖", fontsize=13, fontweight='bold')
        ax.set_xlabel("歷史交易日期 (Timeline)", fontsize=10)
        ax.set_ylabel("資產價格 (Price)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left", fontsize=10)
        
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("🏛️ 基金整體健康度指標摘要")
    m = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 組合預期 20 日總回報", f"{m['ret_20d']:+.2f}%")
    with col2:
        st.metric("📈 整體年化夏普比率 (Sharpe)", f"{m['sharpe']:.2f}")
    with col3:
        st.metric("🎯 系統大腦平均方向勝率", f"{m['avg_win']:.1f}%")
    with col4:
        st.metric("📐 系統大腦價格精準度", f"{m['avg_acc']:.1f}%")
        
    st.info(f"🛰️ **基金經理人風控備註**：當前您正在透過動態面板審計 {selected_stock} 的成功率。利用上方下拉選單，您可以自由穿透資產池中的任意標的。")
else:
    st.info("💡 請點擊左側控制面板按鈕，啟動完全體大腦。")
