
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. Fetch Historical Stock Data
# ==========================================
ticker = "spy"
df = yf.download(ticker, start="2018-01-01", end="2027-11-17", multi_level_index=False)

if df.empty:
    raise ValueError(f"No data returned for ticker {ticker}. Check symbol or network connection.")

# ==========================================
# 2. Enhanced Feature Engineering
# ==========================================
df['gk_vol'] = np.sqrt(
    0.5 * (np.log(df['High'] / df['Low'])) ** 2 -
    (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open'])) ** 2
)
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility'] = df['log_return'].rolling(window=20).std()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df['volume_change'] = np.log((df['Volume'] + 1) / (df['Volume'].shift(1) + 1))

df['hv_short'] = df['log_return'].rolling(window=5).std() * np.sqrt(252)
df['hv_long'] = df['log_return'].rolling(window=60).std() * np.sqrt(252)
df['term_structure_slope'] = df['hv_short'] / df['hv_long']

window_days = 30
df['historical_vol'] = df['log_return'].rolling(window=window_days).std() * np.sqrt(252)
lookback_period = 252
df['iv_low_52w'] = df['historical_vol'].rolling(window=lookback_period).min()
df['iv_high_52w'] = df['historical_vol'].rolling(window=lookback_period).max()

vol_spread = df['iv_high_52w'] - df['iv_low_52w']
df['iv_rank'] = np.where(
    vol_spread > 0,
    ((df['historical_vol'] - df['iv_low_52w']) / vol_spread) * 100.0,
    50.0
)

df.dropna(inplace=True)
features = ['log_return', 'volatility', 'gk_vol', 'rsi', 'volume_change', 'term_structure_slope']

# ==========================================
# 3. Black-Scholes Pricing Engine
# ==========================================
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price for European options."""
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# ==========================================
# 4. Valuation Helper for Spreads
# ==========================================
def evaluate_spread(signal, S, sigma, T, r=0.04):
    """Calculates entry credit/debit for different option structures."""
    if signal in ["BULL_PUT_CREDIT_SPREAD", "BULL_PUT_CREDIT_SPREAD_PANIC"]:
        short_strike = S * 0.95
        long_strike = S * 0.92
        credit = black_scholes_price(S, short_strike, T, r, sigma, "put") - \
                 black_scholes_price(S, long_strike, T, r, sigma, "put")
        return credit, short_strike, long_strike, "credit"

    elif signal == "BEAR_CALL_CREDIT_SPREAD":
        short_strike = S * 1.05
        long_strike = S * 1.08
        credit = black_scholes_price(S, short_strike, T, r, sigma, "call") - \
                 black_scholes_price(S, long_strike, T, r, sigma, "call")
        return credit, short_strike, long_strike, "credit"

    elif signal == "BEAR_PUT_DEBIT_SPREAD":
        long_strike = S * 1.00
        short_strike = S * 0.95
        debit = black_scholes_price(S, long_strike, T, r, sigma, "put") - \
                black_scholes_price(S, short_strike, T, r, sigma, "put")
        return debit, long_strike, short_strike, "debit"

    elif signal == "BULL_CALL_DEBIT_SPREAD":
        long_strike = S * 1.00
        short_strike = S * 1.05
        debit = black_scholes_price(S, long_strike, T, r, sigma, "call") - \
                black_scholes_price(S, short_strike, T, r, sigma, "call")
        return debit, long_strike, short_strike, "debit"

    elif signal == "IRON_CONDOR":
        p_short = S * 0.95
        p_long = S * 0.92
        c_short = S * 1.05
        c_long = S * 1.08
        credit = (black_scholes_price(S, p_short, T, r, sigma, "put") - black_scholes_price(S, p_long, T, r, sigma, "put")) + \
                 (black_scholes_price(S, c_short, T, r, sigma, "call") - black_scholes_price(S, c_long, T, r, sigma, "call"))
        return credit, (p_short, c_short), (p_long, c_long), "credit"

    return 0.0, 0, 0, "hold"

# ==========================================
# 5. Walk-Forward Backtest with Dynamic Exits
# ==========================================
def run_walkforward_backtest(full_data, max_holding_period=10, risk_free_rate=0.04):
    print("\n==========================================================")
    print(" WALK-FORWARD BACKTEST WITH DYNAMIC EXITS & REFINED SPREADS")
    print("==========================================================")

    train_window = 1000
    retrain_frequency = 60
    trades = []

    i = train_window
    while i < len(full_data) - max_holding_period:
        train_slice = full_data.iloc[i - train_window : i].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_slice[features].values)

        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_scaled)

        true_means = scaler.inverse_transform(model.means_)
        sorted_states = np.argsort(true_means[:, 0])
        state_mapping = {old: new for new, old in enumerate(sorted_states)}
        model.means_ = model.means_[sorted_states]
        model.transmat_ = model.transmat_[sorted_states[:, None], sorted_states]

        end_step = min(i + retrain_frequency, len(full_data) - max_holding_period)

        for step in range(i, end_step):
            row = full_data.iloc[step]
            scaled_row = scaler.transform(row[features].values.reshape(1, -1))
            regime = int(state_mapping[model.predict(scaled_row)[0]])

            iv_rank = float(row['iv_rank'])
            rsi = float(row['rsi'])
            S_entry = float(row['Close'])
            sigma_entry = max(float(row['historical_vol']), 0.10)

            signal = "HOLD"
            if regime == 0:
                if iv_rank > 70:
                    signal = "BULL_PUT_CREDIT_SPREAD_PANIC"
                else:
                    signal = "BEAR_CALL_CREDIT_SPREAD"
            elif regime == 1:
                signal = "IRON_CONDOR"
            elif regime == 2:
                if iv_rank < 25 and rsi > 52:
                    signal = "BULL_CALL_DEBIT_SPREAD"
                elif iv_rank >= 25:
                    signal = "BULL_PUT_CREDIT_SPREAD"

            if signal != "HOLD":
                dte_entry = 30 / 365.0
                entry_val, _, _, stype = evaluate_spread(signal, S_entry, sigma_entry, dte_entry, r=risk_free_rate)

                if entry_val <= 0:
                    continue

                trade_pnl = None
                exit_day = step

                for day_offset in range(1, max_holding_period + 1):
                    current_idx = step + day_offset
                    if current_idx >= len(full_data):
                        break

                    curr_row = full_data.iloc[current_idx]
                    S_curr = float(curr_row['Close'])
                    dte_curr = max(30 - day_offset, 1) / 365.0
                    sigma_curr = sigma_entry * (0.85 if stype == "credit" else 0.95)

                    curr_val, _, _, _ = evaluate_spread(signal, S_curr, sigma_curr, dte_curr, r=risk_free_rate)

                    if stype == "credit":
                        pnl = entry_val - curr_val
                        if pnl >= 0.50 * entry_val:
                            trade_pnl = pnl
                            exit_day = current_idx
                            break
                        elif pnl <= -2.0 * entry_val:
                            trade_pnl = pnl
                            exit_day = current_idx
                            break
                    else:
                        pnl = curr_val - entry_val
                        if pnl >= 0.50 * entry_val:
                            trade_pnl = pnl
                            exit_day = current_idx
                            break
                        elif pnl <= -0.50 * entry_val:
                            trade_pnl = pnl
                            exit_day = current_idx
                            break

                if trade_pnl is None:
                    final_idx = min(step + max_holding_period, len(full_data) - 1)
                    final_row = full_data.iloc[final_idx]
                    S_final = float(final_row['Close'])
                    dte_final = max(30 - max_holding_period, 1) / 365.0
                    sigma_final = sigma_entry * (0.85 if stype == "credit" else 0.95)
                    final_val, _, _, _ = evaluate_spread(signal, S_final, sigma_final, dte_final, r=risk_free_rate)

                    trade_pnl = (entry_val - final_val) if stype == "credit" else (final_val - entry_val)
                    exit_day = final_idx

                trades.append({
                    'entry_date': full_data.index[step],
                    'exit_date': full_data.index[exit_day],
                    'signal': signal,
                    'entry_price': S_entry,
                    'trade_pnl': trade_pnl
                })

        i = end_step

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['trade_pnl'] > 0]) / total_trades
        total_pnl = trades_df['trade_pnl'].sum() * 100
        
        # Calculate Annualized Sharpe Ratio
        initial_capital = 10000.0
        strategy_returns_series = pd.Series(0.0, index=full_data.index)
        for _, row in trades_df.iterrows():
            exit_date = row['exit_date']
            trade_dollar_pnl = row['trade_pnl'] * 100
            if exit_date in strategy_returns_series.index:
                strategy_returns_series[exit_date] += trade_dollar_pnl
            else:
                idx = full_data.index.asof(exit_date)
                if idx in strategy_returns_series.index:
                    strategy_returns_series[idx] += trade_dollar_pnl

        strategy_equity = initial_capital + strategy_returns_series.cumsum()
        daily_rets = strategy_equity.pct_change().dropna()
        rf_daily = risk_free_rate / 252
        excess_daily = daily_rets - rf_daily
        annualized_sharpe = np.sqrt(252) * excess_daily.mean() / excess_daily.std() if excess_daily.std() > 0 else 0.0

        print(f"[*] Walk-Forward Total Trades: {total_trades}")
        print(f"[*] Strategy Win Rate: {win_rate:.2%}")
        print(f"[*] Total Dollar PnL (per 1 contract spread): ${total_pnl:,.2f}")
        print(f"[*] Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
        print("\n--- Strategy Breakdown ---")
        print(trades_df.groupby('signal')['trade_pnl'].agg(['count', 'mean', 'sum']))
    return trades_df

wf_results_df = run_walkforward_backtest(df)

# ==========================================
# 6. P&L Performance Visualization Function
# ==========================================
def plot_walkforward_performance(data, trades_df, initial_capital=10000.0):
    if trades_df.empty:
        print("[!] No trades executed to plot.")
        return

    benchmark_equity = (data['Close'] / data['Close'].iloc[0]) * initial_capital
    strategy_returns = pd.Series(0.0, index=data.index)

    for _, row in trades_df.iterrows():
        exit_date = row['exit_date']
        trade_dollar_pnl = row['trade_pnl'] * 100

        if exit_date in strategy_returns.index:
            strategy_returns[exit_date] += trade_dollar_pnl
        else:
            idx = data.index.asof(exit_date)
            if idx in strategy_returns.index:
                strategy_returns[idx] += trade_dollar_pnl

    strategy_equity = initial_capital + strategy_returns.cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, benchmark_equity, label=f'Buy & Hold {ticker.upper()} ($10k Start)', color='gray', alpha=0.8, linewidth=1.5)
    plt.plot(data.index, strategy_equity, label='HMM Walk-Forward Options Strategy ($10k Start)', color='#1f77b4', linewidth=2)

    plt.title(f'Walk-Forward HMM Options Strategy vs. {ticker.upper()} Buy-and-Hold', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_walkforward_performance(df, wf_results_df)

# ==========================================
# 7. Live IV Rank Function & Playbook with Risk Rules
# ==========================================
def get_live_iv_rank(ticker_symbol, current_price):
    try:
        tk = yf.Ticker(ticker_symbol)
        expirations = tk.options
        if not expirations:
            return 40.0
        target_exp = expirations[0]
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        calls['distance'] = abs(calls['strike'] - current_price)
        atm_option = calls.loc[calls['distance'].idxmin()]
        current_iv = float(atm_option['impliedVolatility'])
        if pd.isna(current_iv) or current_iv == 0.0:
            return 40.0
        hist_df = tk.history(period="1y")
        hist_returns = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
        rolling_iv = hist_returns.rolling(window=20).std() * np.sqrt(252)
        iv_low, iv_high = float(rolling_iv.min()), float(rolling_iv.max())
        iv_rank = np.clip(((current_iv - iv_low) / (iv_high - iv_low)) * 100.0, 0.0, 100.0)
        return iv_rank
    except Exception:
        return 40.0

def generate_tomorrow_playbook():
    recent_df = df.iloc[-1000:].copy()
    scaler_live = StandardScaler()
    X_live_scaled = scaler_live.fit_transform(recent_df[features].values)
    model_live = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    model_live.fit(X_live_scaled)

    true_means_live = scaler_live.inverse_transform(model_live.means_)
    sorted_states_live = np.argsort(true_means_live[:, 0])
    state_mapping_live = {old: new for new, old in enumerate(sorted_states_live)}
    model_live.means_ = model_live.means_[sorted_states_live]
    model_live.transmat_ = model_live.transmat_[sorted_states_live[:, None], sorted_states_live]

    latest_scaled = scaler_live.transform(recent_df[features].iloc[-1].values.reshape(1, -1))
    current_state_proba = model_live.predict_proba(latest_scaled)[0]
    tomorrow_state_proba = np.dot(current_state_proba, model_live.transmat_)
    predicted_state = np.argmax(tomorrow_state_proba)
    confidence = tomorrow_state_proba[predicted_state]
    current_price = float(recent_df["Close"].iloc[-1])
    current_rsi = float(recent_df["rsi"].iloc[-1])
    sigma_current = max(float(recent_df["historical_vol"].iloc[-1]), 0.10)

    print("\n==============================================")
    print("      TOMORROW'S LIVE OPTIONS PLAYBOOK        ")
    print("==============================================")
    print(f"Current Stock Price: ${current_price:,.2f} | Current RSI: {current_rsi:.1f}")
    print(f"Target Regime Tomorrow: State {predicted_state} (Confidence: {confidence:.2%})")

    if confidence < 0.60:
        print("\n[!] WARNING: Model confidence below threshold (< 60%). Stand aside.")
        return

    live_iv_rank = get_live_iv_rank(ticker, current_price)

    signal = "HOLD"
    if predicted_state == 0:
        if live_iv_rank > 70:
            signal = "BULL_PUT_CREDIT_SPREAD_PANIC"
        else:
            signal = "BEAR_CALL_CREDIT_SPREAD"
    elif predicted_state == 1:
        signal = "IRON_CONDOR"
    elif predicted_state == 2:
        if live_iv_rank < 25 and current_rsi > 52:
            signal = "BULL_CALL_DEBIT_SPREAD"
        else:
            signal = "BULL_PUT_CREDIT_SPREAD"

    dte_entry = 30 / 365.0
    entry_val, _, _, stype = evaluate_spread(signal, current_price, sigma_current, dte_entry, r=0.04)

    print(f"\n--- Recommended Play & Risk Management Rules ---")
    print(f"Strategy Signal: {signal}")

    if stype == "credit":
        max_profit_dollar = entry_val * 100
        profit_target_dollar = 0.50 * max_profit_dollar
        stop_loss_dollar = 2.0 * max_profit_dollar
        print(f"  - Initial Credit Received: ${entry_val * 100:,.2f} per contract")
        print(f"  - Profit Target (50% Max Profit): Close when gain reaches +${profit_target_dollar:,.2f}")
        print(f"  - Stop Loss (2x Credit Risk): Cut trade if loss reaches -${stop_loss_dollar:,.2f}")
    elif stype == "debit":
        max_risk_dollar = entry_val * 100
        profit_target_dollar = 0.50 * max_risk_dollar
        stop_loss_dollar = 0.50 * max_risk_dollar
        print(f"  - Initial Debit Paid: ${entry_val * 100:,.2f} per contract")
        print(f"  - Profit Target (+50% Gain): Close when gain reaches +${profit_target_dollar:,.2f}")
        print(f"  - Stop Loss (-50% Loss): Cut trade if loss reaches -${stop_loss_dollar:,.2f}")

    put_short_strike = round(current_price * 0.95, 2)
    put_long_strike = round(current_price * 0.92, 2)
    call_short_strike = round(current_price * 1.05, 2)
    call_long_strike = round(current_price * 1.08, 2)
    atm_call_strike = round(current_price * 1.00, 2)

    if signal in ["BULL_PUT_CREDIT_SPREAD", "BULL_PUT_CREDIT_SPREAD_PANIC"]:
        print(f"  - Suggested Strikes (30 DTE): Sell ${put_short_strike} Put / Buy ${put_long_strike} Put")
    elif signal == "BEAR_CALL_CREDIT_SPREAD":
        print(f"  - Suggested Strikes (30 DTE): Sell ${call_short_strike} Call / Buy ${call_long_strike} Call")
    elif signal == "IRON_CONDOR":
        print(f"  - Suggested Strikes (30 DTE): Put Spread [${put_short_strike}/${put_long_strike}] & Call Spread [${call_short_strike}/${call_long_strike}]")
    elif signal == "BULL_CALL_DEBIT_SPREAD":
        print(f"  - Suggested Strikes (30 DTE): Buy ${atm_call_strike} Call / Sell ${call_short_strike} Call")

generate_tomorrow_playbook()
