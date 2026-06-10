import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================
# CONFIGURATION
# ============================================
TICKER = "BTC-USD"              # Change to SPY, QQQ, AAPL, etc. if you want stocks
INTERVAL = "5m"
PERIOD = "1d"
SWING_LENGTH = 5
USE_VWAP_FILTER = True

INITIAL_CASH = 10_000.00
TRADE_QTY = 0.001               # For BTC-USD this means 0.001 BTC per trade
ALLOW_SHORTS = True             # Local simulation only; real brokers may not allow this
RUN_ONCE = False                # True = check one signal and stop
POLL_SECONDS = 60               # How often to refresh candles when RUN_ONCE is False

OUTPUT_DIR = Path(__file__).resolve().parent
STATE_FILE = OUTPUT_DIR / "paper_state.json"
TRADES_FILE = OUTPUT_DIR / "paper_trades.csv"


# ============================================
# FETCH DATA
# ============================================
def fetch_data(ticker, period, interval):
    """Download candles from yfinance."""
    print(f"Fetching {ticker} data ({interval} timeframe)...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ============================================
# CALCULATE VWAP
# ============================================
def calculate_vwap(df):
    """Calculate VWAP, reset daily."""
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["TPV"] = df["Typical_Price"] * df["Volume"]

    df["Date"] = df.index.date
    df["Cumulative_TPV"] = df.groupby("Date")["TPV"].cumsum()
    df["Cumulative_Volume"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]

    return df


# ============================================
# DETECT SWING HIGHS AND LOWS
# ============================================
def detect_swings(df, length):
    """Detect swing highs and lows."""
    df["Swing_High"] = False
    df["Swing_Low"] = False

    for i in range(length, len(df) - length):
        if df["High"].iloc[i] == df["High"].iloc[i - length:i + length + 1].max():
            df.loc[df.index[i], "Swing_High"] = True

        if df["Low"].iloc[i] == df["Low"].iloc[i - length:i + length + 1].min():
            df.loc[df.index[i], "Swing_Low"] = True

    return df


# ============================================
# GENERATE SIGNALS
# ============================================
def generate_signals(df, use_vwap_filter):
    """Generate entry and exit signals."""
    df["Long_Entry"] = False
    df["Short_Entry"] = False
    df["Long_Exit"] = False
    df["Short_Exit"] = False
    df["Live_Exit_Long"] = False
    df["Live_Exit_Short"] = False

    in_long = False
    in_short = False

    for i in range(1, len(df)):
        long_signal = df["Swing_Low"].iloc[i]
        short_signal = df["Swing_High"].iloc[i]

        if use_vwap_filter:
            long_signal = long_signal and df["Close"].iloc[i] > df["VWAP"].iloc[i]
            short_signal = short_signal and df["Close"].iloc[i] < df["VWAP"].iloc[i]

        if long_signal:
            df.loc[df.index[i], "Long_Entry"] = True
            in_long = True
            in_short = False

        if short_signal:
            df.loc[df.index[i], "Short_Entry"] = True
            in_short = True
            in_long = False

        if in_long:
            swing_exit = df["Swing_High"].iloc[i]
            vwap_exit = (
                df["Close"].iloc[i] < df["VWAP"].iloc[i]
                and df["Close"].iloc[i - 1] >= df["VWAP"].iloc[i - 1]
            )

            if swing_exit or (use_vwap_filter and vwap_exit):
                df.loc[df.index[i], "Long_Exit"] = True
                in_long = False

            if (
                df["Close"].iloc[i] < df["VWAP"].iloc[i]
                and df["Close"].iloc[i - 1] >= df["VWAP"].iloc[i - 1]
            ):
                df.loc[df.index[i], "Live_Exit_Long"] = True

        if in_short:
            swing_exit = df["Swing_Low"].iloc[i]
            vwap_exit = (
                df["Close"].iloc[i] > df["VWAP"].iloc[i]
                and df["Close"].iloc[i - 1] <= df["VWAP"].iloc[i - 1]
            )

            if swing_exit or (use_vwap_filter and vwap_exit):
                df.loc[df.index[i], "Short_Exit"] = True
                in_short = False

            if (
                df["Close"].iloc[i] > df["VWAP"].iloc[i]
                and df["Close"].iloc[i - 1] <= df["VWAP"].iloc[i - 1]
            ):
                df.loc[df.index[i], "Live_Exit_Short"] = True

    return df


# ============================================
# LOCAL PAPER ACCOUNT
# ============================================
def default_state():
    """Create a fresh fake brokerage account."""
    return {
        "initial_cash": INITIAL_CASH,
        "cash": INITIAL_CASH,
        "position_side": None,
        "position_qty": 0.0,
        "entry_price": None,
        "entry_time": None,
        "last_processed_candle": None,
        "realized_profit": 0.0,
        "completed_trades": 0,
    }


def load_state():
    """Load account state, or create it if this is the first run."""
    if not STATE_FILE.exists():
        return default_state()

    with STATE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    """Save account state so the simulator can resume later."""
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_trade(trade):
    """Append one completed trade to the CSV trade log."""
    trade_df = pd.DataFrame([trade])
    write_header = not TRADES_FILE.exists()
    trade_df.to_csv(TRADES_FILE, mode="a", header=write_header, index=False)


def unrealized_profit(state, current_price):
    """Calculate open-position profit at the current price."""
    side = state["position_side"]
    qty = float(state["position_qty"])
    entry = state["entry_price"]

    if side is None or qty == 0 or entry is None:
        return 0.0

    if side == "long":
        return (current_price - entry) * qty

    if side == "short":
        return (entry - current_price) * qty

    return 0.0


def account_equity(state, current_price):
    """Calculate current fake account equity."""
    side = state["position_side"]
    qty = float(state["position_qty"])

    if side == "long":
        return state["cash"] + (qty * current_price)

    if side == "short":
        return state["cash"] - (qty * current_price)

    return state["cash"]


def open_position(state, side, price, timestamp):
    """Open a simulated long or short position."""
    if state["position_side"] is not None:
        print("Already in a position; no new entry opened.")
        return

    if side == "long":
        cost = TRADE_QTY * price
        if state["cash"] < cost:
            print(f"Not enough fake cash to buy. Need ${cost:,.2f}, have ${state['cash']:,.2f}.")
            return
        state["cash"] -= cost

    elif side == "short":
        if not ALLOW_SHORTS:
            print("Short signal ignored because ALLOW_SHORTS is False.")
            return
        state["cash"] += TRADE_QTY * price

    state["position_side"] = side
    state["position_qty"] = TRADE_QTY
    state["entry_price"] = price
    state["entry_time"] = str(timestamp)

    print(f"OPEN {side.upper()} | {TRADE_QTY} {TICKER} @ ${price:,.2f}")


def close_position(state, price, timestamp, reason):
    """Close the current simulated position and record realized profit."""
    side = state["position_side"]
    qty = float(state["position_qty"])
    entry = state["entry_price"]

    if side is None:
        print("No open position to close.")
        return

    if side == "long":
        profit = (price - entry) * qty
        state["cash"] += qty * price
    else:
        profit = (entry - price) * qty
        state["cash"] -= qty * price

    trade = {
        "side": side,
        "entry_time": state["entry_time"],
        "entry_price": entry,
        "exit_time": str(timestamp),
        "exit_price": price,
        "quantity": qty,
        "profit": profit,
        "reason": reason,
    }

    append_trade(trade)

    state["realized_profit"] += profit
    state["completed_trades"] += 1
    state["position_side"] = None
    state["position_qty"] = 0.0
    state["entry_price"] = None
    state["entry_time"] = None

    print(f"CLOSE {side.upper()} | {qty} {TICKER} @ ${price:,.2f} | Profit: ${profit:,.2f}")


def print_account_summary(state, current_price):
    """Print fake account cash, position, and P/L."""
    open_pl = unrealized_profit(state, current_price)
    equity = account_equity(state, current_price)
    total_profit = equity - state["initial_cash"]

    print("\n" + "=" * 60)
    print("LOCAL PAPER ACCOUNT")
    print("=" * 60)
    print(f"Cash: ${state['cash']:,.2f}")
    print(f"Position: {state['position_side'] or 'flat'}")
    print(f"Completed Trades: {state['completed_trades']}")
    print(f"Realized Profit: ${state['realized_profit']:,.2f}")
    print(f"Unrealized Profit: ${open_pl:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Equity: ${equity:,.2f}")
    print(f"State file: {STATE_FILE}")
    print(f"Trades file: {TRADES_FILE}")
    print("=" * 60)


# ============================================
# STRATEGY EXECUTION
# ============================================
def build_signal_frame():
    """Fetch candles and add indicators/signals."""
    df = fetch_data(TICKER, PERIOD, INTERVAL)
    df = calculate_vwap(df)
    df = detect_swings(df, SWING_LENGTH)
    df = generate_signals(df, USE_VWAP_FILTER)
    return df


def process_signal(state, row):
    """Apply one confirmed signal candle to the local paper account."""
    price = float(row["Close"])
    timestamp = row.name

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Signal candle: {timestamp} | Close: ${price:,.2f}")

    if row["Long_Exit"] and state["position_side"] == "long":
        close_position(state, price, timestamp, "Long Exit")

    elif row["Short_Exit"] and state["position_side"] == "short":
        close_position(state, price, timestamp, "Short Exit")

    elif row["Long_Entry"]:
        if state["position_side"] == "short":
            close_position(state, price, timestamp, "Reversal to Long")
        open_position(state, "long", price, timestamp)

    elif row["Short_Entry"]:
        if state["position_side"] == "long":
            close_position(state, price, timestamp, "Reversal to Short")
        open_position(state, "short", price, timestamp)

    else:
        print("Signal: none")

    state["last_processed_candle"] = str(timestamp)
    print_account_summary(state, price)


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    state = load_state()
    print("Local paper trader started. No broker account or API keys needed.")
    print(f"TICKER: {TICKER} | INTERVAL: {INTERVAL} | TRADE_QTY: {TRADE_QTY}")

    while True:
        try:
            df = build_signal_frame()

            # Swing signals require future candles, so use the newest fully confirmable row.
            signal_row = df.iloc[-SWING_LENGTH - 1]
            signal_key = str(signal_row.name)

            if signal_key != state["last_processed_candle"]:
                process_signal(state, signal_row)
                save_state(state)
            else:
                current_price = float(df.iloc[-1]["Close"])
                print(f"No new confirmed signal candle yet: {signal_key}")
                print_account_summary(state, current_price)

        except Exception as exc:
            print(f"Error: {exc}")

        if RUN_ONCE:
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
