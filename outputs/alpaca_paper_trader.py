import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError:
    TradingClient = None


# ============================================
# CONFIGURATION
# ============================================
DATA_TICKER = "BTC-USD"          # yfinance symbol used for candles
ALPACA_SYMBOL = "BTC/USD"        # Alpaca symbol used for paper orders
INTERVAL = "5m"
PERIOD = "1d"
SWING_LENGTH = 5
USE_VWAP_FILTER = True

TRADE_QTY = 0.001                # BTC/share quantity per order
ALLOW_SHORTS = False             # Keep False for BTC/USD unless your account/product supports shorts
DRY_RUN = True                   # Set False only after you confirm paper-account behavior
RUN_ONCE = False                 # True = check one candle and stop; False = keep looping
POLL_SECONDS = 60                # How often to refresh data
TIME_IN_FORCE = TimeInForce.GTC if TradingClient else "gtc"

CLIENT_ORDER_PREFIX = "vwap-swing-paper"


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
# PAPER TRADING HELPERS
# ============================================
def get_trading_client():
    """Create an Alpaca paper trading client from environment variables."""
    if TradingClient is None:
        raise ImportError("Install alpaca-py first: pip install alpaca-py")

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY with your Alpaca paper keys."
        )

    return TradingClient(api_key, secret_key, paper=True)


def get_position_side(trading_client, symbol):
    """Return 'long', 'short', or None based on the paper account position."""
    normalized_symbol = symbol.replace("/", "").upper()
    positions = trading_client.get_all_positions()
    position = next(
        (
            item
            for item in positions
            if item.symbol.replace("/", "").upper() == normalized_symbol
        ),
        None,
    )

    if position is None:
        return None

    qty = float(position.qty)
    if qty > 0:
        return "long"
    if qty < 0:
        return "short"
    return None


def close_position(trading_client, symbol):
    """Close an existing paper position."""
    if DRY_RUN:
        print(f"[DRY RUN] Would close existing {symbol} position")
        return None

    order = trading_client.close_position(symbol)
    print(f"Submitted close-position order: {order.id}")
    return order


def submit_market_order(trading_client, symbol, side, qty):
    """Submit a market order to the Alpaca paper account."""
    client_order_id = f"{CLIENT_ORDER_PREFIX}-{int(time.time())}"

    if DRY_RUN:
        print(f"[DRY RUN] Would submit {side.value.upper()} {qty} {symbol}")
        return None

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TIME_IN_FORCE,
        client_order_id=client_order_id,
    )

    order = trading_client.submit_order(order_data=order_data)
    print(f"Submitted {side.value.upper()} order: {order.id}")
    return order


def handle_latest_signal(trading_client, latest):
    """Apply the newest closed candle signal to the paper account."""
    current_side = get_position_side(trading_client, ALPACA_SYMBOL)
    price = latest["Close"]
    ts = latest.name.strftime("%Y-%m-%d %H:%M")

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Latest candle: {ts} @ ${price:,.2f}")
    print(f"Current paper position: {current_side or 'flat'}")

    if latest["Long_Exit"] and current_side == "long":
        print("Signal: LONG EXIT")
        close_position(trading_client, ALPACA_SYMBOL)
        return

    if latest["Short_Exit"] and current_side == "short":
        print("Signal: SHORT EXIT")
        close_position(trading_client, ALPACA_SYMBOL)
        return

    if latest["Long_Entry"]:
        print("Signal: LONG ENTRY")

        if current_side == "short":
            close_position(trading_client, ALPACA_SYMBOL)
            current_side = None

        if current_side is None:
            submit_market_order(trading_client, ALPACA_SYMBOL, OrderSide.BUY, TRADE_QTY)
        else:
            print("Already long; no new order submitted.")
        return

    if latest["Short_Entry"]:
        print("Signal: SHORT ENTRY")

        if current_side == "long":
            close_position(trading_client, ALPACA_SYMBOL)
            current_side = None

        if ALLOW_SHORTS and current_side is None:
            submit_market_order(trading_client, ALPACA_SYMBOL, OrderSide.SELL, TRADE_QTY)
        elif not ALLOW_SHORTS:
            print("Short entry ignored because ALLOW_SHORTS is False.")
        else:
            print("Already short; no new order submitted.")
        return

    print("Signal: none")


def build_signal_frame():
    """Fetch candles and add indicators/signals."""
    df = fetch_data(DATA_TICKER, PERIOD, INTERVAL)
    df = calculate_vwap(df)
    df = detect_swings(df, SWING_LENGTH)
    df = generate_signals(df, USE_VWAP_FILTER)
    return df


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    trading_client = get_trading_client()

    account = trading_client.get_account()
    print("Connected to Alpaca paper account.")
    print(f"Account status: {account.status}")
    print(f"Buying power: ${float(account.buying_power):,.2f}")
    print(f"DRY_RUN: {DRY_RUN}")

    last_processed_candle = None

    while True:
        try:
            df = build_signal_frame()

            # Swing signals need future candles for confirmation, so use the latest fully confirmable row.
            signal_row = df.iloc[-SWING_LENGTH - 1]

            if signal_row.name != last_processed_candle:
                handle_latest_signal(trading_client, signal_row)
                last_processed_candle = signal_row.name
            else:
                print(f"No new confirmable candle yet: {signal_row.name}")

        except Exception as exc:
            print(f"Error: {exc}")

        if RUN_ONCE:
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
