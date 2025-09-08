import websocket
import json
import time
import signal
import sys
import threading
import csv
import os
import requests
from datetime import datetime, timedelta

# =================== CONFIGURATION ===================
SYMBOL = "DOGE_USDT"  # DOGE/USDT futures symbol
# =====================================================

# MEXC Futures WebSocket for real OHLCV kline data
WS_URL = "wss://contract.mexc.com/edge"
INTERVAL = "Min5"

# Auto-generate CSV filename based on symbol (lowercase, remove USDT)
def get_csv_filename(symbol):
    if not symbol:
        return "data.csv"
    # Remove _USDT and convert to lowercase
    clean_symbol = symbol.replace('_USDT', '').replace('_usdt', '').lower()
    return f"data{clean_symbol}.csv"

CSV_FILENAME = get_csv_filename(SYMBOL)

# Global variables for WebSocket data
latest_kline = None
last_candle_time = None
ws = None
heartbeat_timer = None
is_running = True

def download_historical_data(symbol, limit=310):
    """Download historical kline data from MEXC Futures API"""
    print(f"📥 Downloading last {limit} minutes of data for {symbol} (FUTURES)...")
    
    # Calculate timestamps (last 310 minutes)
    import time
    now = int(time.time())
    start = now - (limit * 5 *60)  # 310 minutes ago in seconds
    
    # MEXC futures API endpoint for historical klines
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {
        "interval": INTERVAL,
        "start": start,
        "end": now
    }
    
    print(f"🔍 Futures API URL: {url}")
    print(f"🔍 Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"🔍 Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"🔍 Response keys: {list(response_data.keys())}")
            
            # MEXC futures API returns data in format: {"success": true, "code": 0, "data": {...}}
            if response_data.get('success') and 'data' in response_data:
                data_obj = response_data['data']
                
                # Data format: {"time": [timestamps], "open": [prices], "high": [prices], "low": [prices], "close": [prices], "vol": [volumes]}
                if all(key in data_obj for key in ['time', 'open', 'high', 'low', 'close', 'vol']):
                    times = data_obj['time']
                    opens = data_obj['open']
                    highs = data_obj['high']
                    lows = data_obj['low']
                    closes = data_obj['close']
                    volumes = data_obj['vol']
                    
                    # Convert to standard format [timestamp, open, high, low, close, volume]
                    converted_data = []
                    for i in range(len(times)):
                        converted_candle = [
                            int(times[i]) * 1000,  # Convert to milliseconds
                            str(opens[i]),
                            str(highs[i]),
                            str(lows[i]),
                            str(closes[i]),
                            str(volumes[i])
                        ]
                        converted_data.append(converted_candle)
                    
                    print(f"✅ Downloaded {len(converted_data)} historical futures candles")
                    return converted_data
                else:
                    print(f"❌ Unexpected data format: {data_obj.keys()}")
                    return []
            else:
                print(f"❌ Futures API error: {response_data}")
                return []
        else:
            print(f"❌ Futures historical data request failed: {response.status_code}")
            print(f"❌ Response text: {response.text}")
            return []
            
    except Exception as e:
        print(f"⚠️ Error downloading futures historical data: {e}")
        return []

def save_historical_data(historical_data):
    """Save historical kline data to CSV"""
    if not historical_data:
        return
        
    print(f"💾 Saving {len(historical_data)} historical candles to {CSV_FILENAME}...")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(CSV_FILENAME)
    
    with open(CSV_FILENAME, 'w', newline='') as csvfile:  # 'w' to overwrite existing file
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Timestamp', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Write historical data
        for candle in historical_data:
            timestamp = int(candle[0])  # Already in milliseconds from MEXC
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            writer.writerow([
                timestamp,
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                float(candle[1]),  # Open
                float(candle[2]),  # High
                float(candle[3]),  # Low
                float(candle[4]),  # Close
                float(candle[5])   # Volume
            ])
    
    print(f"✅ Historical data saved to {CSV_FILENAME}")
    
    # Set last_candle_time to the last historical candle to avoid duplicates
    global last_candle_time
    if historical_data:
        last_candle_time = int(historical_data[-1][0])
        print(f"🕒 Last historical candle: {datetime.fromtimestamp(last_candle_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}")

def save_to_csv(timestamp, open_price, high_price, low_price, close_price, volume):
    """Save candle data to CSV file"""
    try:
        # Convert timestamp to readable datetime
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(CSV_FILENAME)
        
        with open(CSV_FILENAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                print(f"📄 Created new CSV file: {CSV_FILENAME}")
            
            # Write candle data
            writer.writerow([
                timestamp,
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
            
    except Exception as e:
        print(f"⚠️ CSV save error: {e}")

def on_message(ws, message):
    global latest_kline, last_candle_time
    try:
        data = json.loads(message)
        
        if data.get('channel') == 'push.kline':
            kline_data = data.get('data', {})
            if kline_data.get('symbol') == SYMBOL:
                # Extract OHLCV data from WebSocket message
                # Format: {"a": amount, "c": close, "h": high, "l": low, "o": open, "q": volume, "t": timestamp}
                timestamp = int(kline_data.get('t', 0)) * 1000  # Convert to milliseconds
                open_price = float(kline_data.get('o', 0))
                high_price = float(kline_data.get('h', 0))
                low_price = float(kline_data.get('l', 0))
                close_price = float(kline_data.get('c', 0))
                volume = float(kline_data.get('q', 0))
                
                kline_array = [timestamp, open_price, high_price, low_price, close_price, volume]
                
                latest_kline = {
                    'source': 'mexc-futures-websocket',
                    'data': kline_array
                }
                
                # Only print and save when we get a NEW candle (different timestamp)
                if timestamp != last_candle_time:
                    print(f"🕒 NEW CANDLE: {ts_to_time(timestamp)} | O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}")
                    
                    # Save to CSV
                    save_to_csv(timestamp, open_price, high_price, low_price, close_price, volume)
                    print(f"💾 Saved to {CSV_FILENAME}")
                    
                    last_candle_time = timestamp
        
    except Exception as e:
        print(f"⚠️ WebSocket message error: {e}")

def on_error(ws, error):
    print(f"🔴 WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    global heartbeat_timer, is_running
    print("🔌 WebSocket connection closed")
    
    # Stop heartbeat timer if connection closes
    if heartbeat_timer:
        heartbeat_timer.cancel()
        heartbeat_timer = None
    
    # Auto-reconnect if still running
    if is_running:
        print("🔄 Auto-reconnecting in 5 seconds...")
        time.sleep(5)
        start_websocket()

def send_heartbeat():
    """Send ping to keep connection alive"""
    global ws, heartbeat_timer
    if ws:
        try:
            ping_msg = {"method": "ping"}
            ws.send(json.dumps(ping_msg))
            print("💓 Heartbeat sent")
            
            # Schedule next heartbeat in 30 seconds
            heartbeat_timer = threading.Timer(30.0, send_heartbeat)
            heartbeat_timer.start()
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")

def on_open(ws):
    print(f"🟢 WebSocket connected - subscribing to {SYMBOL} {INTERVAL} klines...")
    
    # Subscribe to kline data
    subscribe_msg = {
        "method": "sub.kline",
        "param": {
            "symbol": SYMBOL,
            "interval": INTERVAL
        }
    }
    
    ws.send(json.dumps(subscribe_msg))
    print(f"📡 Subscribed to {SYMBOL} {INTERVAL} klines")
    
    # Start heartbeat to keep connection alive
    send_heartbeat()

def start_websocket():
    global ws
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # Run WebSocket in a separate thread
    ws.run_forever()

def get_latest_kline():
    return latest_kline

def signal_handler(sig, frame):
    global heartbeat_timer, ws, is_running
    print("\n🛑 Shutting down gracefully...")
    
    # Stop auto-reconnection
    is_running = False
    
    # Stop heartbeat timer
    if heartbeat_timer:
        heartbeat_timer.cancel()
        
    # Close WebSocket connection
    if ws:
        ws.close()
        
    sys.exit(0)

def ts_to_time(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000).strftime('%H:%M:%S')

signal.signal(signal.SIGINT, signal_handler)

print("🚀 Starting MEXC Futures WebSocket kline monitor...")
print(f"📊 Symbol: {SYMBOL}")
print(f"💾 Saving to: {CSV_FILENAME}")
print()

# Download historical data first
historical_data = download_historical_data(SYMBOL, 310)
if historical_data:
    save_historical_data(historical_data)
    print()
else:
    print("⚠️ No historical data downloaded, starting with live data only")
    print()

print("🎯 Starting real-time data collection...")
print("Press Ctrl+C to stop")

# Start WebSocket in background thread
ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

print("⏳ Connecting to MEXC WebSocket...")

# Keep the main thread alive - WebSocket will handle all events
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    signal_handler(None, None)