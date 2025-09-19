import pandas as pd
import numpy as np
import time
import requests
import json
import hmac
import hashlib
import threading
import os
import asyncio
import websockets
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =================== CONFIGURATION ===================
from live_config import *
# =====================================================

def create_working_hmac_signature(timestamp, method, path, body):
    """
    Working HMAC signature generation - tested and verified
    This is the exact method that works in the test script
    """
    base_string = f"{timestamp}\n{method}\n{path}\n{body}"
    key = SECRET.encode('utf-8')
    signature = hmac.new(key, base_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

class COMWebSocketClient:
    """
    WebSocket client for real-time COM updates
    Handles position and order status updates
    """
    
    def __init__(self, base_url, api_key, secret_key, strategy_id):
        self.base_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.api_key = api_key
        self.secret_key = secret_key
        self.strategy_id = strategy_id
        self.websocket = None
        self.connected = False
        self.position_updates = {}
        self.order_updates = {}
    
    def create_hmac_signature(self, timestamp, method, path, body):
        """Create HMAC signature for WebSocket authentication"""
        # Use the same HMAC function as the main trader
        return create_working_hmac_signature(timestamp, method, path, body)
        
    def create_websocket_signature(self, timestamp, key_id):
        """Create HMAC signature for WebSocket authentication"""
        data_string = f"{key_id}\n{timestamp}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            data_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def connect_and_authenticate(self):
        """Connect to WebSocket and authenticate"""
        try:
            uri = f"{self.base_url}/api/v1/stream"
            print(f"🔌 Connecting to WebSocket: {uri}")
            
            # Enable ping/pong to maintain connection
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                close_timeout=10   # Wait 10 seconds for close
            )
            
            # Step 1: Authenticate (COM handles timestamp management)
            # Step 1: Authenticate with required fields
            timestamp = int(time.time())
            auth_msg = {
                "type": "AUTH",
                "key_id": self.api_key,
                "ts": timestamp,
                "signature": self.create_hmac_signature(timestamp, "POST", "/ws/auth", "")
            }
            
            await self.websocket.send(json.dumps(auth_msg))
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            
            # Check for AUTH_SUCCESS or AUTH_ACK
            if auth_data.get("type") not in ["AUTH_SUCCESS", "AUTH_ACK"]:
                raise Exception(f"WebSocket authentication failed: {auth_data}")
            
            print("✅ WebSocket authenticated successfully")
            
            # Step 2: Subscribe to strategy
            subscribe_msg = {
                "type": "SUBSCRIBE",
                "strategy_id": self.strategy_id
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            sub_response = await self.websocket.recv()
            sub_data = json.loads(sub_response)
            
            # Check for SUBSCRIBE_SUCCESS or SUBSCRIBED
            if sub_data.get("type") not in ["SUBSCRIBE_SUCCESS", "SUBSCRIBED"]:
                raise Exception(f"WebSocket subscription failed: {sub_data}")
            
            print(f"✅ WebSocket subscribed to strategy: {self.strategy_id}")
            self.connected = True
            
            return True
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            self.connected = False
            return False
    
    async def listen_for_events(self, event_handler):
        """Listen for WebSocket events with periodic ping to maintain connection"""
        try:
            last_ping = time.time()
            
            while self.connected:
                try:
                    # Set a timeout for receiving messages
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    
                    # Print raw message first
                    print(f"📨 RAW WebSocket Message:")
                    print(f"📄 {message}")
                    print(f"─" * 60)
                    
                    # Parse and handle the event
                    event = json.loads(message)
                    await event_handler(event)
                    
                except asyncio.TimeoutError:
                    # Send periodic ping to keep connection alive
                    current_time = time.time()
                    if current_time - last_ping > 25:  # Ping every 25 seconds
                        try:
                            await self.websocket.ping()
                            last_ping = current_time
                            print("🏓 WebSocket ping sent to maintain connection")
                        except Exception as e:
                            print(f"❌ WebSocket ping failed: {e}")
                            break
                        
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
            self.connected = False
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            self.connected = False
    
    async def acknowledge_heartbeat(self, heartbeat_id):
        """Acknowledge a heartbeat to maintain connection"""
        try:
            ack_msg = {
                "type": "HEARTBEAT_ACK",
                "heartbeat_id": heartbeat_id,
                "timestamp": int(time.time())
            }
            await self.websocket.send(json.dumps(ack_msg))
            return True
        except Exception as e:
            print(f"❌ Failed to send heartbeat acknowledgment: {e}")
            return False
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print("🔌 WebSocket connection closed")

class LiveEdge5RAVFTrader:
    """
    LIVE EDGE-5 RAVF Trader - MEAN REVERSION ONLY
    
    ENTRY STRATEGY:
    ===============
    MEAN-REVERSION AVWAP SNAPBACK:
    - Long: Price <= VWAP - 0.6*ATR + CLV <= -0.4 + zVol >= 1.2
    - Short: Price >= VWAP + 0.6*ATR + CLV >= 0.4 + zVol >= 1.2
    
    EXIT STRATEGY (5-Bar Exit Engine):
    ===================================
    
    1. SCALE-OUT APPROACH (60% + 40%):
       - TP1: Exit 60% of position at first target
       - Runner: Keep 40% for extended gains
    
    2. 5-BAR EXIT ENGINE:
       - Bar 1: Entry
       - Bar 2: Velocity check + Early adverse management
       - Bar 3: Stall filter (scratch if < +0.25% + volume stall)
       - Bar 4: Snap-lock for strong extension + velocity trail adjustment
       - Bar 5: Hard time stop (mandatory exit)
    
    3. EXIT CONDITIONS (in priority order):
       a) 5-Bar Time Stop (mandatory)
       b) Stop Loss (regime-specific levels)
       c) TP1 Hit (60% scale-out)
       d) TP2 Hit (40% runner)
       e) Trailing Stop (after TP1)
       f) Catastrophic Stop (≤ -0.90%)
       g) Stall Filter (Bar 3 volume stall)
    
    4. REGIME-SPECIFIC LEVELS:
       - Mean-reversion: TP1 +0.50%, TP2 +0.95%, SL -0.80%
    
    5. FEES: 0.03% per trade (0.06% round trip)
    
    6. COM INTEGRATION:
       - Uses COM's trigger orders for automatic stop loss management
       - Automatic TP1 and TP2 execution via COM
       - Dynamic trailing stop updates via COM
       - Full exit plan managed automatically by COM system
    """
    
    def __init__(self, fee_per_trade=0.0003):
        """
        Initialize live trader
        
        Args:
            fee_per_trade: Fee per trade (0.03% = 0.0003)
        """
        self.fee_per_trade = fee_per_trade
        self.round_trip_fee = fee_per_trade * 2  # 0.06% round trip
        
        # Strategy parameters - match backtester exactly
        self.VWAP_ATR_MULTIPLIER = 0.6
        self.CLV_LONG_THRESHOLD = -0.4
        self.CLV_SHORT_THRESHOLD = 0.4
        self.VOLUME_THRESHOLD = 1.2
        
        # Trading state
        self.position = None
        self.current_equity = 10000  # Starting capital
        self.trades = []
        self.equity_curve = [self.current_equity]
        
        # Daily guardrails tracking
        self.daily_pnl = 0
        self.daily_time_stops = 0
        self.last_signal_time = None
        self.current_date = None
        
        # COM connection
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        
        # Position tracking
        self.active_orders = {}  # Track order_ref -> order details
        self.position_ref = None
        self.trigger_orders = {}  # Track trigger orders for exit management
        
        # Historical data storage
        self.candles = []  # Store historical candle data for indicators
        
        # CSV logging for indicators
        self.indicators_csv_filename = 'live_indicators_log.csv'
        self.indicators_csv_initialized = False
        
        # WebSocket connection for real-time updates
        self.ws_client = COMWebSocketClient(
            base_url=COM_BASE_URL,
            api_key=API_KEY,
            secret_key=SECRET,
            strategy_id=STRATEGY_ID
        )
        self.ws_connected = False
        self.position_status = None  # Track current position status from COM
        
        print(f"🚀 Live Mean Reversion Strategy Started")
        print(f"💰 Round-trip fees: {self.round_trip_fee*100:.2f}%")
        print(f"🔗 COM URL: {COM_BASE_URL}")
        print(f"📊 Symbol: {SYMBOL}")
        print(f"🎯 Using COM's full exit plan management capabilities")
        print(f"🔐 Strategy: {STRATEGY_NAME} ({STRATEGY_ID})")
        print(f"🔑 API Key: {API_KEY[:20]}...")
        print(f"🧂 Salt: {SALT[:20]}...")
        print(f"💾 Indicators will be logged to: {self.indicators_csv_filename}")
    
    async def handle_websocket_event(self, event):
        """Handle WebSocket events from COM with raw event logging"""
        try:
            # Always print raw event first
            print(f"🔌 RAW WebSocket Event Received:")
            print(f"📄 {json.dumps(event, indent=2)}")
            print(f"─" * 80)
            
            event_type = event.get('type', '')
            print(f"🎯 Event Type: {event_type}")
            
            if event_type == 'ORDER_UPDATE':
                print(f"📋 Processing ORDER_UPDATE...")
                await self._handle_order_update(event)
            elif event_type == 'FILL':
                print(f"💰 Processing FILL...")
                await self._handle_fill(event)
            elif event_type == 'POSITION_UPDATE':
                print(f"📊 Processing POSITION_UPDATE...")
                await self._handle_position_update(event)
            elif event_type == 'POSITION_CLOSED':
                print(f"🔚 Processing POSITION_CLOSED...")
                await self._handle_position_closed(event)
            elif event_type == 'STOP_TRIGGERED':
                print(f"🛑 Processing STOP_TRIGGERED...")
                await self._handle_stop_triggered(event)
            elif event_type == 'TAKE_PROFIT_TRIGGERED':
                print(f"📈 Processing TAKE_PROFIT_TRIGGERED...")
                await self._handle_take_profit_triggered(event)
            elif event_type == 'POSITION_CLEANUP':
                print(f"🧹 Processing POSITION_CLEANUP...")
                await self._handle_position_cleanup(event)
            elif event_type == 'HEARTBEAT':
                print(f"💓 Processing HEARTBEAT...")
                await self._handle_heartbeat(event)
            elif event_type == 'AUTH_SUCCESS':
                print(f"🔐 AUTH_SUCCESS received - WebSocket authenticated!")
            elif event_type == 'AUTH_ERROR':
                print(f"❌ AUTH_ERROR received - Authentication failed!")
            elif event_type == 'SUBSCRIBE_SUCCESS':
                print(f"✅ SUBSCRIBE_SUCCESS received - Successfully subscribed to strategy!")
            elif event_type == 'SUBSCRIBE_ERROR':
                print(f"❌ SUBSCRIBE_ERROR received - Subscription failed!")
            else:
                print(f"⚠️ Unknown event type: {event_type}")
                print(f"🔍 Full event keys: {list(event.keys())}")
                
        except Exception as e:
            print(f"❌ Error handling WebSocket event: {e}")
            print(f"📄 Raw event that caused error: {json.dumps(event, indent=2)}")
            import traceback
            traceback.print_exc()
    
    async def _handle_order_update(self, event):
        """Handle order update events"""
        order_ref = event.get('order_ref')
        state = event.get('state')
        
        if order_ref in self.active_orders:
            self.active_orders[order_ref]['status'] = state
            print(f"📊 Order update: {order_ref} -> {state}")
            
            # If order is filled, update position status
            if state == 'FILLED' and self.position:
                print(f"✅ Entry order filled: {order_ref}")
                # Position is now open, COM will manage exit plan
    
    async def _handle_fill(self, event):
        """Handle fill events"""
        order_ref = event.get('order_ref')
        price = event.get('price')
        quantity = event.get('quantity')
        
        print(f"💰 Fill: {order_ref} - {quantity} @ {price}")
        
        # Update position tracking if this is our position
        if self.position and order_ref in self.position.get('order_refs', []):
            print(f"📈 Position fill confirmed: {quantity} @ {price}")
    
    async def _handle_position_update(self, event):
        """Handle position update events"""
        position_ref = event.get('position_ref')
        data = event.get('data', {})
        
        if position_ref == self.position_ref:
            self.position_status = data
            print(f"📊 Position update: {data}")
            
            # Check if position is closed
            if data.get('size', 0) == 0:
                print("🔚 Position closed by COM")
                self.position = None
                self.position_ref = None
                self.position_status = None
    
    async def _handle_stop_triggered(self, event):
        """Handle stop loss triggered events"""
        order_ref = event.get('order_ref')
        details = event.get('details', {})
        
        print(f"🛑 Stop triggered: {order_ref}")
        print(f"📊 Details: {details}")
        
        # Update position if this affects our position
        if self.position and order_ref in self.position.get('order_refs', []):
            print("🛑 Our position stop loss triggered")
    
    async def _handle_take_profit_triggered(self, event):
        """Handle take profit triggered events"""
        order_ref = event.get('order_ref')
        details = event.get('details', {})
        
        print(f"💰 Take profit triggered: {order_ref}")
        print(f"📊 Details: {details}")
        
        # Update position if this affects our position
        if self.position and order_ref in self.position.get('order_refs', []):
            print("💰 Our position take profit triggered")
            
            # Check if this is TP1 (60% scale-out)
            if 'TP1' in str(details):
                if self.position:
                    self.position['tp1_hit'] = True
                    print("✅ TP1 hit - 60% scale-out executed by COM")
    
    async def _handle_position_closed(self, event):
        """Handle position closed events with final results"""
        position_ref = event.get('position_ref')
        details = event.get('details', {})
        
        print(f"🔚 POSITION CLOSED: {position_ref}")
        print(f"📊 Final Results:")
        print(f"   Symbol: {details.get('symbol', 'N/A')}")
        print(f"   Side: {details.get('side', 'N/A')}")
        print(f"   Size: {details.get('size', 'N/A')}")
        print(f"   Entry Price: {details.get('entry_price', 'N/A')}")
        print(f"   Exit Price: {details.get('exit_price', 'N/A')}")
        print(f"   Realized PnL: {details.get('realized_pnl', 'N/A')}")
        print(f"   Total Fees: {details.get('total_fees', 'N/A')}")
        print(f"   Volume: {details.get('volume', 'N/A')}")
        print(f"   Leverage: {details.get('leverage', 'N/A')}")
        print(f"   Duration: {details.get('duration_seconds', 'N/A')} seconds")
        print(f"   Max Favorable: {details.get('max_favorable_pct', 'N/A')}%")
        print(f"   Max Adverse: {details.get('max_adverse_pct', 'N/A')}%")
        print(f"   Close Reason: {details.get('close_reason', 'N/A')}")
        print(f"   Open Time: {details.get('open_time', 'N/A')}")
        print(f"   Close Time: {details.get('close_time', 'N/A')}")
        
        # Update our position tracking
        if position_ref == self.position_ref:
            print("✅ Our position closed - updating local tracking")
            self.position = None
            self.position_ref = None
            self.position_status = None
            
            # Update daily tracking if this was our position
            realized_pnl = details.get('realized_pnl', 0)
            close_reason = details.get('close_reason', 'UNKNOWN')
            
            if realized_pnl:
                self.daily_pnl += realized_pnl
                print(f"📈 Daily PnL updated: {self.daily_pnl:.4f}")
            
            if close_reason == 'POSITION_CLOSED':
                print("📊 Position closed by COM exit plan")

        
    async def _handle_position_cleanup(self, event):
        """Handle position cleanup events"""
        position_id = event.get('position_id')
        data = event.get('data', {})
        
        print(f"🧹 Position cleanup: {position_id}")
        print(f"📊 Cancelled orders: {data.get('cancelled_orders', [])}")
        
        if position_id == self.position_ref:
            print("🔚 Our position cleaned up by COM")
            self.position = None
            self.position_ref = None
            self.position_status = None
    
    async def _handle_heartbeat(self, event):
        """Handle heartbeat events and send acknowledgment"""
        heartbeat_id = event.get('heartbeat_id')
        timestamp = event.get('timestamp')
        
        print(f"💓 Heartbeat received: {heartbeat_id} at {timestamp}")
        
        # Send acknowledgment back to COM
        try:
            success = await self.com_client.acknowledge_heartbeat(heartbeat_id)
            if success:
                print(f"✅ Heartbeat acknowledged: {heartbeat_id}")
            else:
                print(f"❌ Failed to acknowledge heartbeat: {heartbeat_id}")
        except Exception as e:
            print(f"❌ Heartbeat acknowledgment error: {e}")
    
    async def start_websocket_monitoring(self):
        """Start WebSocket monitoring in background"""
        try:
            # Connect to WebSocket
            success = await self.ws_client.connect_and_authenticate()
            if not success:
                print("❌ WebSocket connection failed")
                return False
            
            self.ws_connected = True
            print("✅ WebSocket monitoring started")
            
            # Start listening for events
            await self.ws_client.listen_for_events(self.handle_websocket_event)
                
        except Exception as e:
            print(f"❌ WebSocket monitoring error: {e}")
            self.ws_connected = False
            return False
    
    def create_hmac_signature(self, timestamp, method, path, body):
        """Create HMAC signature for COM authentication"""
        # Use the standalone working HMAC function
        return create_working_hmac_signature(timestamp, method, path, body)
    
    def create_entry_order_with_exit_plan(self, side, entry_price, tp1_price, tp2_price, stop_loss_price):
        """
        Create entry order with complete exit plan managed by COM using percentage-based risk sizing
        
        This leverages COM's ability to manage the full exit strategy automatically:
        - Entry order with 25% of balance and 25x leverage
        - TP1 trigger order (60% scale-out) - POST-ONLY LIMIT
        - TP2 trigger order (40% runner) - POST-ONLY LIMIT  
        - Stop loss trigger order - MARKET
        - Dynamic trailing stop management
        """
        timestamp = int(time.time())
        idempotency_key = f"entry_plan_{timestamp}_{SYMBOL}_{side}_25pct_25x"
        
        # Create comprehensive order with exit plan using correct COM schema
        payload = {
            "idempotency_key": idempotency_key,
            "environment": {"sandbox": PAPER_TRADING},  # Use config setting
            "source": {
                "strategy_id": STRATEGY_ID,
                "instance_id": INSTANCE_ID,
                "owner": OWNER
            },
            "order": {
                "instrument": {
                    "class": "crypto_perp",
                    "symbol": SYMBOL
                },
                "side": side,
                "order_type": "MARKET",
                "time_in_force": "GTC",
                "flags": {
                    "post_only": False,
                    "reduce_only": False,
                    "hidden": False,
                    "iceberg": {},
                    "allow_partial_fills": True
                },
                "routing": {"mode": "AUTO"},
                "leverage": {
                    "enabled": True,
                    "leverage": 25.0
                },
                "risk": {
                    "sizing": {
                        "mode": "PCT_BALANCE",
                        "value": 25.0,
                        "floor": {
                            "notional": 10.0
                        }
                    }
                },
                "exit_plan": {
                    "legs": [
                        {
                            "kind": "TP",
                            "label": "TP1 (60% Scale-out)",
                            "allocation": {
                                "type": "percentage",
                                "value": 60.0
                            },
                            "trigger": {
                                "mode": "PRICE",
                                "price_type": "MARK",
                                "value": tp1_price
                            },
                            "exec": {
                                "order_type": "LIMIT",
                                "price": tp1_price,
                                "time_in_force": "GTC",
                                "flags": {
                                    "post_only": True,
                                    "reduce_only": True,
                                    "hidden": False,
                                    "iceberg": {},
                                    "allow_partial_fills": True
                                }
                            },
                            "after_fill_actions": [
                                {
                                    "action": "SET_SL_TO_BREAKEVEN"
                                }
                            ]
                        },
                        {
                            "kind": "TP",
                            "label": "TP2 (40% Runner)",
                            "allocation": {
                                "type": "percentage",
                                "value": 40.0
                            },
                            "trigger": {
                                "mode": "PRICE",
                                "price_type": "MARK",
                                "value": tp2_price
                            },
                            "exec": {
                                "order_type": "LIMIT",
                                "price": tp2_price,
                                "time_in_force": "GTC",
                                "flags": {
                                    "post_only": True,
                                    "reduce_only": True,
                                    "hidden": False,
                                    "iceberg": {},
                                    "allow_partial_fills": True
                                }
                            }
                        },
                        {
                            "kind": "SL",
                            "label": "Stop Loss",
                            "allocation": {
                                "type": "percentage",
                                "value": 100.0
                            },
                            "trigger": {
                                "mode": "PRICE",
                                "price_type": "MARK",
                                "value": stop_loss_price
                            },
                            "exec": {
                                "order_type": "MARKET",
                                "time_in_force": "GTC"
                            }
                        }
                    ],
                    "timestop": {
                        "enabled": True,
                        "duration_minutes": 5.0,
                        "action": "MARKET_EXIT"
                    }
                }
            }
        }
        
        body = json.dumps(payload)
        signature = self.create_hmac_signature(timestamp, "POST", "/api/v1/orders/orders", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}',
            "X-Idempotency-Key": idempotency_key
        }
        
        try:
            response = self.session.post(
                f"{COM_BASE_URL}/api/v1/orders/orders",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                order_ref = result.get('order_ref')
                position_ref = result.get('position_ref')
                
                # Track the complete order plan
                self.active_orders[order_ref] = {
                                'side': side,
                    'risk_sizing': '25% of balance @ 25x leverage',
                    'order_type': 'LIMIT',
                    'price': entry_price,
                    'status': 'PENDING',
                    'exit_plan': payload['order']['exit_plan']
                }
                
                # Store position reference for future updates
                if position_ref:
                    self.position_ref = position_ref
                
                print(f"✅ Entry order with exit plan created: {side} 25% of balance @ 25x leverage {SYMBOL} - MARKET ORDER")
                print(f"📊 TP1: {tp1_price} (60% scale-out) - POST-ONLY LIMIT")
                print(f"📊 TP2: {tp2_price} (40% runner) - POST-ONLY LIMIT")
                print(f"🛑 Stop Loss: {stop_loss_price} - MARKET")
                print(f"⏰ TimeStop: 5-minute auto-exit - MARKET_EXIT")
                print(f"📈 After TP1: Stop loss moves to breakeven automatically")
                
                return order_ref
            else:
                print(f"❌ Entry order creation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Entry order creation error: {e}")
            return None
    
    def create_trigger_order(self, trigger_type, trigger_price, order_side, quantity, order_type, price=None, reduce_only=False):
        """
        Create trigger order via COM for advanced exit management
        
        Args:
            trigger_type: 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP'
            trigger_price: Price that triggers the order
            order_side: 'BUY' or 'SELL' for the triggered order
            quantity: Quantity to execute
            order_type: 'MARKET' or 'LIMIT'
            price: Limit price (if order_type is LIMIT)
            reduce_only: Whether this reduces position
        """
        timestamp = int(time.time())
        idempotency_key = f"trigger_{trigger_type}_{timestamp}_{SYMBOL}_{order_side}_{quantity}"
        
        payload = {
            "idempotency_key": idempotency_key,
            "environment": {"sandbox": PAPER_TRADING},
            "source": {
                "strategy_id": STRATEGY_ID,
                "instance_id": INSTANCE_ID,
                "owner": OWNER
            },
            "trigger_order": {
                "instrument": {
                    "class": "crypto_perp",
                    "symbol": SYMBOL
                },
                "trigger_type": trigger_type,
                "trigger_price": trigger_price,
                "order_side": order_side,
                "quantity": {
                    "type": "contracts",
                    "value": quantity
                },
                "order_type": order_type,
                "price": price,
                "time_in_force": "GTC",
                "flags": {
                    "post_only": order_type == "LIMIT",
                    "reduce_only": reduce_only,
                    "hidden": False,
                    "iceberg": {},
                    "allow_partial_fills": True
                }
            }
        }
        
        body = json.dumps(payload)
        signature = self.create_hmac_signature(timestamp, "POST", "/api/v1/trigger-orders", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}',
            "X-Idempotency-Key": idempotency_key
        }
        
        try:
            response = self.session.post(
                f"{COM_BASE_URL}/api/v1/trigger-orders",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                trigger_ref = result.get('trigger_order_ref')
                
                self.trigger_orders[trigger_ref] = {
                    'trigger_type': trigger_type,
                    'trigger_price': trigger_price,
                    'order_side': order_side,
                            'quantity': quantity,
                    'order_type': order_type,
                    'price': price,
                    'status': 'ACTIVE'
                }
                
                print(f"✅ Trigger order created: {trigger_type} at {trigger_price}")
                return trigger_ref
            else:
                print(f"❌ Trigger order creation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Trigger order creation error: {e}")
            return False
                
    def update_trailing_stop(self, position_ref, new_trail_distance):
        """
        Update trailing stop distance via COM
        
        This allows dynamic adjustment of trailing stops based on:
        - Bar 4 snap-lock (raise trail by +0.05%)
        - Velocity-based trail tightening (to 0.15%)
        - Dynamic trail adjustments during position management
        """
        timestamp = int(time.time())
        
        payload = {
            "trail_distance": new_trail_distance,
            "update_type": "TRAILING_STOP_ADJUSTMENT"
        }
        
        body = json.dumps(payload)
        signature = self.create_hmac_signature(timestamp, "PUT", f"/api/v1/positions/{position_ref}/trailing-stop", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}',
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.put(
                f"{COM_BASE_URL}/api/v1/positions/{position_ref}/trailing-stop",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✅ Trailing stop updated: {new_trail_distance*100:.3f}%")
                return True
            else:
                print(f"❌ Trailing stop update failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Trailing stop update error: {e}")
            return False
                
    def cancel_order(self, order_ref):
        """Cancel order via COM API"""
        timestamp = int(time.time())
        body = ""
        signature = self.create_hmac_signature(timestamp, "DELETE", f"/api/v1/orders/orders/{order_ref}/cancel", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}'
        }
        
        try:
            response = self.session.delete(
                f"{COM_BASE_URL}/api/v1/orders/orders/{order_ref}/cancel",
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✅ Order cancelled: {order_ref}")
                if order_ref in self.active_orders:
                    del self.active_orders[order_ref]
                return True
            else:
                print(f"❌ Order cancellation failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Order cancellation error: {e}")
            return False
            
    def cancel_trigger_order(self, trigger_ref):
        """Cancel trigger order via COM API"""
        timestamp = int(time.time())
        body = ""
        signature = self.create_hmac_signature(timestamp, "DELETE", f"/api/v1/trigger-orders/{trigger_ref}/cancel", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}'
        }
        
        try:
            response = self.session.delete(
                f"{COM_BASE_URL}/api/v1/trigger-orders/{trigger_ref}/cancel",
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✅ Trigger order cancelled: {trigger_ref}")
                if trigger_ref in self.trigger_orders:
                    del self.trigger_orders[trigger_ref]
                return True
            else:
                print(f"❌ Trigger order cancellation failed: {response.status_code}")
                return False
            
        except Exception as e:
            print(f"❌ Trigger order cancellation error: {e}")
            return False
            
    def get_positions(self):
        """Get current positions via COM API"""
        timestamp = int(time.time())
        body = ""
        signature = self.create_hmac_signature(timestamp, "GET", "/api/v1/positions", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}'
        }
        
        try:
            response = self.session.get(
                f"{COM_BASE_URL}/api/v1/positions",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Position fetch failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Position fetch error: {e}")
            return None
                
    def close_position(self, position_ref, quantity=None):
        """Close position via COM API - COM handles all execution"""
        print(f"🔄 Requesting COM to close position: {position_ref}")
        print(f"📊 Quantity: {quantity if quantity else 'ALL'}")
        print(f"💡 COM will handle all execution and order management")
        
        # COM handles all the actual closing logic
        # This method just logs the intent
        return True
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=window).mean()
        return df['atr']
    
    def _calculate_clv(self, df):
        """Calculate Close Location Value"""
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['clv'] = df['clv'].clip(-1, 1)
        return df['clv']
    
    def _calculate_vwap(self, df, window=48):
        """Calculate rolling VWAP to avoid look-ahead bias - matches backtester exactly"""
        volume_price = df['close'] * df['volume']
        rolling_vwap = volume_price.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        df['vwap'] = rolling_vwap
        return df['vwap']
    
    def _calculate_relative_volume(self, df, lookback=48):
        """Calculate relative volume (z-score) - matches backtester exactly"""
        df['zvol'] = df['volume'] / df['volume'].rolling(window=lookback).mean()
        return df['zvol']
    
    
    def _calculate_returns(self, df):
        """Calculate log returns - matches backtester"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        return df['returns']
    
    def _calculate_realized_volatility(self, df, window=48):
        """Calculate rolling realized volatility (std of returns) - matches backtester"""
        df['rv'] = df['returns'].rolling(window=window).std()
        return df['rv']
    
    def _calculate_skew_kurt(self, df, window=48):
        """Calculate rolling skewness and kurtosis - matches backtester"""
        df['skew'] = df['returns'].rolling(window=window).skew()
        df['kurt'] = df['returns'].rolling(window=window).kurt()
        return df[['skew', 'kurt']]
    
    def _get_safe_timestamp(self, current):
        """Helper method to safely get timestamp from current data"""
        if isinstance(current['timestamp'], str):
            return pd.to_datetime(current['timestamp'])
        else:
            return current['timestamp']
    
    def _get_safe_numeric(self, current, field):
        """Helper method to safely get numeric value from current data"""
        value = current[field]
        if isinstance(value, str):
            return float(value)
        else:
            return value
    
    def _print_debug_info(self, current):
        """Print detailed debug information about indicators and signal proximity"""
        try:
            close_price = self._get_safe_numeric(current, 'close')
            vwap = self._get_safe_numeric(current, 'vwap')
            volume = self._get_safe_numeric(current, 'volume')
            atr = self._get_safe_numeric(current, 'atr')
            clv = self._get_safe_numeric(current, 'clv')
            z_volume = self._get_safe_numeric(current, 'zvol')
            
            z_volume_display = f"{z_volume:.2f}" if not pd.isna(z_volume) else "NaN"
            print(f"🔍 Debug Info - Stored values: CLV={clv:.4f}, zVol={z_volume_display}, ATR={atr:.6f}, VWAP={vwap:.6f}")
            print(f"🔍 Debug Info - Current candle count: {len(self.candles)}")
            
            # Use entropy from stored candle data (calculated in process_new_candle)
            entropy = self._get_safe_numeric(current, 'entropy')
            print(f"🔍 Debug Info - Using stored entropy: {entropy:.4f}")
            
            # Also calculate entropy here for comparison
            entropy_calc = 0.0
            if len(self.candles) >= 20:  # Need some history for entropy
                try:
                    # Create DataFrame from historical candles for rolling entropy calculation
                    hist_df = pd.DataFrame(self.candles)
                    hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')
                    hist_df['returns'] = np.log(hist_df['close'] / hist_df['close'].shift(1))
                    
                    # Calculate rolling entropy using backtester method
                    def entropy_calc_func(returns):
                        if len(returns) < 2:
                            return 0.0
                        returns = returns.dropna()
                        if len(returns) < 2:
                            return 0.0
                        try:
                            # Calculate entropy as -sum(p * log(p)) where p is normalized returns
                            abs_returns = np.abs(returns)
                            if abs_returns.sum() == 0:
                                return 0.0
                            p = abs_returns / abs_returns.sum()
                            p = p[p > 0]  # Remove zeros
                            return -np.sum(p * np.log(p + 1e-10))
                        except:
                            return 0.0
                    
                    hist_df['entropy'] = hist_df['returns'].rolling(window=20).apply(entropy_calc_func, raw=False)
                    entropy_calc = hist_df['entropy'].iloc[-1] if len(hist_df) > 0 and not pd.isna(hist_df['entropy'].iloc[-1]) else 0.0
                    print(f"🔍 Debug Info - Calculated entropy: {entropy_calc:.4f}")
                except Exception as e:
                    print(f"🔍 Entropy calculation error: {e}")
                    entropy_calc = 0.0
            else:
                entropy_calc = 0.0  # Not enough data for entropy calculation
                print(f"🔍 Debug Info - Not enough data for entropy calculation: {len(self.candles)} candles")
            
            print(f"\n{'='*60}")
            print(f"🔍 DETAILED INDICATOR ANALYSIS")
            print(f"{'='*60}")
            
            # Basic price info
            print(f"💰 Price: ${close_price:.6f} | VWAP: ${vwap:.6f} | Volume: {volume:,.0f}")
            
            # Indicator values
            print(f"\n📊 INDICATOR VALUES:")
            print(f"   ATR: {atr:.6f} ({atr*100:.4f}%)")
            print(f"   CLV: {clv:.4f}")
            print(f"   z-Volume: {z_volume_display}")
            print(f"   Entropy: {entropy:.4f}")
            
            # VWAP bands
            vwap_upper = vwap + (self.VWAP_ATR_MULTIPLIER * atr)
            vwap_lower = vwap - (self.VWAP_ATR_MULTIPLIER * atr)
            
            print(f"\n🎯 VWAP BANDS:")
            print(f"   VWAP: ${vwap:.6f}")
            print(f"   Upper: ${vwap_upper:.6f} (+{self.VWAP_ATR_MULTIPLIER}×ATR)")
            print(f"   Lower: ${vwap_lower:.6f} (-{self.VWAP_ATR_MULTIPLIER}×ATR)")
            
            # Signal analysis
            print(f"\n🔍 SIGNAL ANALYSIS:")
            
            # Long signal analysis
            long_vwap_ok = close_price <= vwap_lower
            long_clv_ok = clv <= self.CLV_LONG_THRESHOLD
            long_volume_ok = not pd.isna(z_volume) and z_volume >= self.VOLUME_THRESHOLD
            
            print(f"   📈 LONG SIGNAL:")
            print(f"      VWAP Band: {'✅' if long_vwap_ok else '❌'} Price ${close_price:.6f} {'≤' if long_vwap_ok else '>'} ${vwap_lower:.6f}")
            if not long_vwap_ok:
                vwap_distance = ((close_price - vwap_lower) / vwap_lower) * 100
                print(f"         Distance from trigger: +{vwap_distance:.3f}%")
            
            print(f"      CLV: {'✅' if long_clv_ok else '❌'} {clv:.4f} {'≤' if long_clv_ok else '>'} {self.CLV_LONG_THRESHOLD}")
            if not long_clv_ok:
                clv_distance = clv - self.CLV_LONG_THRESHOLD
                print(f"         Distance from trigger: +{clv_distance:.4f}")
            
            if pd.isna(z_volume):
                print(f"      Volume: ❌ {z_volume_display} (insufficient data)")
            else:
                print(f"      Volume: {'✅' if long_volume_ok else '❌'} {z_volume:.2f} {'≥' if long_volume_ok else '<'} {self.VOLUME_THRESHOLD}")
                if not long_volume_ok:
                    volume_distance = self.VOLUME_THRESHOLD - z_volume
                    print(f"         Distance from trigger: {volume_distance:.2f}")
            
            long_signal_ready = long_vwap_ok and long_clv_ok and long_volume_ok
            print(f"      Overall: {'🎯 READY' if long_signal_ready else '⏳ Waiting'}")
            
            # Short signal analysis
            short_vwap_ok = close_price >= vwap_upper
            short_clv_ok = clv >= self.CLV_SHORT_THRESHOLD
            short_volume_ok = not pd.isna(z_volume) and z_volume >= self.VOLUME_THRESHOLD
            
            print(f"\n   📉 SHORT SIGNAL:")
            print(f"      VWAP Band: {'✅' if short_vwap_ok else '❌'} Price ${close_price:.6f} {'≥' if short_vwap_ok else '<'} ${vwap_upper:.6f}")
            if not short_vwap_ok:
                vwap_distance = ((vwap_upper - close_price) / close_price) * 100
                print(f"         Distance from trigger: +{vwap_distance:.3f}%")
            
            print(f"      CLV: {'✅' if short_clv_ok else '❌'} {clv:.4f} {'≥' if short_clv_ok else '<'} {self.CLV_SHORT_THRESHOLD}")
            if not short_clv_ok:
                clv_distance = self.CLV_SHORT_THRESHOLD - clv
                print(f"         Distance from trigger: +{clv_distance:.4f}")
            
            if pd.isna(z_volume):
                print(f"      Volume: ❌ {z_volume_display} (insufficient data)")
            else:
                print(f"      Volume: {'✅' if short_volume_ok else '❌'} {z_volume:.2f} {'≥' if short_volume_ok else '<'} {self.VOLUME_THRESHOLD}")
                if not short_volume_ok:
                    volume_distance = self.VOLUME_THRESHOLD - z_volume
                    print(f"         Distance from trigger: {volume_distance:.2f}")
            
            short_signal_ready = short_vwap_ok and short_clv_ok and short_volume_ok
            print(f"      Overall: {'🎯 READY' if short_signal_ready else '⏳ Waiting'}")
            
            # Market context
            print(f"\n📈 MARKET CONTEXT:")
            price_vs_vwap = ((close_price - vwap) / vwap) * 100
            print(f"   Price vs VWAP: {price_vs_vwap:+.3f}%")
            print(f"   Market Structure: {'Bullish' if price_vs_vwap > 0 else 'Bearish' if price_vs_vwap < -0.1 else 'Neutral'}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ Debug info error: {e}")
    
    def _should_skip_signal(self, current):
        """Check daily guardrails"""
        # Get safe timestamp
        current_timestamp = self._get_safe_timestamp(current)
        
        print(f"🛡️ DAILY GUARDRAILS CHECK:")
        print(f"   Current date: {current_timestamp.date()}")
        print(f"   Last signal time: {self.last_signal_time}")
        print(f"   Daily PnL: {self.daily_pnl:.4f}")
        print(f"   Daily time stops: {self.daily_time_stops}")
        
        # Reset daily tracking if new day
        if self.current_date != current_timestamp.date():
            print(f"   New day detected - resetting daily tracking")
            self.daily_pnl = 0
            self.daily_time_stops = 0
        
        self.current_date = current_timestamp.date()
        
        # Check if we should stop taking signals
        pnl_threshold = -1.5 * 0.0085
        time_stop_threshold = 3
        
        print(f"   PnL threshold: {pnl_threshold:.4f} (current: {self.daily_pnl:.4f})")
        print(f"   Time stop threshold: {time_stop_threshold} (current: {self.daily_time_stops})")
        
        if self.daily_pnl <= pnl_threshold or self.daily_time_stops >= time_stop_threshold:
            print(f"   ⚠️ Daily limits hit - checking 6-hour cooldown")
            # Check if 6 hours have passed since last signal
            if self.last_signal_time and isinstance(self.last_signal_time, pd.Timestamp):
                time_diff = (current_timestamp - self.last_signal_time).total_seconds()
                hours_passed = time_diff / 3600
                print(f"   Hours since last signal: {hours_passed:.2f}")
                if time_diff < 6 * 3600:
                    print(f"   ❌ Signal blocked - 6-hour cooldown active")
                    return True
                else:
                    print(f"   ✅ 6-hour cooldown passed - signals allowed")
            else:
                print(f"   ✅ No previous signal - signals allowed")
        else:
            print(f"   ✅ Daily limits OK - signals allowed")
        
        return False
    
    def _create_position(self, current, signal_type, direction):
        """Create new position with mean reversion parameters"""
        # Get safe timestamp
        entry_time = self._get_safe_timestamp(current)
        
        position = {
            'entry_time': entry_time,
            'entry_price': self._get_safe_numeric(current, 'close'),
            'signal_type': signal_type,
            'direction': direction,
            'regime': 'Mean-reversion',
            'entry_bar': 0,
            'bars_held': 0,
            'tp1_hit': False,
            'tp1_exit_price': None,
            'tp1_exit_time': None,
            'tp1_pnl': None,
            'runner_size': 0.4,  # 40% of position for runner
            'remaining_size': 1.0,  # Start with 100% position size
            'trailing_active': False,
            'high_since_entry': self._get_safe_numeric(current, 'high') if direction == 1 else self._get_safe_numeric(current, 'low'),
            'low_since_entry': self._get_safe_numeric(current, 'low') if direction == 1 else self._get_safe_numeric(current, 'high'),
            'mfe_history': [],  # Track MFE at each bar
            'mfa_history': [],  # Track MFA at each bar
            'velocity_check': False,  # Fast run check
            'stall_check': False,  # Stall filter check
            'early_adverse': False,  # Early adverse management
            'snap_lock': False,  # Snap-lock for strong extension
            'velocity_trail_adjustment': False,  # Velocity-based trail adjustment
            'scratch_reason': None,
            'entry_equity': self.current_equity,
            'order_refs': [],  # Track COM order references
            'position_ref': None,
            'trigger_orders': {}  # Track trigger orders
        }
        
        # Set mean reversion TP/SL parameters
        self._set_regime_parameters(position, current)
        
        return position
    
    def _set_regime_parameters(self, position, current):
        """Set mean reversion TP/SL parameters"""
        entry_price = self._get_safe_numeric(current, 'close')
        
        if position['direction'] == 1:  # MeanRev_Long
            position['tp1'] = entry_price * 1.0050  # +0.50%
            position['tp2'] = entry_price * 1.0095  # +0.95%
            position['stop_loss'] = entry_price * 0.9920  # -0.80%
            position['trail_distance'] = 0.0020  # 0.20%
        else:  # MeanRev_Short
            position['tp1'] = entry_price * 0.9945  # -0.55%
            position['tp2'] = entry_price * 0.9900  # -1.00%
            position['stop_loss'] = entry_price * 1.0080  # +0.80%
            position['trail_distance'] = 0.0020  # 0.20%
    
    def _update_position_tracking(self, position, current):
        """Update position tracking variables"""
        current_high = self._get_safe_numeric(current, 'high')
        current_low = self._get_safe_numeric(current, 'low')
        
        if position['direction'] == 1:  # Long
            position['high_since_entry'] = max(position['high_since_entry'], current_high)
            position['low_since_entry'] = min(position['low_since_entry'], current_low)
            
            # Calculate current MFE and MFA
            mfe = (position['high_since_entry'] - position['entry_price']) / position['entry_price']
            mfa = (position['low_since_entry'] - position['entry_price']) / position['entry_price']
        else:  # Short
            position['high_since_entry'] = max(position['high_since_entry'], current_high)
            position['low_since_entry'] = min(position['low_since_entry'], current_low)
            
            # Calculate current MFE and MFA
            mfe = (position['entry_price'] - position['low_since_entry']) / position['entry_price']
            mfa = (position['entry_price'] - position['high_since_entry']) / position['entry_price']
        
        position['mfe_history'].append(mfe)
        position['mfa_history'].append(mfa)
    
    def _check_exit_conditions(self, current, position):
        """Check exit conditions with 5-Bar Exit Engine logic"""
        # Check 5-bar hard time stop first
        if position['bars_held'] >= 5:
            return '5-Bar Time Stop'
        
        # Bar-specific checks
        if position['bars_held'] == 2:
            self._bar2_checks(position, current)
        elif position['bars_held'] == 3:
            self._bar3_checks(position, current)
        elif position['bars_held'] == 4:
            self._bar4_checks(position, current)
        
        # Check stall filter (Bar 3) - HIGH PRIORITY
        if position['stall_check']:
            return 'Stall Filter (Bar 3)'
        
        # Check stop loss
        if self._check_stop_loss(current, position):
            return 'Stop Loss'
        
        # Check TP1 (scale-out 60% of position)
        if not position['tp1_hit'] and self._check_tp1(current, position):
            return 'TP1 Hit (60% scale-out)'
        
        # Check TP2 (runner 40% of position)
        if position['tp1_hit'] and self._check_tp2(current, position):
            return 'TP2 Hit (Runner)'
        
        # Check trailing stop after TP1
        if position['tp1_hit'] and self._check_trailing_stop(current, position):
            return 'Trailing Stop'
        
        # Check catastrophic stop loss
        if self._check_catastrophic_stop(position):
            return 'Catastrophic Stop Loss'
        
        return None
    
    def _bar2_checks(self, position, current):
        """Bar 2 checks: velocity and early adverse management"""
        if len(position['mfe_history']) >= 2:
            # Fast run check (impulse)
            mfe_bar2 = position['mfe_history'][1]
            velocity_bar2 = mfe_bar2 / 2
            if velocity_bar2 >= 0.00225:  # 0.225%/bar
                position['velocity_check'] = True
            
            # Early adverse management
            if len(position['mfa_history']) >= 2:
                mfa_bar2 = position['mfa_history'][1]
                if mfa_bar2 <= -0.0050 and not position['tp1_hit']:  # -0.50%
                    # Tighten stop loss via COM
                    new_stop_loss = position['entry_price'] * (1 + 0.0050 if position['direction'] == -1 else 1 - 0.0050)
                    if self.position_ref:
                        self._update_stop_loss_via_com(new_stop_loss)
                    position['stop_loss'] = new_stop_loss
                    position['early_adverse'] = True
    
    def _bar3_checks(self, position, current):
        """Bar 3 checks: stall filter and volume checks - EXECUTES stall scratch"""
        if len(position['mfe_history']) >= 3:
            mfe_bar3 = position['mfe_history'][2]
            if mfe_bar3 < 0.0025:  # < +0.25%
                # Check volume stall
                if hasattr(current, 'zvol') and current['zvol'] < 1.0:
                    position['stall_check'] = True
                    position['scratch_reason'] = 'Volume stall at bar 3'
    
    def _bar4_checks(self, position, current):
        """Bar 4 checks: snap-lock for strong extension and velocity-based trail adjustment"""
        if position['tp1_hit'] and len(position['mfe_history']) >= 4:
            mfe_bar4 = position['mfe_history'][3]
            if mfe_bar4 >= 0.0080:  # ≥ +0.80%
                # Snap-lock: raise trail by +0.05% via COM
                new_trail_distance = position['trail_distance'] + 0.0005
                if self.position_ref:
                    self.update_trailing_stop(self.position_ref, new_trail_distance)
                position['trail_distance'] = new_trail_distance
                position['snap_lock'] = True
        
        # Velocity-based trail tightening rule
        if position['tp1_hit'] and len(position['mfe_history']) >= 4:
            # If TP1 hit on bar 3-4, tighten trail to 0.15% via COM
            if position['bars_held'] >= 3 and position['bars_held'] <= 4:
                if position['trail_distance'] > 0.0015:  # Only tighten if not already tight
                    new_trail_distance = 0.0015
                    if self.position_ref:
                        self.update_trailing_stop(self.position_ref, new_trail_distance)
                    position['trail_distance'] = new_trail_distance
                    position['velocity_trail_adjustment'] = True
    
    def _check_stop_loss(self, current, position):
        """Check if stop loss is hit"""
        if position['direction'] == 1:  # Long
            return current['low'] <= position['stop_loss']
        else:  # Short
            return current['high'] >= position['stop_loss']
    
    def _check_tp1(self, current, position):
        """Check if TP1 is hit"""
        if position['direction'] == 1:  # Long
            return current['high'] >= position['tp1']
        else:  # Short
            return current['low'] <= position['tp1']
    
    def _check_tp2(self, current, position):
        """Check if TP2 is hit"""
        if position['direction'] == 1:  # Long
            return current['high'] >= position['tp2']
        else:  # Short
            return current['low'] <= position['tp2']
    
    def _check_trailing_stop(self, current, position):
        """Check trailing stop after TP1"""
        if not position['tp1_hit']:
            return False
        
        # Activate trailing stop if not already active
        if not position['trailing_active']:
            position['trailing_active'] = True
            if position['direction'] == 1:  # Long
                position['trail_stop'] = position['high_since_entry'] * (1 - position['trail_distance'])
            else:  # Short
                position['trail_stop'] = position['low_since_entry'] * (1 + position['trail_distance'])
        
        # Update trailing stop
        if position['direction'] == 1:  # Long
            new_trail_stop = position['high_since_entry'] * (1 - position['trail_distance'])
            position['trail_stop'] = max(position['trail_stop'], new_trail_stop)
            
            # Check if trailing stop hit
            return current['low'] <= position['trail_stop']
        else:  # Short
            new_trail_stop = position['low_since_entry'] * (1 + position['trail_distance'])
            position['trail_stop'] = min(position['trail_stop'], new_trail_stop)
            
            # Check if trailing stop hit
            return current['high'] >= position['trail_stop']
    
    def _check_catastrophic_stop(self, position):
        """Check catastrophic stop loss"""
        if len(position['mfa_history']) > 0:
            current_mfa = position['mfa_history'][-1]
            if current_mfa <= -0.0090:  # ≤ -0.90%
                return True
            return False
            
    def _execute_exit(self, exit_reason, current):
        """Execute exit based on exit reason"""
        if not self.position:
            return
            
        position = self.position
        exit_price = self._get_exit_price(current, position, exit_reason)
        
        print(f"🚪 EXIT: {exit_reason} at {exit_price}")
        
        # Handle entropy exits (monitored locally, then send market close order to COM)
        if exit_reason in ['5-Bar Time Stop', 'Stall Filter (Bar 3)', 'Catastrophic Stop Loss']:
            self._execute_entropy_exit(exit_reason, current)
            return
            
        # Handle trailing stops - send to COM
        if exit_reason == 'Trailing Stop':
            self._execute_trailing_stop_exit(current)
            return
            
        # COM automatically handles TP1, TP2, and regular stop losses via exit plan
        if exit_reason == 'TP1 Hit (60% scale-out)':
            # COM automatically handles TP1 execution
            position['tp1_hit'] = True
            position['tp1_exit_price'] = position['tp1']
            position['tp1_exit_time'] = current['timestamp']
            print(f"✅ TP1 executed automatically by COM: 60% scale-out at {position['tp1']}")
            
        elif position['tp1_hit'] and exit_reason in ['TP2 Hit (Runner)']:
            # COM automatically handles runner exit
            print(f"✅ Runner exit executed automatically by COM: {exit_reason}")
            # Position will be fully closed by COM
            self.position = None
        else:
            # Regular stop loss - COM automatically handles via exit plan
            print(f"✅ Stop loss executed automatically by COM: {exit_reason}")
            self.position = None
                    
    def _execute_entropy_exit(self, exit_reason, current):
        """
        Execute entropy exits (time stop, stall filter, catastrophic) as market close orders
        
        These exits are monitored locally by the strategy logic and then sent as 
        market close orders to COM for immediate execution.
        """
        if not self.position or not self.position_ref:
            print(f"❌ Cannot execute entropy exit: No position or position_ref")
            return
            
        print(f"🔥 ENTROPY EXIT: {exit_reason} - Strategy detected locally, sending market close order to COM")
        
        # Send market close order to COM for immediate execution
        success = self.close_position(self.position_ref, quantity="ALL")
        
        if success:
            print(f"✅ Entropy exit executed: {exit_reason} - Position closed via COM")
            self.position = None
        else:
            print(f"❌ Entropy exit failed: {exit_reason} - COM close order failed")
                    
    def _execute_trailing_stop_exit(self, current):
        """Execute trailing stop exit via COM trigger order"""
        if not self.position:
            return
            
        # Create trailing stop trigger order via COM
        trigger_type = "TRAILING_STOP"
        trigger_price = self.position.get('trail_stop', current['close'])
        order_side = "SELL" if self.position['direction'] == 1 else "BUY"
        quantity = self.position.get('remaining_size', 1.0) * 0.001  # Convert to quantity
        
        print(f"📈 TRAILING STOP: Creating trigger order at {trigger_price}")
        
        trigger_ref = self.create_trigger_order(
            trigger_type=trigger_type,
            trigger_price=trigger_price,
            order_side=order_side,
            quantity=quantity,
            order_type="MARKET",
            reduce_only=True
        )
        
        if trigger_ref:
            print(f"✅ Trailing stop trigger order created: {trigger_ref}")
            self.position['trigger_orders'][trigger_ref] = {
                'type': 'TRAILING_STOP',
                'trigger_price': trigger_price,
                'status': 'ACTIVE'
            }
        else:
            print(f"❌ Trailing stop trigger order failed")
    
    def _update_stop_loss_via_com(self, new_stop_loss):
        """Update stop loss via COM API"""
        if not self.position_ref:
            return False
        
        timestamp = int(time.time())
        
        payload = {
            "stop_loss_price": new_stop_loss,
            "update_type": "STOP_LOSS_ADJUSTMENT"
        }
        
        body = json.dumps(payload)
        signature = self.create_hmac_signature(timestamp, "PUT", f"/api/v1/positions/{self.position_ref}/stop-loss", body)
        
        headers = {
            "Authorization": f'HMAC key_id="{API_KEY}", signature="{signature}", ts={timestamp}',
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.put(
                f"{COM_BASE_URL}/api/v1/positions/{self.position_ref}/stop-loss",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✅ Stop loss updated via COM: {new_stop_loss}")
                return True
            else:
                print(f"❌ Stop loss update failed: {response.status_code}")
                return False
            
        except Exception as e:
            print(f"❌ Stop loss update error: {e}")
            return False
    
    def _get_exit_price(self, current, position, exit_reason):
        """Get exit price based on exit reason"""
        if exit_reason == 'Stop Loss':
            return position['stop_loss']
        elif exit_reason == 'TP1 Hit (60% scale-out)':
            return position['tp1']
        elif exit_reason == 'TP2 Hit (Runner)':
            return position['tp2']
        elif exit_reason == 'Trailing Stop':
            return position['trail_stop']
        elif exit_reason == '5-Bar Time Stop':
            return current['close']  # Exit at current market price
        elif exit_reason == 'Stall Filter (Bar 3)':
            return current['close']  # Exit at current market price
        elif exit_reason == 'Catastrophic Stop Loss':
            return current['close']  # Exit at current market price
        else:
            return current['close']
    
    def _update_daily_tracking(self, exit_reason, net_pnl):
        """Update daily tracking for guardrails"""
        self.daily_pnl += net_pnl
        if exit_reason == '5-Bar Time Stop':
            self.daily_time_stops += 1
    
    def _load_data_like_backtester(self, file_path):
        """Load data using EXACT same method as backtester"""
        try:
            # Check if this is the live data format (datadoge.csv)
            if file_path.endswith("datadoge.csv"):
                # Live data format: Timestamp,DateTime,Open,High,Low,Close,Volume
                df = pd.read_csv(file_path)
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Timestamp': 'timestamp',
                    'DateTime': 'datetime',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Convert timestamp from milliseconds to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            else:
                # Historical data format: timestamp, open, high, low, close, volume, vwap, trades, buy_volume, buy_value
                df = pd.read_csv(file_path, header=None)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades', 'buy_volume', 'buy_value']
                
                # Convert timestamp from nanoseconds to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Ensure we have the required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None
            
            print(f"✅ Data loaded: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _calculate_indicators_like_backtester(self, df):
        """Calculate indicators using EXACT same method as backtester"""
        print("📊 Calculating indicators (backtester method)...")
        
        # Basic calculations - EXACT same order as backtester
        df = self._calculate_returns_like_backtester(df)
        df = self._calculate_realized_volatility_like_backtester(df)
        df = self._calculate_skew_kurt_like_backtester(df)
        df = self._calculate_atr_like_backtester(df)
        df = self._calculate_clv_like_backtester(df)
        df = self._calculate_vwap_like_backtester(df)
        df = self._calculate_relative_volume_like_backtester(df)
        df = self._calculate_entropy_like_backtester(df)
        df = self._calculate_regime_like_backtester(df)
        
        print("✅ Indicators calculated (backtester method)")
        return df
    
    def _calculate_returns_like_backtester(self, df):
        """Calculate log returns - EXACT match to backtester"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def _calculate_realized_volatility_like_backtester(self, df, window=48):
        """Calculate rolling realized volatility - EXACT match to backtester"""
        df['rv'] = df['returns'].rolling(window=window).std()
        return df
    
    def _calculate_skew_kurt_like_backtester(self, df, window=48):
        """Calculate rolling skewness and kurtosis - EXACT match to backtester"""
        df['skew'] = df['returns'].rolling(window=window).skew()
        df['kurt'] = df['returns'].rolling(window=window).kurt()
        return df
    
    def _calculate_atr_like_backtester(self, df, window=14):
        """Calculate Average True Range - EXACT match to backtester"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=window).mean()
        return df
            
    def _calculate_clv_like_backtester(self, df):
        """Calculate Close Location Value - EXACT match to backtester"""
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['clv'] = df['clv'].clip(-1, 1)
        return df
    
    def _calculate_vwap_like_backtester(self, df, window=48):
        """Calculate rolling VWAP - EXACT match to backtester"""
        volume_price = df['close'] * df['volume']
        rolling_vwap = volume_price.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        df['vwap'] = rolling_vwap
        return df
    
    def _calculate_relative_volume_like_backtester(self, df, lookback=48):
        """Calculate relative volume - EXACT match to backtester"""
        df['zvol'] = df['volume'] / df['volume'].rolling(window=lookback).mean()
        return df
    
    def _calculate_entropy_like_backtester(self, df, window=20):
        """Calculate entropy - EXACT match to backtester"""
        def entropy_calc(returns):
            if len(returns) < 2:
                return 0.0
            returns = returns.dropna()
            if len(returns) < 2:
                return 0.0
            try:
                # Calculate entropy as -sum(p * log(p)) where p is normalized returns
                abs_returns = np.abs(returns)
                if abs_returns.sum() == 0:
                    return 0.0
                p = abs_returns / abs_returns.sum()
                p = p[p > 0]  # Remove zeros
                return -np.sum(p * np.log(p + 1e-10))
            except:
                return 0.0
        
        df['entropy'] = df['returns'].rolling(window=window).apply(entropy_calc, raw=False)
        return df
            
    def _calculate_regime_like_backtester(self, df):
        """Calculate regime - EXACT match to backtester"""
        # For now, set all to Mean-reversion (as per live strategy focus)
        df['regime'] = 'Mean-reversion'
        return df
    
    def _save_indicators_to_csv(self, current):
        """Save current candle indicators to CSV for analysis (append mode)"""
        try:
            import csv
            
            # Prepare indicator data
            indicator_data = {
                'timestamp': current['timestamp'],
                'datetime': current['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'open': self._get_safe_numeric(current, 'open'),
                'high': self._get_safe_numeric(current, 'high'),
                'low': self._get_safe_numeric(current, 'low'),
                'close': self._get_safe_numeric(current, 'close'),
                'volume': self._get_safe_numeric(current, 'volume'),
                'atr': self._get_safe_numeric(current, 'atr'),
                'vwap': self._get_safe_numeric(current, 'vwap'),
                'clv': self._get_safe_numeric(current, 'clv'),
                'zvol': self._get_safe_numeric(current, 'zvol'),
                'entropy': self._get_safe_numeric(current, 'entropy'),
                'regime': current.get('regime', 'Mean-reversion'),
                'returns': current.get('returns', np.nan)
            }
            
            # Add calculated analysis fields
            close_price = indicator_data['close']
            vwap = indicator_data['vwap']
            atr = indicator_data['atr']
            clv = indicator_data['clv']
            zvol = indicator_data['zvol']
            
            # VWAP bands
            indicator_data['vwap_upper'] = vwap + (self.VWAP_ATR_MULTIPLIER * atr)
            indicator_data['vwap_lower'] = vwap - (self.VWAP_ATR_MULTIPLIER * atr)
            indicator_data['price_vs_vwap_pct'] = ((close_price - vwap) / vwap) * 100
            
            # Signal analysis
            indicator_data['long_vwap_ok'] = close_price <= indicator_data['vwap_lower']
            indicator_data['short_vwap_ok'] = close_price >= indicator_data['vwap_upper']
            indicator_data['long_clv_ok'] = clv <= self.CLV_LONG_THRESHOLD
            indicator_data['short_clv_ok'] = clv >= self.CLV_SHORT_THRESHOLD
            indicator_data['volume_ok'] = not pd.isna(zvol) and zvol >= self.VOLUME_THRESHOLD
            
            # Signal flags
            indicator_data['long_signal'] = (
                indicator_data['long_vwap_ok'] and 
                indicator_data['long_clv_ok'] and 
                indicator_data['volume_ok']
            )
            indicator_data['short_signal'] = (
                indicator_data['short_vwap_ok'] and 
                indicator_data['short_clv_ok'] and 
                indicator_data['volume_ok']
            )
            
            # Position tracking
            indicator_data['position_open'] = self.is_position_open()
            if self.position:
                indicator_data['position_direction'] = 'LONG' if self.position['direction'] == 1 else 'SHORT'
                indicator_data['position_bars_held'] = self.position['bars_held']
                indicator_data['position_mfe'] = self.position['mfe_history'][-1] * 100 if self.position['mfe_history'] else 0
                indicator_data['position_mfa'] = self.position['mfa_history'][-1] * 100 if self.position['mfa_history'] else 0
            else:
                indicator_data['position_direction'] = 'NONE'
                indicator_data['position_bars_held'] = 0
                indicator_data['position_mfe'] = 0
                indicator_data['position_mfa'] = 0
            
            # Write to CSV
            file_exists = os.path.exists(self.indicators_csv_filename)
            
            with open(self.indicators_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = list(indicator_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists or not self.indicators_csv_initialized:
                    writer.writeheader()
                    self.indicators_csv_initialized = True
                
                # Write data
                writer.writerow(indicator_data)
            
            # Print periodic status (every 10 candles)
            if len(self.candles) % 10 == 0:
                print(f"💾 Indicators logged to CSV (candle {len(self.candles)})")
            
        except Exception as e:
            print(f"❌ Error saving indicators to CSV: {e}")
    
    def _save_all_historical_candles(self, df):
        """Save all historical candles with indicators to CSV for analysis"""
        try:
            import csv
            
            # Create CSV file with all historical data
            csv_filename = 'live_indicators_log.csv'
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume',
                    'atr', 'vwap', 'clv', 'zvol', 'entropy', 'regime', 'returns',
                    'vwap_upper', 'vwap_lower', 'long_signal', 'short_signal', 'signal_strength'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for _, row in df.iterrows():
                    # Prepare indicator data
                    indicator_data = {
                        'timestamp': row['timestamp'],
                        'datetime': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'atr': row['atr'],
                        'vwap': row['vwap'],
                        'clv': row['clv'],
                        'zvol': row['zvol'],
                        'entropy': row['entropy'],
                        'regime': row['regime'],
                        'returns': row['returns']
                    }
                    
                    # Add calculated analysis fields
                    close_price = indicator_data['close']
                    vwap = indicator_data['vwap']
                    atr = indicator_data['atr']
                    clv = indicator_data['clv']
                    zvol = indicator_data['zvol']
                    
                    # VWAP bands
                    vwap_upper = vwap + (0.6 * atr)
                    vwap_lower = vwap - (0.6 * atr)
                    
                    # Signal analysis
                    long_signal = (close_price <= vwap_lower and clv <= -0.4 and zvol >= 1.2)
                    short_signal = (close_price >= vwap_upper and clv >= 0.4 and zvol >= 1.2)
                    
                    # Signal strength
                    signal_strength = 0
                    if long_signal:
                        signal_strength = 1
                    elif short_signal:
                        signal_strength = -1
                    
                    # Add analysis fields
                    indicator_data.update({
                        'vwap_upper': vwap_upper,
                        'vwap_lower': vwap_lower,
                        'long_signal': long_signal,
                        'short_signal': short_signal,
                        'signal_strength': signal_strength
                    })
                    
                    writer.writerow(indicator_data)
            
            print(f"💾 Saved {len(df)} historical candles to {csv_filename}")
            
        except Exception as e:
            print(f"❌ Error saving historical candles to CSV: {e}")
    
    def process_new_candle(self, candle_data):
        """Process new 5-minute candle data by recalculating all indicators (live trader method)"""
        try:
            # Validate input data
            if len(candle_data) < 7:
                print(f"❌ Invalid candle data: expected 7 columns, got {len(candle_data)}")
                return
                
            # Parse candle data
            timestamp = int(candle_data[0])
            open_price = float(candle_data[2])
            high_price = float(candle_data[3])
            low_price = float(candle_data[4])
            close_price = float(candle_data[5])
            volume = float(candle_data[6])
            
            # Convert timestamp to datetime
            current_timestamp = pd.to_datetime(timestamp, unit='ms')
            
            # Add new candle to our historical buffer
            new_candle = {
                'timestamp': current_timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            self.candles.append(new_candle)
            
            # Keep only last 200 candles to prevent memory issues (need more for rolling calculations)
            if len(self.candles) > 200:
                self.candles = self.candles[-200:]
            
            # Recalculate ALL indicators using the updated dataset (including new candle)
            # This ensures we use the EXACT same method as backtester
            df = pd.DataFrame(self.candles)
            df = self._calculate_indicators_like_backtester(df)
            
            # Get the latest calculated values (for the new candle)
            latest_row = df.iloc[-1]
            
            # Create current data with freshly calculated indicators
            current = {
                'timestamp': current_timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'atr': latest_row['atr'],
                'vwap': latest_row['vwap'],
                'clv': latest_row['clv'],
                'zvol': latest_row['zvol'],
                'entropy': latest_row['entropy'],
                'regime': latest_row['regime'],
                'returns': latest_row['returns']
            }
            
            print(f"🔍 Live trader - Freshly calculated indicators:")
            print(f"   ATR: {current['atr']:.6f}")
            print(f"   VWAP: {current['vwap']:.6f}")
            print(f"   CLV: {current['clv']:.4f}")
            print(f"   zVol: {current['zvol']:.2f}")
            print(f"   Entropy: {current['entropy']:.4f}")
            
            # Print detailed debug information
            self._print_debug_info(current)
            
            # Save indicators to CSV for analysis
            self._save_indicators_to_csv(current)
            
            # Check for new signal if not in position
            if not self.position:
                print(f"🔍 Checking for entry signal (no position open)...")
                signal = self._check_entry_signal(current)
                if signal:
                    print(f"🎯 Signal detected: {signal}")
                    self._handle_entry_signal(signal, current)
                else:
                    print(f"⏳ No signal generated")
            else:
                print(f"📊 Position already open, checking exit conditions...")
                # Update position tracking
                self._update_position_tracking(self.position, current)
                
                # Increment bars held
                self.position['bars_held'] += 1
                
                # Check for exit conditions
                exit_reason = self._check_exit_conditions(current, self.position)
                if exit_reason:
                    print(f"🚪 Exit condition detected: {exit_reason}")
                    self._execute_exit(exit_reason, current)
            
        except Exception as e:
            print(f"❌ Error processing candle: {e}")
            import traceback
            traceback.print_exc()
            traceback.print_exc()
    
    def is_position_open(self):
        """Check if we have an open position based on WebSocket status"""
        # Check local position state
        if self.position is None:
            return False
        
        # Check WebSocket position status if available
        if self.position_status:
            return self.position_status.get('size', 0) > 0
        
        # Fallback to local position state
        return self.position is not None
    
    def _check_entry_signal(self, current):
        """Check for mean reversion entry signals"""
        # Get safe numeric values
        close_price = self._get_safe_numeric(current, 'close')
        vwap = current['vwap']
        atr = current['atr']
        clv = current['clv']
        zvol = current['zvol']
        
        print(f"🔍 SIGNAL CHECK DEBUG:")
        print(f"   Close: {close_price:.6f}")
        print(f"   VWAP: {vwap:.6f}")
        print(f"   ATR: {atr:.6f}")
        print(f"   CLV: {clv:.4f}")
        print(f"   zVol: {zvol:.2f}")
        print(f"   Regime: {current['regime']}")
        
        # Skip signal generation if z-Volume is NaN (insufficient data)
        if pd.isna(zvol):
            print(f"❌ zVol is NaN - skipping signal")
            return None
        
        # Calculate VWAP bands
        vwap_upper = vwap + self.VWAP_ATR_MULTIPLIER * atr
        vwap_lower = vwap - self.VWAP_ATR_MULTIPLIER * atr
        
        print(f"   VWAP Upper: {vwap_upper:.6f}")
        print(f"   VWAP Lower: {vwap_lower:.6f}")
        
        # Check LONG conditions
        long_vwap_ok = close_price <= vwap_lower
        long_clv_ok = clv <= self.CLV_LONG_THRESHOLD
        long_volume_ok = zvol >= self.VOLUME_THRESHOLD
        long_regime_ok = current['regime'] == 'Mean-reversion'
        
        print(f"   LONG CHECKS:")
        print(f"     VWAP: {close_price:.6f} <= {vwap_lower:.6f} = {long_vwap_ok}")
        print(f"     CLV: {clv:.4f} <= {self.CLV_LONG_THRESHOLD} = {long_clv_ok}")
        print(f"     Volume: {zvol:.2f} >= {self.VOLUME_THRESHOLD} = {long_volume_ok}")
        print(f"     Regime: {current['regime']} == 'Mean-reversion' = {long_regime_ok}")
        
        # Check SHORT conditions
        short_vwap_ok = close_price >= vwap_upper
        short_clv_ok = clv >= self.CLV_SHORT_THRESHOLD
        short_volume_ok = zvol >= self.VOLUME_THRESHOLD
        short_regime_ok = current['regime'] == 'Mean-reversion'
        
        print(f"   SHORT CHECKS:")
        print(f"     VWAP: {close_price:.6f} >= {vwap_upper:.6f} = {short_vwap_ok}")
        print(f"     CLV: {clv:.4f} >= {self.CLV_SHORT_THRESHOLD} = {short_clv_ok}")
        print(f"     Volume: {zvol:.2f} >= {self.VOLUME_THRESHOLD} = {short_volume_ok}")
        print(f"     Regime: {current['regime']} == 'Mean-reversion' = {short_regime_ok}")
            
        # MEAN-REVERSION AVWAP SNAPBACK LONG
        if (long_regime_ok and long_vwap_ok and long_clv_ok and long_volume_ok):
            print(f"✅ LONG SIGNAL TRIGGERED!")
            return 'MeanRev_Long', 1
        
        # MEAN-REVERSION AVWAP SNAPBACK SHORT
        elif (short_regime_ok and short_vwap_ok and short_clv_ok and short_volume_ok):
            print(f"✅ SHORT SIGNAL TRIGGERED!")
            return 'MeanRev_Short', -1
        else:
            print(f"❌ No signal - conditions not met")
            return None
    
    def _handle_entry_signal(self, signal, current):
        """Handle entry signal by creating position and sending order to COM"""
        try:
            signal_type, direction = signal
            close_price = self._get_safe_numeric(current, 'close')
            
            print(f"🎯 SIGNAL DETECTED: {signal_type} at {close_price:.6f}")
            
            # Check daily guardrails
            if self._should_skip_signal(current):
                print("⏸️ Signal skipped due to daily guardrails")
                return
            
            # Create position object
            self.position = self._create_position(current, signal_type, direction)
            
            # Calculate TP/SL prices
            if direction == 1:  # Long
                tp1_price = close_price * 1.0050  # +0.50%
                tp2_price = close_price * 1.0095  # +0.95%
                stop_loss_price = close_price * 0.9920  # -0.80%
                order_side = "BUY"
            else:  # Short
                tp1_price = close_price * 0.9945  # -0.55%
                tp2_price = close_price * 0.9900  # -1.00%
                stop_loss_price = close_price * 1.0080  # +0.80%
                order_side = "SELL"
            
            print(f"📊 Order Details:")
            print(f"   Side: {order_side}")
            print(f"   Entry: {close_price:.6f}")
            print(f"   TP1: {tp1_price:.6f} (60% scale-out)")
            print(f"   TP2: {tp2_price:.6f} (40% runner)")
            print(f"   SL: {stop_loss_price:.6f}")
            
            # Send order to COM with complete exit plan
            order_ref = self.create_entry_order_with_exit_plan(
                side=order_side,
                entry_price=close_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                stop_loss_price=stop_loss_price
            )
            
            if order_ref:
                print(f"✅ Order sent to COM: {order_ref}")
                self.position['order_refs'] = [order_ref]
                self.last_signal_time = current['timestamp']
                print(f"🚀 Position created: {signal_type} - COM will manage full exit plan")
            else:
                print(f"❌ Order creation failed - clearing position")
                self.position = None
                
        except Exception as e:
            print(f"❌ Error handling entry signal: {e}")
            import traceback
            traceback.print_exc()
            self.position = None
            
    def load_historical_data(self):
        """Load historical data using EXACT same method as backtester"""
        if not os.path.exists(CSV_FILENAME):
            print(f"❌ CSV file not found: {CSV_FILENAME}")
            return False
        
        try:
            print(f"📊 Loading historical data from {CSV_FILENAME} (backtester method)...")
            
            # Load the SAME data as backtester using identical method
            df = self._load_data_like_backtester(CSV_FILENAME)
            
            if df is None or len(df) == 0:
                print("❌ Failed to load historical data")
                self.candles = []
                return False
            
            print(f"📊 Loaded {len(df)} candles (same as backtester)")
            
            # Calculate indicators using EXACT same method as backtester
            df = self._calculate_indicators_like_backtester(df)
            
            # Convert to candle format for live processing
            self.candles = []
            for _, row in df.iterrows():
                candle_data = {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'atr': row['atr'],
                    'vwap': row['vwap'],
                    'clv': row['clv'],
                    'zvol': row['zvol'],
                    'entropy': row['entropy'],
                    'regime': row['regime'],
                    'returns': row['returns']
                }
                self.candles.append(candle_data)
            
            print(f"✅ Loaded {len(self.candles)} historical candles with pre-calculated indicators (backtester method)")
            
            # Save ALL historical candles to CSV for analysis
            self._save_all_historical_candles(df)
            
            return len(self.candles) > 0
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            self.candles = []
            return False
    

    def monitor_csv_file(self):
        """Monitor CSV file for new candles"""
        print(f"📁 Monitoring {CSV_FILENAME} for new candles...")
        
        # Load historical data first
        if not self.load_historical_data():
            print(f"⏳ Waiting for file to appear...")
        
        last_modified = 0
        last_line_count = 0
        file_waiting = True
        consecutive_missing = 0
        
        while True:
            try:
                # Check if file exists and has been modified
                if os.path.exists(CSV_FILENAME):
                    if file_waiting:
                        print(f"✅ CSV file found: {CSV_FILENAME}")
                        file_waiting = False
                        consecutive_missing = 0
                    elif consecutive_missing > 0:
                        # File was missing but now found again
                        consecutive_missing = 0
                    
                    current_modified = os.path.getmtime(CSV_FILENAME)
                    
                    if current_modified > last_modified:
                        # File has been modified, check for new lines
                        with open(CSV_FILENAME, 'r') as f:
                            lines = f.readlines()
                            current_line_count = len(lines)
                        
                        if current_line_count > last_line_count:
                            # New lines added, process the latest candle
                            if current_line_count >= 2:  # At least header + 1 data row
                                latest_candle = lines[-1].strip().split(',')
                                
                                if len(latest_candle) >= 7:  # Ensure we have all columns
                                    self.process_new_candle(latest_candle)
                            
                            last_line_count = current_line_count
                        
                        last_modified = current_modified
                else:
                    # File doesn't exist
                    consecutive_missing += 1
                    if not file_waiting and consecutive_missing >= 3:
                        print(f"⚠️ CSV file disappeared: {CSV_FILENAME} (missing for {consecutive_missing} checks)")
                        file_waiting = True
                
                # Adaptive sleep: shorter when file is being updated, longer when stable
                if current_modified > last_modified:
                    time.sleep(0.5)  # Check more frequently when file is updating
                else:
                    time.sleep(2)    # Check less frequently when file is stable
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping CSV monitor...")
                break
            except Exception as e:
                print(f"❌ CSV monitor error: {e}")
                time.sleep(5)
    
    def start_live_trading(self):
        """Start live trading with WebSocket monitoring"""
        print("🚀 Starting live trading with WebSocket monitoring...")
        print(f"🔐 Using {'PAPER' if PAPER_TRADING else 'LIVE'} trading environment")
        
        # Start WebSocket monitoring in background
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_websocket_monitoring())
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        # Start CSV monitoring in background thread
        csv_thread = threading.Thread(target=self.monitor_csv_file, daemon=True)
        csv_thread.start()
        
        print("✅ Live trading started!")
        print("📡 WebSocket monitoring: Real-time position and order updates")
        print("📁 CSV monitoring: New candle data from datadoge.csv")
        print("🎯 Mean Reversion Strategy: Monitoring for entry signals")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping live trading...")
            
            # Close WebSocket connection
            if self.ws_connected:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.ws_client.close())
            
            # Cancel any active orders
            for order_ref in list(self.active_orders.keys()):
                self.cancel_order(order_ref)
            
            # Cancel any trigger orders
            for trigger_ref in list(self.trigger_orders.keys()):
                self.cancel_trigger_order(trigger_ref)
            
            # Close any open position
            if self.position and self.position_ref:
                self.close_position(self.position_ref)
            
            print("✅ Live trading stopped")

def main():
    # Initialize live trader
    trader = LiveEdge5RAVFTrader()
    
    # Start live trading
    trader.start_live_trading()

if __name__ == "__main__":
    main()
