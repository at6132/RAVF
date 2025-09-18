# =================== LIVE TRADING CONFIGURATION ===================

# Trading Environment
PAPER_TRADING = True  # Set to False for live trading

# COM API Configuration
COM_BASE_URL = "http://localhost:8000"
API_KEY = "UkFWRl8yMDI1MDkxOF8xNzQ4MTnzttqDrGFUBzAw5ytHSlbutqm_v_txxcjBm5f6g8k-lA"
SECRET = "YzCtRLfEGxjrpNCDOUA146KgR10WTedzB8JiEom2tLR031Y-g5M8_VJD3eTMoKMJE56Efbb-B6C52tq4Nr-Aog"
SALT = "5OYw8B2yCc6_IdOo25eNavAntwnwNyZux-nroed0si0"

# Strategy Configuration
STRATEGY_NAME = "RAVF_V2"
STRATEGY_ID = "edge5_ravf_live"
INSTANCE_ID = "instance_001"
OWNER = "live_trader"

# Trading Configuration
SYMBOL = "DOGE_USDT"
CSV_FILENAME = "datadoge.csv"
STARTING_EQUITY = 10000  # Starting capital in USD

# Risk Management
MAX_DAILY_LOSS = -0.01275  # -1.5R daily loss limit (1.5 * 0.0085)
MAX_DAILY_TIME_STOPS = 3  # Maximum time stops per day
DAILY_COOLDOWN_HOURS = 6  # Hours to wait after hitting daily limits

# Position Sizing
POSITION_SIZE = 0.001  # Fixed quantity in contracts
MAX_POSITION_SIZE = 0.005  # Maximum position size

# Fee Configuration
FEE_PER_TRADE = 0.0003  # 0.03% per trade
ROUND_TRIP_FEE = 0.0006  # 0.06% round trip

# Strategy Parameters (Mean Reversion)
MEAN_REV_TP1_LONG = 0.0050   # +0.50% for long positions
MEAN_REV_TP2_LONG = 0.0095   # +0.95% for long positions
MEAN_REV_SL_LONG = -0.0080   # -0.80% for long positions

MEAN_REV_TP1_SHORT = -0.0055  # -0.55% for short positions
MEAN_REV_TP2_SHORT = -0.0100  # -1.00% for short positions
MEAN_REV_SL_SHORT = 0.0080    # +0.80% for short positions

# 5-Bar Exit Engine Parameters
BAR2_VELOCITY_THRESHOLD = 0.00225  # 0.225%/bar for velocity check
BAR2_EARLY_ADVERSE_THRESHOLD = -0.0050  # -0.50% for early adverse management
BAR3_STALL_MFE_THRESHOLD = 0.0025  # < +0.25% for stall filter
BAR3_VOLUME_STALL_THRESHOLD = 1.0  # zVol < 1.0 for volume stall
BAR4_SNAP_LOCK_THRESHOLD = 0.0080  # ≥ +0.80% for snap-lock
BAR4_VELOCITY_TRAIL_TIGHTENING = 0.0015  # Tighten trail to 0.15%
BAR5_HARD_TIME_STOP = 5  # Mandatory exit after 5 bars

# Trailing Stop Parameters
INITIAL_TRAIL_DISTANCE = 0.0020  # 0.20% initial trailing distance
SNAP_LOCK_TRAIL_ADJUSTMENT = 0.0005  # +0.05% trail adjustment for snap-lock
VELOCITY_TRAIL_TIGHTENING = 0.0015  # Tighten trail to 0.15% for velocity

# Entry Signal Parameters
VWAP_ATR_MULTIPLIER = 0.6  # VWAP ± 0.6*ATR for entry
CLV_LONG_THRESHOLD = -0.4  # CLV <= -0.4 for long entry
CLV_SHORT_THRESHOLD = 0.4  # CLV >= 0.4 for short entry
VOLUME_THRESHOLD = 1.2  # zVol >= 1.2 for entry

# Technical Indicator Windows
ATR_WINDOW = 14  # ATR calculation window
VWAP_WINDOW = 48  # VWAP calculation window (rolling)
VOLUME_LOOKBACK = 48  # Volume relative calculation lookback

# Catastrophic Stop Loss
CATASTROPHIC_STOP_THRESHOLD = -0.0090  # ≤ -0.90% catastrophic stop

# ================================================================
