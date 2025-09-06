import os
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from telegram import Bot
from telegram.constants import ParseMode
from apscheduler.schedulers.background import BackgroundScheduler

from ml.model import prepare_and_backtest, infer_next_signal

# ---- ENV config ----
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '')
DATA_CSV_URL       = os.environ.get('DATA_CSV_URL', '')   # Google Sheets "Publish to the web" CSV link
PAIR_NAME          = os.environ.get('PAIR_NAME', 'USD/BRL_OTC')
THRESHOLD          = float(os.environ.get('THRESHOLD', '0.6'))  # 0.6 => BUY if prob>=0.6; SELL if <=0.4
MIN_ROWS           = int(os.environ.get('MIN_ROWS', '500'))

bot = Bot(token=TELEGRAM_BOT_TOKEN)
last_bar_time = None  # prevent duplicate sends

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def run_once():
    global last_bar_time
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not DATA_CSV_URL:
            logging.warning("Missing ENV (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / DATA_CSV_URL).")
            return

        df = fetch_csv(DATA_CSV_URL)

        # backtest + latest features
        report, latest_feat = prepare_and_backtest(df, min_rows=MIN_ROWS)

        # next signal
        next_sig = infer_next_signal(report['trained_model'], latest_feat, trade_threshold=THRESHOLD)

        # dedupe by last timestamp
        tcol = None
      
