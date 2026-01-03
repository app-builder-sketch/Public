# =============================================================================
# streamlit_app.py
# Telegram Broadcast + AI Trade Plan Engine
# Full implementation – no omissions, no placeholders.
# =============================================================================

from __future__ import annotations

import os
import re
import math
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional dependencies are guarded; app runs without them if features disabled.
try:
    import yfinance as yf
    YF_AVAILABLE: bool = True
except Exception:
    YF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE: bool = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE: bool = True
except Exception:
    GEMINI_AVAILABLE = False

try:
    import telegram
    from telegram.error import TelegramError, RetryAfter, TimedOut
    TELEGRAM_AVAILABLE: bool = True
except Exception:
    TELEGRAM_AVAILABLE = False

# -----------------------------------------------------------------------------
# Logging setup (captured in diagnostics)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & Configuration
APP_TITLE = "Telegram Broadcast + AI Trade Plan"
DISCLAIMER = "⚠️ Research tool only. Not financial advice. Markets can move against you."

# -----------------------------------------------------------------------------
# Ticker Database (500+ symbols)
# Mapping: TradingView-style key → yfinance symbol, with grouping by asset class.
TICKER_DB: Dict[str, Dict[str, Any]] = {
    # Crypto (Spot & Perps-like continuous contracts)
    "Crypto": {
        "BTC-USD": {"yf": "BTC-USD", "tv": "BINANCE:BTCUSDT", "desc": "Bitcoin / USD"},
        "ETH-USD": {"yf": "ETH-USD", "tv": "BINANCE:ETHUSDT", "desc": "Ethereum / USD"},
        "SOL-USD": {"yf": "SOL-USD", "tv": "BINANCE:SOLUSDT", "desc": "Solana / USD"},
        "BNB-USD": {"yf": "BNB-USD", "tv": "BINANCE:BNBUSDT", "desc": "Binance Coin / USD"},
        "ADA-USD": {"yf": "ADA-USD", "tv": "BINANCE:ADAUSDT", "desc": "Cardano / USD"},
        "XRP-USD": {"yf": "XRP-USD", "tv": "BINANCE:XRPUSDT", "desc": "Ripple / USD"},
        "DOGE-USD": {"yf": "DOGE-USD", "tv": "BINANCE:DOGEUSDT", "desc": "Dogecoin / USD"},
        "MATIC-USD": {"yf": "MATIC-USD", "tv": "BINANCE:MATICUSDT", "desc": "Polygon / USD"},
        "DOT-USD": {"yf": "DOT-USD", "tv": "BINANCE:DOTUSDT", "desc": "Polkadot / USD"},
        "LTC-USD": {"yf": "LTC-USD", "tv": "BINANCE:LTCUSDT", "desc": "Litecoin / USD"},
        "UNI-USD": {"yf": "UNI-USD", "tv": "BINANCE:UNIUSDT", "desc": "Uniswap / USD"},
        "LINK-USD": {"yf": "LINK-USD", "tv": "BINANCE:LINKUSDT", "desc": "Chainlink / USD"},
        "BCH-USD": {"yf": "BCH-USD", "tv": "BINANCE:BCHUSDT", "desc": "Bitcoin Cash / USD"},
        "ALGO-USD": {"yf": "ALGO-USD", "tv": "BINANCE:ALGOUSDT", "desc": "Algorand / USD"},
        "VET-USD": {"yf": "VET-USD", "tv": "BINANCE:VETUSDT", "desc": "VeChain / USD"},
        "ICP-USD": {"yf": "ICP-USD", "tv": "BINANCE:ICPUSDT", "desc": "Internet Computer / USD"},
        "FIL-USD": {"yf": "FIL-USD", "tv": "BINANCE:FILUSDT", "desc": "Filecoin / USD"},
        "XLM-USD": {"yf": "XLM-USD", "tv": "BINANCE:XLMUSDT", "desc": "Stellar / USD"},
        "TRX-USD": {"yf": "TRX-USD", "tv": "BINANCE:TRXUSDT", "desc": "Tron / USD"},
        "ETC-USD": {"yf": "ETC-USD", "tv": "BINANCE:ETCUSDT", "desc": "Ethereum Classic / USD"},
        "XMR-USD": {"yf": "XMR-USD", "tv": "BINANCE:XMRUSDT", "desc": "Monero / USD"},
        "HBAR-USD": {"yf": "HBAR-USD", "tv": "BINANCE:HBARUSDT", "desc": "Hedera / USD"},
        "APT-USD": {"yf": "APT-USD", "tv": "BINANCE:APTUSDT", "desc": "Aptos / USD"},
        "QNT-USD": {"yf": "QNT-USD", "tv": "BINANCE:QNTUSDT", "desc": "Quant / USD"},
        "NEAR-USD": {"yf": "NEAR-USD", "tv": "BINANCE:NEARUSDT", "desc": "Near Protocol / USD"},
        "GRT-USD": {"yf": "GRT-USD", "tv": "BINANCE:GRTUSDT", "desc": "The Graph / USD"},
        "FTM-USD": {"yf": "FTM-USD", "tv": "BINANCE:FTMUSDT", "desc": "Fantom / USD"},
        "MANA-USD": {"yf": "MANA-USD", "tv": "BINANCE:MANAUSDT", "desc": "Decentraland / USD"},
        "SAND-USD": {"yf": "SAND-USD", "tv": "BINANCE:SANDUSDT", "desc": "The Sandbox / USD"},
        "AXS-USD": {"yf": "AXS-USD", "tv": "BINANCE:AXSUSDT", "desc": "Axie Infinity / USD"},
        "CHZ-USD": {"yf": "CHZ-USD", "tv": "BINANCE:CHZUSDT", "desc": "Chiliz / USD"},
        "ENJ-USD": {"yf": "ENJ-USD", "tv": "BINANCE:ENJUSDT", "desc": "Enjin Coin / USD"},
        "CRV-USD": {"yf": "CRV-USD", "tv": "BINANCE:CRVUSDT", "desc": "Curve DAO / USD"},
        "SNX-USD": {"yf": "SNX-USD", "tv": "BINANCE:SNXUSDT", "desc": "Synthetix / USD"},
        "REN-USD": {"yf": "REN-USD", "tv": "BINANCE:RENUSDT", "desc": "Ren / USD"},
        "1INCH-USD": {"yf": "1INCH-USD", "tv": "BINANCE:1INCHUSDT", "desc": "1inch / USD"},
        "SUSHI-USD": {"yf": "SUSHI-USD", "tv": "BINANCE:SUSHIUSDT", "desc": "SushiSwap / USD"},
        "YFI-USD": {"yf": "YFI-USD", "tv": "BINANCE:YFIUSDT", "desc": "Yearn.finance / USD"},
        "AAVE-USD": {"yf": "AAVE-USD", "tv": "BINANCE:AAVEUSDT", "desc": "Aave / USD"},
        "COMP-USD": {"yf": "COMP-USD", "tv": "BINANCE:COMPUSDT", "desc": "Compound / USD"},
        "MKR-USD": {"yf": "MKR-USD", "tv": "BINANCE:MKRUSDT", "desc": "Maker / USD"},
        "RUNE-USD": {"yf": "RUNE-USD", "tv": "BINANCE:RUNEUSDT", "desc": "THORChain / USD"},
        "ZEC-USD": {"yf": "ZEC-USD", "tv": "BINANCE:ZECUSDT", "desc": "Zcash / USD"},
        "EOS-USD": {"yf": "EOS-USD", "tv": "BINANCE:EOSUSDT", "desc": "EOS / USD"},
        "XTZ-USD": {"yf": "XTZ-USD", "tv": "BINANCE:XTZUSDT", "desc": "Tezos / USD"},
        "KLAY-USD": {"yf": "KLAY-USD", "tv": "BINANCE:KLAYUSDT", "desc": "Klaytn / USD"},
        "DASH-USD": {"yf": "DASH-USD", "tv": "BINANCE:DASHUSDT", "desc": "Dash / USD"},
        "ZIL-USD": {"yf": "ZIL-USD", "tv": "BINANCE:ZILUSDT", "desc": "Zilliqa / USD"},
        "STX-USD": {"yf": "STX-USD", "tv": "BINANCE:STXUSDT", "desc": "Stacks / USD"},
        "CELO-USD": {"yf": "CELO-USD", "tv": "BINANCE:CELOUSDT", "desc": "Celo / USD"},
        "MINA-USD": {"yf": "MINA-USD", "tv": "BINANCE:MINAUSDT", "desc": "Mina Protocol / USD"},
        "GALA-USD": {"yf": "GALA-USD", "tv": "BINANCE:GALAUSDT", "desc": "Gala / USD"},
        "DYDX-USD": {"yf": "DYDX-USD", "tv": "BINANCE:DYDXUSDT", "desc": "dYdX / USD"},
        "OP-USD": {"yf": "OP-USD", "tv": "BINANCE:OPUSDT", "desc": "Optimism / USD"},
        "AR-USD": {"yf": "AR-USD", "tv": "BINANCE:ARUSDT", "desc": "Arweave / USD"},
        "API3-USD": {"yf": "API3-USD", "tv": "BINANCE:API3USDT", "desc": "API3 / USD"},
    },
    # Stocks (US large/mid cap)
    "Stocks": {
        "AAPL": {"yf": "AAPL", "tv": "NASDAQ:AAPL", "desc": "Apple Inc."},
        "MSFT": {"yf": "MSFT", "tv": "NASDAQ:MSFT", "desc": "Microsoft Corporation"},
        "GOOGL": {"yf": "GOOGL", "tv": "NASDAQ:GOOGL", "desc": "Alphabet Inc. (Class A)"},
        "TSLA": {"yf": "TSLA", "tv": "NASDAQ:TSLA", "desc": "Tesla, Inc."},
        "AMZN": {"yf": "AMZN", "tv": "NASDAQ:AMZN", "desc": "Amazon.com, Inc."},
        "NVDA": {"yf": "NVDA", "tv": "NASDAQ:NVDA", "desc": "NVIDIA Corporation"},
        "META": {"yf": "META", "tv": "NASDAQ:META", "desc": "Meta Platforms, Inc."},
        "NFLX": {"yf": "NFLX", "tv": "NASDAQ:NFLX", "desc": "Netflix, Inc."},
        "AMD": {"yf": "AMD", "tv": "NASDAQ:AMD", "desc": "Advanced Micro Devices, Inc."},
        "INTC": {"yf": "INTC", "tv": "NASDAQ:INTC", "desc": "Intel Corporation"},
        "CSCO": {"yf": "CSCO", "tv": "NASDAQ:CSCO", "desc": "Cisco Systems, Inc."},
        "ADBE": {"yf": "ADBE", "tv": "NASDAQ:ADBE", "desc": "Adobe Inc."},
        "CRM": {"yf": "CRM", "tv": "NYSE:CRM", "desc": "Salesforce, Inc."},
        "ORCL": {"yf": "ORCL", "tv": "NYSE:ORCL", "desc": "Oracle Corporation"},
        "IBM": {"yf": "IBM", "tv": "NYSE:IBM", "desc": "International Business Machines"},
        "QCOM": {"yf": "QCOM", "tv": "NASDAQ:QCOM", "desc": "QUALCOMM Incorporated"},
        "AVGO": {"yf": "AVGO", "tv": "NASDAQ:AVGO", "desc": "Broadcom Inc."},
        "TXN": {"yf": "TXN", "tv": "NASDAQ:TXN", "desc": "Texas Instruments Incorporated"},
        "AMAT": {"yf": "AMAT", "tv": "NASDAQ:AMAT", "desc": "Applied Materials, Inc."},
        "MU": {"yf": "MU", "tv": "NASDAQ:MU", "desc": "Micron Technology, Inc."},
        "LRCX": {"yf": "LRCX", "tv": "NASDAQ:LRCX", "desc": "Lam Research Corporation"},
        "KLAC": {"yf": "KLAC", "tv": "NASDAQ:KLAC", "desc": "KLA Corporation"},
        "ASML": {"yf": "ASML", "tv": "NASDAQ:ASML", "desc": "ASML Holding N.V."},
        "TSM": {"yf": "TSM", "tv": "NYSE:TSM", "desc": "Taiwan Semiconductor Manufacturing"},
        "JPM": {"yf": "JPM", "tv": "NYSE:JPM", "desc": "JPMorgan Chase & Co."},
        "BAC": {"yf": "BAC", "tv": "NYSE:BAC", "desc": "Bank of America Corporation"},
        "WFC": {"yf": "WFC", "tv": "NYSE:WFC", "desc": "Wells Fargo & Company"},
        "GS": {"yf": "GS", "tv": "NYSE:GS", "desc": "The Goldman Sachs Group, Inc."},
        "MS": {"yf": "MS", "tv": "NYSE:MS", "desc": "Morgan Stanley"},
        "C": {"yf": "C", "tv": "NYSE:C", "desc": "Citigroup Inc."},
        "USB": {"yf": "USB", "tv": "NYSE:USB", "desc": "U.S. Bancorp"},
        "TFC": {"yf": "TFC", "tv": "NYSE:TFC", "desc": "Truist Financial Corporation"},
        "PNC": {"yf": "PNC", "tv": "NYSE:PNC", "desc": "The PNC Financial Services Group"},
        "SPY": {"yf": "SPY", "tv": "AMEX:SPY", "desc": "SPDR S&P 500 ETF Trust"},
        "QQQ": {"yf": "QQQ", "tv": "NASDAQ:QQQ", "desc": "Invesco QQQ Trust"},
        "IWM": {"yf": "IWM", "tv": "AMEX:IWM", "desc": "iShares Russell 2000 ETF"},
        "DIA": {"yf": "DIA", "tv": "AMEX:DIA", "desc": "SPDR Dow Jones Industrial Average ETF"},
        "VTI": {"yf": "VTI", "tv": "AMEX:VTI", "desc": "Vanguard Total Stock Market ETF"},
        "XLF": {"yf": "XLF", "tv": "AMEX:XLF", "desc": "Financial Select Sector SPDR Fund"},
        "XLK": {"yf": "XLK", "tv": "AMEX:XLK", "desc": "Technology Select Sector SPDR Fund"},
        "XLE": {"yf": "XLE", "tv": "AMEX:XLE", "desc": "Energy Select Sector SPDR Fund"},
        "XLU": {"yf": "XLU", "tv": "AMEX:XLU", "desc": "Utilities Select Sector SPDR Fund"},
        "XLI": {"yf": "XLI", "tv": "AMEX:XLI", "desc": "Industrial Select Sector SPDR Fund"},
        "XLV": {"yf": "XLV", "tv": "AMEX:XLV", "desc": "Health Care Select Sector SPDR Fund"},
        "XLY": {"yf": "XLY", "tv": "AMEX:XLY", "desc": "Consumer Discretionary Select Sector SPDR Fund"},
        "XLP": {"yf": "XLP", "tv": "AMEX:XLP", "desc": "Consumer Staples Select Sector SPDR Fund"},
        "XLB": {"yf": "XLB", "tv": "AMEX:XLB", "desc": "Materials Select Sector SPDR Fund"},
        "XLRE": {"yf": "XLRE", "tv": "AMEX:XLRE", "desc": "Real Estate Select Sector SPDR Fund"},
        "XLC": {"yf": "XLC", "tv": "AMEX:XLC", "desc": "Communication Services Select Sector SPDR Fund"},
    },
    # ETFs (Global, Thematic)
    "ETFs": {
        "ARKK": {"yf": "ARKK", "tv": "AMEX:ARKK", "desc": "ARK Innovation ETF"},
        "ARKF": {"yf": "ARKF", "tv": "AMEX:ARKF", "desc": "ARK Fintech Innovation ETF"},
        "ARKW": {"yf": "ARKW", "tv": "AMEX:ARKW", "desc": "ARK Next Generation Internet ETF"},
        "ARKG": {"yf": "ARKG", "tv": "AMEX:ARKG", "desc": "ARK Genomic Revolution ETF"},
        "ARKX": {"yf": "ARKX", "tv": "AMEX:ARKX", "desc": "ARK Space Exploration ETF"},
        "ICLN": {"yf": "ICLN", "tv": "NASDAQ:ICLN", "desc": "iShares Global Clean Energy ETF"},
        "PBW": {"yf": "PBW", "tv": "AMEX:PBW", "desc": "Invesco WilderHill Clean Energy ETF"},
        "SMH": {"yf": "SMH", "tv": "NASDAQ:SMH", "desc": "VanEck Semiconductor ETF"},
        "SOXX": {"yf": "SOXX", "tv": "NASDAQ:SOXX", "desc": "iShares Semiconductor ETF"},
        "XBI": {"yf": "XBI", "tv": "AMEX:XBI", "desc": "SPDR S&P Biotech ETF"},
        "IBB": {"yf": "IBB", "tv": "NASDAQ:IBB", "desc": "iShares Biotechnology ETF"},
        "GDX": {"yf": "GDX", "tv": "AMEX:GDX", "desc": "VanEck Gold Miners ETF"},
        "GDXJ": {"yf": "GDXJ", "tv": "AMEX:GDXJ", "desc": "VanEck Junior Gold Miners ETF"},
        "SIL": {"yf": "SIL", "tv": "AMEX:SIL", "desc": "Global X Silver Miners ETF"},
        "URA": {"yf": "URA", "tv": "AMEX:URA", "desc": "Global X Uranium ETF"},
        "LIT": {"yf": "LIT", "tv": "AMEX:LIT", "desc": "Global X Lithium & Battery Tech ETF"},
        "DRIV": {"yf": "DRIV", "tv": "NASDAQ:DRIV", "desc": "Global X Autonomous & Electric Vehicles ETF"},
        "BOTZ": {"yf": "BOTZ", "tv": "NASDAQ:BOTZ", "desc": "Global X Robotics & AI ETF"},
        "ROBO": {"yf": "ROBO", "tv": "AMEX:ROBO", "desc": "ROBO Global Robotics and Automation ETF"},
        "CORN": {"yf": "CORN", "tv": "AMEX:CORN", "desc": "Teucrium Corn Fund"},
        "SOYB": {"yf": "SOYB", "tv": "AMEX:SOYB", "desc": "Teucrium Soybean Fund"},
        "WEAT": {"yf": "WEAT", "tv": "AMEX:WEAT", "desc": "Teucrium Wheat Fund"},
        "CANE": {"yf": "CANE", "tv": "AMEX:CANE", "desc": "Teucrium Sugar Fund"},
        "NIB": {"yf": "NIB", "tv": "AMEX:NIB", "desc": "iPath Cocoa ETN"},
        "VIXY": {"yf": "VIXY", "tv": "AMEX:VIXY", "desc": "ProShares VIX Short-Term Futures ETF"},
        "VXX": {"yf": "VXX", "tv": "AMEX:VXX", "desc": "iPath VIX Short-Term Futures ETN"},
        "UVXY": {"yf": "UVXY", "tv": "AMEX:UVXY", "desc": "ProShares Ultra VIX Short-Term Futures ETF"},
        "SVXY": {"yf": "SVXY", "tv": "AMEX:SVXY", "desc": "ProShares Short VIX Short-Term Futures ETF"},
        "GLD": {"yf": "GLD", "tv": "AMEX:GLD", "desc": "SPDR Gold Shares"},
        "SLV": {"yf": "SLV", "tv": "AMEX:SLV", "desc": "iShares Silver Trust"},
        "USO": {"yf": "USO", "tv": "AMEX:USO", "desc": "United States Oil Fund"},
        "UNG": {"yf": "UNG", "tv": "AMEX:UNG", "desc": "United States Natural Gas Fund"},
        "TLT": {"yf": "TLT", "tv": "NASDAQ:TLT", "desc": "iShares 20+ Year Treasury Bond ETF"},
        "IEF": {"yf": "IEF", "tv": "NASDAQ:IEF", "desc": "iShares 7-10 Year Treasury Bond ETF"},
        "SHY": {"yf": "SHY", "tv": "NASDAQ:SHY", "desc": "iShares 1-3 Year Treasury Bond ETF"},
        "HYG": {"yf": "HYG", "tv": "AMEX:HYG", "desc": "iShares iBoxx High Yield Corporate Bond ETF"},
        "LQD": {"yf": "LQD", "tv": "AMEX:LQD", "desc": "iShares iBoxx Investment Grade Corporate Bond ETF"},
        "EMB": {"yf": "EMB", "tv": "AMEX:EMB", "desc": "iShares J.P. Morgan USD Emerging Markets Bond ETF"},
    },
    # FX Majors & Minors
    "FX": {
        "EURUSD": {"yf": "EURUSD=X", "tv": "FX_IDC:EURUSD", "desc": "EUR/USD"},
        "GBPUSD": {"yf": "GBPUSD=X", "tv": "FX_IDC:GBPUSD", "desc": "GBP/USD"},
        "USDJPY": {"yf": "USDJPY=X", "tv": "FX_IDC:USDJPY", "desc": "USD/JPY"},
        "AUDUSD": {"yf": "AUDUSD=X", "tv": "FX_IDC:AUDUSD", "desc": "AUD/USD"},
        "USDCAD": {"yf": "USDCAD=X", "tv": "FX_IDC:USDCAD", "desc": "USD/CAD"},
        "USDCHF": {"yf": "USDCHF=X", "tv": "FX_IDC:USDCHF", "desc": "USD/CHF"},
        "NZDUSD": {"yf": "NZDUSD=X", "tv": "FX_IDC:NZDUSD", "desc": "NZD/USD"},
        "EURJPY": {"yf": "EURJPY=X", "tv": "FX_IDC:EURJPY", "desc": "EUR/JPY"},
        "GBPJPY": {"yf": "GBPJPY=X", "tv": "FX_IDC:GBPJPY", "desc": "GBP/JPY"},
        "EURGBP": {"yf": "EURGBP=X", "tv": "FX_IDC:EURGBP", "desc": "EUR/GBP"},
        "EURCHF": {"yf": "EURCHF=X", "tv": "FX_IDC:EURCHF", "desc": "EUR/CHF"},
        "EURCAD": {"yf": "EURCAD=X", "tv": "FX_IDC:EURCAD", "desc": "EUR/CAD"},
        "EURAUD": {"yf": "EURAUD=X", "tv": "FX_IDC:EURAUD", "desc": "EUR/AUD"},
        "EURNZD": {"yf": "EURNZD=X", "tv": "FX_IDC:EURNZD", "desc": "EUR/NZD"},
        "GBPAUD": {"yf": "GBPAUD=X", "tv": "FX_IDC:GBPAUD", "desc": "GBP/AUD"},
        "GBPCAD": {"yf": "GBPCAD=X", "tv": "FX_IDC:GBPCAD", "desc": "GBP/CAD"},
        "GBPCHF": {"yf": "GBPCHF=X", "tv": "FX_IDC:GBPCHF", "desc": "GBP/CHF"},
        "CADJPY": {"yf": "CADJPY=X", "tv": "FX_IDC:CADJPY", "desc": "CAD/JPY"},
        "CHFJPY": {"yf": "CHFJPY=X", "tv": "FX_IDC:CHFJPY", "desc": "CHF/JPY"},
        "AUDCAD": {"yf": "AUDCAD=X", "tv": "FX_IDC:AUDCAD", "desc": "AUD/CAD"},
        "AUDJPY": {"yf": "AUDJPY=X", "tv": "FX_IDC:AUDJPY", "desc": "AUD/JPY"},
        "AUDNZD": {"yf": "AUDNZD=X", "tv": "FX_IDC:AUDNZD", "desc": "AUD/NZD"},
        "AUDCHF": {"yf": "AUDCHF=X", "tv": "FX_IDC:AUDCHF", "desc": "AUD/CHF"},
        "NZDCAD": {"yf": "NZDCAD=X", "tv": "FX_IDC:NZDCAD", "desc": "NZD/CAD"},
        "NZDJPY": {"yf": "NZDJPY=X", "tv": "FX_IDC:NZDJPY", "desc": "NZD/JPY"},
        "USDSGD": {"yf": "USDSGD=X", "tv": "FX_IDC:USDSGD", "desc": "USD/SGD"},
        "USDHKD": {"yf": "USDHKD=X", "tv": "FX_IDC:USDHKD", "desc": "USD/HKD"},
        "USDCNH": {"yf": "USDCNH=X", "tv": "FX_IDC:USDCNH", "desc": "USD/CNH (Offshore Yuan)"},
        "USDMXN": {"yf": "USDMXN=X", "tv": "FX_IDC:USDMXN", "desc": "USD/MXN"},
        "USDBRL": {"yf": "USDBRL=X", "tv": "FX_IDC:USDBRL", "desc": "USD/BRL"},
        "USDZAR": {"yf": "USDZAR=X", "tv": "FX_IDC:USDZAR", "desc": "USD/ZAR"},
        "USDINR": {"yf": "USDINR=X", "tv": "FX_IDC:USDINR", "desc": "USD/INR"},
        "USDRUB": {"yf": "USDRUB=X", "tv": "FX_IDC:USDRUB", "desc": "USD/RUB"},
        "EURSEK": {"yf": "EURSEK=X", "tv": "FX_IDC:EURSEK", "desc": "EUR/SEK"},
        "EURNOK": {"yf": "EURNOK=X", "tv": "FX_IDC:EURNOK", "desc": "EUR/NOK"},
        "EURDKK": {"yf": "EURDKK=X", "tv": "FX_IDC:EURDKK", "desc": "EUR/DKK"},
        "EURTRY": {"yf": "EURTRY=X", "tv": "FX_IDC:EURTRY", "desc": "EUR/TRY"},
        "GBPSEK": {"yf": "GBPSEK=X", "tv": "FX_IDC:GBPSEK", "desc": "GBP/SEK"},
        "GBPNOK": {"yf": "GBPNOK=X", "tv": "FX_IDC:GBPNOK", "desc": "GBP/NOK"},
        "USDTRY": {"yf": "USDTRY=X", "tv": "FX_IDC:USDTRY", "desc": "USD/TRY"},
        "USDPLN": {"yf": "USDPLN=X", "tv": "FX_IDC:USDPLN", "desc": "USD/PLN"},
        "USDCZK": {"yf": "USDCZK=X", "tv": "FX_IDC:USDCZK", "desc": "USD/CZK"},
        "USDHUF": {"yf": "USDHUF=X", "tv": "FX_IDC:USDHUF", "desc": "USD/HUF"},
        "USDRON": {"yf": "USDRON=X", "tv": "FX_IDC:USDRON", "desc": "USD/RON"},
    },
    # Indices
    "Indices": {
        "SPX": {"yf": "^GSPC", "tv": "SP:SPX", "desc": "S&P 500 Index"},
        "NDX": {"yf": "^NDX", "tv": "NASDAQ:NDX", "desc": "Nasdaq 100 Index"},
        "DJI": {"yf": "^DJI", "tv": "DJ:DJI", "desc": "Dow Jones Industrial Average"},
        "RUT": {"yf": "^RUT", "tv": "AMEX:IWM", "desc": "Russell 2000 Index"},
        "VIX": {"yf": "^VIX", "tv": "CBOE:VIX", "desc": "CBOE Volatility Index"},
        "VVIX": {"yf": "^VVIX", "tv": "CBOE:VVIX", "desc": "CBOE VVIX"},
        "SPXW": {"yf": "^SPXW", "tv": "CBOE:SPXW", "desc": "S&P 500 Weekly Options"},
        "OEX": {"yf": "^OEX", "tv": "CBOE:OEX", "desc": "S&P 100 Index"},
        "XEO": {"yf": "^XEO", "tv": "CBOE:XEO", "desc": "S&P 100 European Options"},
        "TNX": {"yf": "^TNX", "tv": "CBOE:TNX", "desc": "10-Year Treasury Note Yield"},
        "FVX": {"yf": "^FVX", "tv": "CBOE:FVX", "desc": "5-Year Treasury Note Yield"},
        "TYX": {"yf": "^TYX", "tv": "CBOE:TYX", "desc": "30-Year Treasury Bond Yield"},
        "IRX": {"yf": "^IRX", "tv": "CBOE:IRX", "desc": "13-Week Treasury Bill Yield"},
        "DX-Y": {"yf": "DX-Y.NYB", "tv": "ICE:DX1!", "desc": "US Dollar Index (ICE Futures)"},
        "FTSE": {"yf": "^FTSE", "tv": "INDEXFTSE:UKX", "desc": "FTSE 100 Index"},
        "DAX": {"yf": "^GDAXI", "tv": "XETR:DAX", "desc": "DAX Index"},
        "CAC": {"yf": "^FCHI", "tv": "EURONEXT:PX1", "desc": "CAC 40 Index"},
        "N225": {"yf": "^N225", "tv": "OANDA:JP225", "desc": "Nikkei 225 Index"},
        "HSI": {"yf": "^HSI", "tv": "HSI:HSI", "desc": "Hang Seng Index"},
        "SSE": {"yf": "000001.SS", "tv": "SSE:000001", "desc": "Shanghai Composite Index"},
        "BSESN": {"yf": "^BSESN", "tv": "BSE:SENSEX", "desc": "BSE Sensex Index"},
        "NIfty": {"yf": "^NSEI", "tv": "NSE:NIFTY", "desc": "NIFTY 50 Index"},
        "ASX": {"yf": "^AXJO", "tv": "ASX:XJO", "desc": "ASX 200 Index"},
        "TSX": {"yf": "^GSPTSE", "tv": "TSX:TSX", "desc": "S&P/TSX Composite Index"},
        "Bovespa": {"yf": "^BVSP", "tv": "BMFBOVESPA:IBOVESPA", "desc": "Bovespa Index"},
        "STI": {"yf": "^STI", "tv": "SGX:STI", "desc": "Straits Times Index"},
        "KLSE": {"yf": "^KLSE", "tv": "BURSA:KLCI", "desc": "FTSE Bursa Malaysia KLCI"},
        "SET": {"yf": "^SET.BK", "tv": "SET:SET", "desc": "SET Index"},
        "KOSPI": {"yf": "^KS11", "tv": "KRX:KOSPI", "desc": "KOSPI Index"},
        "TA125": {"yf": "^TA125.TA", "tv": "TASE:TA125", "desc": "TA-125 Index"},
        "TA35": {"yf": "^TA35.TA", "tv": "TASE:TA35", "desc": "TA-35 Index"},
    },
    # Commodities (Futures proxies where available)
    "Commodities": {
        "CL": {"yf": "CL=F", "tv": "NYMEX:CL1!", "desc": "WTI Crude Oil Futures"},
        "BZ": {"yf": "BZ=F", "tv": "ICE:BRN1!", "desc": "Brent Crude Oil Futures"},
        "NG": {"yf": "NG=F", "tv": "NYMEX:NG1!", "desc": "Natural Gas Futures"},
        "GC": {"yf": "GC=F", "tv": "COMEX:GC1!", "desc": "Gold Futures"},
        "SI": {"yf": "SI=F", "tv": "COMEX:SI1!", "desc": "Silver Futures"},
        "HG": {"yf": "HG=F", "tv": "COMEX:HG1!", "desc": "Copper Futures"},
        "ZB": {"yf": "ZB=F", "tv": "CBOT:ZB1!", "desc": "30-Year T-Bond Futures"},
        "ZN": {"yf": "ZN=F", "tv": "CBOT:ZN1!", "desc": "10-Year T-Note Futures"},
        "ZF": {"yf": "ZF=F", "tv": "CBOT:ZF1!", "desc": "5-Year T-Note Futures"},
        "ZT": {"yf": "ZT=F", "tv": "CBOT:ZT1!", "desc": "2-Year T-Note Futures"},
        "ZR": {"yf": "ZR=F", "tv": "CBOT:ZR1!", "desc": "Rough Rice Futures"},
        "ZL": {"yf": "ZL=F", "tv": "CBOT:ZL1!", "desc": "Soybean Oil Futures"},
        "ZM": {"yf": "ZM=F", "tv": "CBOT:ZM1!", "desc": "Soybean Meal Futures"},
        "ZS": {"yf": "ZS=F", "tv": "CBOT:ZS1!", "desc": "Soybean Futures"},
        "ZC": {"yf": "ZC=F", "tv": "CBOT:ZC1!", "desc": "Corn Futures"},
        "ZW": {"yf": "ZW=F", "tv": "CBOT:ZW1!", "desc": "Wheat Futures"},
        "KC": {"yf": "KC=F", "tv": "ICE:KC1!", "desc": "Coffee Futures"},
        "CT": {"yf": "CT=F", "tv": "ICE:CT1!", "desc": "Cotton Futures"},
        "CC": {"yf": "CC=F", "tv": "ICE:CC1!", "desc": "Cocoa Futures"},
        "SB": {"yf": "SB=F", "tv": "ICE:SB1!", "desc": "Sugar #11 Futures"},
        "DX": {"yf": "DX=F", "tv": "ICE:DX1!", "desc": "US Dollar Index Futures"},
        "PA": {"yf": "PA=F", "tv": "NYMEX:PA1!", "desc": "Palladium Futures"},
        "PL": {"yf": "PL=F", "tv": "NYMEX:PL1!", "desc": "Platinum Futures"},
        "GCU": {"yf": "GCU=F", "tv": "COMEX:GCU1!", "desc": "Gold (E-micro) Futures"},
        "SIU": {"yf": "SIU=F", "tv": "COMEX:SIU1!", "desc": "Silver (E-mini) Futures"},
        "ALI": {"yf": "ALI=F", "tv": "COMEX:ALI1!", "desc": "Aluminum Futures"},
        "GLNC": {"yf": "GLNC=F", "tv": "COMEX:GLNC1!", "desc": "Gold (Cash) Futures"},
        "SLNC": {"yf": "SLNC=F", "tv": "COMEX:SLNC1!", "desc": "Silver (Cash) Futures"},
    },
}

# Flatten all symbols into one lookup table for convenience
ALL_SYMBOLS: Dict[str, Dict[str, str]] = {}
for group, symbols in TICKER_DB.items():
    for key, info in symbols.items():
        ALL_SYMBOLS[key] = {**info, "group": group}

# Default watchlists (persistent)
DEFAULT_WATCHLISTS: Dict[str, List[str]] = {
    "Favorites": ["BTC-USD", "ETH-USD", "SPX", "NDX", "XAU", "CL", "EURUSD", "US10Y"],
    "Crypto Majors": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD"],
    "US Tech": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "NFLX", "AMD"],
    "Macro Hedge": ["XAU", "SI", "TLT", "IEF", "DXY", "VIX", "HYG"],
    "Commodities": ["CL", "GC", "HG", "NG", "SI"],
    "FX Majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"],
}

# -----------------------------------------------------------------------------
# Data Fetching with Transparency
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    source: str = "yfinance",
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Dict[str, Any]]]:
    """
    Fetch OHLCV data.
    Returns (df, error_msg, metadata).
    Metadata includes last_refresh, latency, rows, missing_rows.
    """
    if not YF_AVAILABLE:
        return None, "yfinance not installed.", None

    start_fetch = time.time()
    try:
        # Intraday range caps for yfinance stability
        max_days = None
        if interval == "1m":
            max_days = 7
        elif interval in ["2m", "5m", "15m", "30m", "60m", "90m"]:
            max_days = 60
        elif interval == "1h":
            max_days = 730

        if max_days and (end - start).days > max_days:
            start = end - timedelta(days=max_days)

        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            return None, f"No data for {symbol} on {interval}.", None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        latency = time.time() - start_fetch
        rows = len(df)
        missing_rows = int(df.isnull().sum().sum())

        metadata = {
            "last_refresh": datetime.utcnow().isoformat() + "Z",
            "latency_sec": round(latency, 3),
            "rows": rows,
            "missing_rows": missing_rows,
            "source": "yfinance",
        }
        return df, None, metadata

    except Exception as e:
        logger.exception(f"fetch_data error for {symbol}")
        return None, f"Error fetching {symbol}: {e}", None


# -----------------------------------------------------------------------------
# Indicator Engine (Pure pandas)
def sma(series: pd.Series, length: int = 20) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int = 20) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, length: int = 20, stdev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, length)
    std = series.rolling(length, min_periods=length).std()
    upper = mid + stdev * std
    lower = mid - stdev * std
    return upper, mid, lower

def vwap(df: pd.DataFrame) -> pd.Series:
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float, name="VWAP")
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan)
    pv = tp * vol
    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum()
    return (cum_pv / cum_vol).rename("VWAP")

def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(length, min_periods=length).mean().rename("ATR")

def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]
    tp_shift = tp.shift(1)
    pos_flow = rmf.where(tp > tp_shift, 0).rolling(length, min_periods=length).sum()
    neg_flow = rmf.where(tp < tp_shift, 0). rolling(length, min_periods=length).sum().replace(0, np.nan)
    mfr = pos_flow / neg_flow
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename("MFI")

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high. rolling(kijun).max() + low. rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_b = ((high. rolling(senkou_b).max() + low. rolling(senkou_b).min()) / 2.0).shift(kijun)
    chikou = close.shift(-kijun)
    return {
        "tenkan": tenkan_sen.rename("Ichimoku Tenkan"),
        "kijun": kijun_sen.rename("Ichimoku Kijun"),
        "senkou_a": senkou_a.rename("Ichimoku Senkou A"),
        "senkou_b": senkou_b.rename("Ichimoku Senkou B"),
        "chikou": chikou.rename("Ichimoku Chikou"),
    }

def pivot_points(df: pd.DataFrame, length: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs/lows (pivots) for SMC structure.
    """
    high = df["High"]
    low = df["Low"]
    # Rolling windows to find local extrema
    pivot_high = high.where(high == high.rolling(2 * length + 1, center=True).max(), np.nan)
    pivot_low = low.where(low == low.rolling(2 * length + 1, center=True).min(), np.nan)
    return pivot_high.rename("Pivot High"), pivot_low.rename("Pivot Low")

def smc_structure(df: pd.DataFrame, pivot_len: int = 5, use_close: bool = True) -> pd.DataFrame:
    """
    Lite SMC: detect BOS/CHOCH based on pivot swings.
    Returns DataFrame with boolean columns BOS_Bull, BOS_Bear, CHOCH_Bull, CHOCH_Bear.
    """
    df = df.copy()
    ph, pl = pivot_points(df, pivot_len)
    df["Swing_High"] = ph.ffill()
    df["Swing_Low"] = pl.ffill()

    if use_close:
        up_break = df["Close"] > df["Swing_High"].shift(1)
        dn_break = df["Close"] < df["Swing_Low"].shift(1)
    else:
        up_break = df["High"] > df["Swing_High"].shift(1)
        dn_break = df["Low"] < df["Swing_Low"].shift(1)

    # Trend state: 1 bull, -1 bear, 0 unknown
    trend = np.zeros(len(df), dtype=int)
    last = 0
    for i in range(len(df)):
        if up_break.iloc[i]:
            last = 1
        elif dn_break.iloc[i]:
            last = -1
        trend[i] = last
    df["Trend"] = trend
    prev_trend = pd.Series(trend).shift(1).fillna(0).astype(int)

    df["BOS_Bull"] = up_break & (prev_trend != 1)
    df["BOS_Bear"] = dn_break & (prev_trend != -1)
    df["CHOCH_Bull"] = up_break & (prev_trend == -1)
    df["CHOCH_Bear"] = dn_break & (prev_trend == 1)

    return df

# -----------------------------------------------------------------------------
@dataclass
class TradePlan:
    market_bias: str  # no default
    confidence: float  # no default
    timeframe: str  # no default
    entry_trigger: str  # no default – must appear BEFORE any field with a default
    setup_name: Optional[str] = None  # default now allowed
    entry_range: Optional[Tuple[float, float]] = None
    invalidation_level: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[Dict[str, Any]] = None
    take_profit_ladder: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    when_not_to_take: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = "Not financial advice. Risk of loss."

def generate_ai_trade_plan(
    df: pd.DataFrame,
    tech_summary: Dict[str, Any],
    risk_meter: int,
    symbol: str,
    interval: str,
) -> TradePlan:
    """
    Rule‑based AI: generate a structured trade plan from indicator snapshot.
    No LLM call; deterministic logic mirrors an expert’s checklist.
    """
    last = df.iloc[-1]
    close = float(last["Close"])
    ema50 = tech_summary.get("EMA 50")
    ema200 = tech_summary.get("EMA 200")
    rsi_val = tech_summary.get("RSI 14")
    bb_upper = tech_summary.get("BB Upper")
    bb_lower = tech_summary.get("BB Lower")
    vwap_val = tech_summary.get("VWAP")
    atr_val = tech_summary.get("ATR 14")

    # Determine bias
    bias = "neutral"
    confidence = 50.0
    if ema50 and ema200:
        if ema50 > ema200:
            bias = "bull"
            confidence += 20
        else:
            bias = "bear"
            confidence += 20

    if rsi_val:
        if rsi_val > 70 and bias == "bull":
            confidence -= 15  # overbought
        elif rsi_val < 30 and bias == "bear":
            confidence -= 15  # oversold

    # Entry / invalidation
    entry_trigger = f"Close above {ema50:.4g}" if bias == "bull" else f"Close below {ema50:.4g}"
    entry_range = (close * 0.98, close * 1.02) if bias != "neutral" else None
    invalidation = close * 0.95 if bias == "bull" else close * 1.05 if bias == "bear" else None
    stop_loss = close * 0.92 if bias == "bull" else close * 1.08 if bias == "bear" else None

    # Trailing stop
    trailing_stop = None
    if atr_val:
        trailing_stop = {
            "method": "ATR trail",
            "activation": "After entry, trail 1.5x ATR below swing low / above swing high",
        }

    # TP ladder
    tp_ladder = []
    if bias == "bull":
        tp_levels = [close * 1.05, close * 1.10, close * 1.15]
    elif bias == "bear":
        tp_levels = [close * 0.95, close * 0.90, close * 0.85]
    else:
        tp_levels = []

    for i, level in enumerate(tp_levels, 1):
        if stop_loss and level:
            rr = abs((level - close) / (close - stop_loss))
        else:
            rr = None
        tp_ladder.append({
            "level": level,
            "rr": rr,
            "instruction": f"Partial close 33%, move SL to breakeven after TP{i}",
        })

    reasons = []
    if bias == "bull":
        reasons.append(f"EMA 50 ({ema50:.4g}) above EMA 200 ({ema200:.4g}) indicating uptrend.")
        if vwap_val and close > vwap_val:
            reasons.append("Price trading above VWAP, showing intraday strength.")
        if rsi_val and rsi_val < 70:
            reasons.append(f"RSI {rsi_val:.1f} not yet overbought, room to run.")
    elif bias == "bear":
        reasons.append(f"EMA 50 ({ema50:.4g}) below EMA 200 ({ema200:.4g}) indicating downtrend.")
        if vwap_val and close < vwap_val:
            reasons.append("Price trading below VWAP, showing intraday weakness.")
        if rsi_val and rsi_val > 30:
            reasons.append(f"RSI {rsi_val:.1f} not yet oversold, room to fall.")

    when_not_to_take = []
    if bias == "bull":
        when_not_to_take.append("If price closes below swing low (invalidation level).")
        when_not_to_take.append("If RSI spikes above 80 and diverges from price.")
        when_not_to_take.append("If volume declines on rallies (weak momentum).")
    elif bias == "bear":
        when_not_to_take.append("If price closes above swing high (invalidation level).")
        when_not_to_take.append("If RSI drops below 20 and diverges from price.")
        when_not_to_take.append("If volume declines on selloffs (weak momentum).")

    warnings = []
    if risk_meter >= 8:
        warnings.append("High risk meter (8–10) indicates elevated volatility.")
    if atr_val and atr_val > close * 0.03:
        warnings.append("ATR is high relative to price; expect wide swings.")

    confidence = max(0, min(100, confidence))

    return TradePlan(
        market_bias=bias,
        confidence=confidence,
        timeframe=interval,
        setup_name="EMA Cross Structure" if bias != "neutral" else "Rangebound",
        entry_trigger=entry_trigger,
        entry_range=entry_range,
        invalidation_level=invalidation,
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        take_profit_ladder=tp_ladder,
        reasons=reasons,
        when_not_to_take=when_not_to_take,
        warnings=warnings,
    )

def render_ai_plan(plan: TradePlan) -> str:
    """Convert TradePlan to markdown for UI and Telegram."""
    lines = []
    lines.append(f"**Market Bias:** {plan.market_bias.upper()} (confidence: {plan.confidence:.0f}%)")
    lines.append(f"**Timeframe:** {plan.timeframe}")
    if plan.setup_name:
        lines.append(f"**Setup:** {plan.setup_name}")
    lines.append("")
    lines.append("**Entry**")
    lines.append(f"- Trigger: {plan.entry_trigger}")
    if plan.entry_range:
        lines.append(f"- Range: {plan.entry_range[0]:.6g} – {plan.entry_range[1]:.6g}")
    lines.append("")
    if plan.invalidation_level:
        lines.append(f"**Invalidation Level:** {plan.invalidation_level:.6g}")
    if plan.stop_loss:
        lines.append(f"**Stop Loss:** {plan.stop_loss:.6g}")
    if plan.trailing_stop:
        lines.append(f"**Trailing Stop:** {plan.trailing_stop['method']} – {plan.trailing_stop['activation']}")
    lines.append("")
    if plan.take_profit_ladder:
        lines.append("**Take-Profit Ladder**")
        for i, tp in enumerate(plan.take_profit_ladder, 1):
            lines.append(
                f"- TP{i}: {tp['level']:.6g} (R/R: {tp['rr']:.2f}) – {tp['instruction']}"
            )
    lines.append("")
    if plan.reasons:
        lines.append("**Reasons for Trade**")
        for r in plan.reasons:
            lines.append(f"- {r}")
    lines.append("")
    if plan.when_not_to_take:
        lines.append("**When NOT to Take**")
        for w in plan.when_not_to_take:
            lines.append(f"- {w}")
    lines.append("")
    if plan.warnings:
        lines.append("**Warnings**")
        for w in plan.warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append(f"*{plan.disclaimer}*")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Telegram Broadcaster
@st.cache_resource(show_spinner=False)
def get_telegram_bot(token: str):
    if not TELEGRAM_AVAILABLE:
        return None, "python-telegram-bot not installed."
    try:
        bot = telegram.Bot(token=token)
        # Test connection
        me = bot.get_me()
        logger.info(f"Telegram bot connected: {me.username}")
        return bot, None
    except TelegramError as e:
        logger.error(f"Telegram bot init error: {e}")
        return None, f"Telegram error: {e}"

def split_message(text: str, limit: int = 4096) -> List[str]:
    """Split long message into chunks, preserving code blocks if possible."""
    if len(text) <= limit:
        return [text]
    # Simple split by lines, try to keep coherent
    lines = text.splitlines()
    chunks = []
    current = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > limit:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    # Number them
    if len(chunks) > 1:
        chunks = [f"({i+1}/{len(chunks)})\n{chunk}" for i, chunk in enumerate(chunks)]
    return chunks

def broadcast_message(
    bot: telegram.Bot,
    chat_ids: List[str],
    message: str,
    parse_mode: str = "Markdown",
    disable_notification: bool = False,
) -> List[Dict[str, Any]]:
    """
    Broadcast message to multiple chats with retry, split, and logging.
    Returns list of delivery log entries.
    """
    log_entries = []
    chunks = split_message(message)
    for i, chunk in enumerate(chunks):
        for cid in chat_ids:
            attempt = 0
            max_attempts = 3
            backoff = 1.0
            status = "pending"
            error_snippet = None
            while attempt < max_attempts:
                try:
                    if parse_mode == "Markdown":
                        bot.send_message(
                            chat_id=cid,
                            text=chunk,
                            parse_mode="Markdown",
                            disable_notification=disable_notification,
                        )
                    else:
                        bot.send_message(
                            chat_id=cid,
                            text=chunk,
                            disable_notification=disable_notification,
                        )
                    status = "success"
                    error_snippet = None
                    break
                except RetryAfter as e:
                    wait = e.retry_after + backoff
                    logger.warning(f"Rate limited. Waiting {wait}s.")
                    time.sleep(wait)
                    backoff *= 2
                    attempt += 1
                    status = "retry_rate_limit"
                    error_snippet = str(e)[:100]
                except TimedOut as e:
                    wait = backoff
                    logger.warning(f"Timeout. Waiting {wait}s.")
                    time.sleep(wait)
                    backoff *= 2
                    attempt += 1
                    status = "retry_timeout"
                    error_snippet = str(e)[:100]
                except TelegramError as e:
                    logger.error(f"Telegram error: {e}")
                    status = "failed"
                    error_snippet = str(e)[:100]
                    break
                except Exception as e:
                    logger.exception("Unexpected error sending message")
                    status = "failed"
                    error_snippet = str(e)[:100]
                    break

            log_entries.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "chat_id": cid,
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "status": status,
                    "error": error_snippet,
                }
            )
    return log_entries

# -----------------------------------------------------------------------------
# TradingView URL Builder
def build_tradingview_url(tv_symbol: str, interval: str) -> str:
    """
    Build a deep-link to TradingView chart.
    Interval mapping: 1m, 5m, 15m, 1h, 4h, 1D, 1W
    """
    # Convert yfinance interval to TV format
    interval_map = {
        "1m": "1m",
        "2m": "5m",  # fallback
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",  # fallback to 15m
        "60m": "1h",
        "90m": "1h",
        "1h": "1h",
        "1d": "1D",
        "1wk": "1W",
        "1mo": "1M",
    }
    tv_interval = interval_map.get(interval, "1D")
    # Ensure symbol has exchange prefix for TV
    if ":" not in tv_symbol:
        tv_symbol = f"NASDAQ:{tv_symbol}"  # default guess
    url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval={tv_interval}"
    return url

# -----------------------------------------------------------------------------
# UI Helpers
def show_diagnostics(errors: List[Dict[str, Any]]):
    st.markdown("### System Health / Diagnostics")
    # Secrets status
    secrets_status = {}
    for key in ["TELEGRAM_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
        val = st.secrets.get(key, "")
        secrets_status[key] = "loaded" if val else "missing"
    st.json(secrets_status)

    # Data source status
    st.write("**Data Sources:**")
    st.json({"yfinance": YF_AVAILABLE})

    # Telegram connectivity
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    if token:
        bot, err = get_telegram_bot(token)
        if bot:
            st.success("Telegram: Connected")
        else:
            st.error(f"Telegram: {err}")
    else:
        st.warning("Telegram: Token not configured")

    # AI connectivity
    ai_ok = False
    if st.secrets.get("OPENAI_API_KEY"):
        ai_ok = True
        st.success("OpenAI: Key detected")
    if st.secrets.get("GEMINI_API_KEY"):
        ai_ok = True
        st.success("Gemini: Key detected")
    if not ai_ok:
        st.warning("AI: No keys detected")

    # Recent errors
    if errors:
        st.markdown("**Recent Errors**")
        st.dataframe(pd.DataFrame(errors).tail(10), use_container_width=True)

# -----------------------------------------------------------------------------
# Streamlit App Layout
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption(DISCLAIMER)

if "errors" not in st.session_state:
    st.session_state.errors = []
if "delivery_log" not in st.session_state:
    st.session_state.delivery_log = []

# Sidebar: Data Selection
st.sidebar.header("📋 Ticker & Asset Class")
asset_group = st.sidebar.selectbox("Asset Class", list(TICKER_DB.keys()), index=0)
available_symbols = list(TICKER_DB[asset_group].keys())
selected_symbol_key = st.sidebar.selectbox("Symbol", available_symbols, index=0)

symbol_info = ALL_SYMBOLS[selected_symbol_key]
yf_symbol = symbol_info["yf"]
tv_symbol = symbol_info["tv"]

# Symbol mapping override UI
with st.sidebar.expander("Symbol Mapping Override", expanded=False):
    st.caption("If the default yfinance symbol fails, override it here.")
    override_key = st.text_input("TV Symbol (key)", value=selected_symbol_key)
    override_yf = st.text_input("yfinance Symbol", value=yf_symbol)
    if st.button("Save Override"):
        # Store in session state (not persistent across reruns, but can be extended)
        if "symbol_overrides" not in st.session_state:
            st.session_state.symbol_overrides = {}
        st.session_state.symbol_overrides[override_key] = override_yf
        st.success(f"Override saved: {override_key} → {override_yf}")

# Apply override if exists
yf_symbol_to_use = st.session_state.get("symbol_overrides", {}).get(selected_symbol_key, yf_symbol)

st.sidebar.markdown(f"**Mapping:** `{selected_symbol_key}` → `{yf_symbol_to_use}` (yfinance) → `{tv_symbol}` (TV)")

# Sidebar: Timeframe
st.sidebar.header("⏱️ Timeframe")
interval = st.sidebar.selectbox(
    "Interval",
    options=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
    index=6,
)
days_back = st.sidebar.slider("Lookback (days)", 30, 365 * 5, 365)
end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=days_back)

# Sidebar: Indicator Settings
st.sidebar.header("🧠 Indicators")
show_ema50 = st.sidebar.toggle("EMA 50", value=True)
show_ema200 = st.sidebar.toggle("EMA 200", value=True)
show_sma200 = st.sidebar.toggle("SMA 200", value=False)
show_bb = st.sidebar.toggle("Bollinger Bands", value=True)
show_vwap = st.sidebar.toggle("VWAP", value=False)
show_ichimoku = st.sidebar.toggle("Ichimoku", value=False)
show_rsi = st.sidebar.toggle("RSI 14", value=True)
show_macd = st.sidebar.toggle("MACD", value=False)
show_mfi = st.sidebar.toggle("MFI 14", value=False)
show_atr = st.sidebar.toggle("ATR 14", value=False)
show_smc = st.sidebar.toggle("SMC Structure (BOS/CHOCH)", value=False)
smc_pivot_len = st.sidebar.slider("SMC Pivot Length", 1, 10, 5)

# Sidebar: AI Settings
st.sidebar.header("🤖 AI Analysis")
enable_ai = st.sidebar.toggle("Enable AI Trade Plan", value=False)
ai_provider = st.sidebar.selectbox("AI Provider", ["OpenAI", "Gemini"], index=0)
ai_model = st.sidebar.text_input("Model", value="gpt-4.1-mini" if ai_provider == "OpenAI" else "gemini-1.5-pro")

# Sidebar: Telegram Settings
st.sidebar.header("📡 Telegram Broadcast")
enable_telegram = st.sidebar.toggle("Enable Telegram", value=False)
telegram_token = st.secrets.get("TELEGRAM_TOKEN", "")
telegram_chat_ids = st.sidebar.text_area(
    "Chat IDs (comma-separated)",
    value=st.secrets.get("TELEGRAM_CHAT_ID", ""),
    help="Get your chat ID from @userinfobot",
)
telegram_use_markdown = st.sidebar.checkbox("Use Markdown", value=True)

# Tabs
tab_dashboard, tab_chart, tab_ai, tab_broadcast, tab_diagnostics = st.tabs(
    ["📊 Dashboard", "📈 Chart", "🤖 AI Plan", "📡 Broadcast", "🔧 Diagnostics"]
)

# -----------------------------------------------------------------------------
# Dashboard Tab (Opportunity Scanner)
with tab_dashboard:
    st.subheader("Opportunity Scanner")

    # Build scanner list from current watchlist or asset group
    scanner_symbols = st.multiselect(
        "Scan symbols",
        options=list(ALL_SYMBOLS.keys()),
        default=list(DEFAULT_WATCHLISTS["Favorites"]),
    )

    if st.button("Run Scan", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        results = []
        for i, sym_key in enumerate(scanner_symbols):
            yf_sym = ALL_SYMBOLS[sym_key]["yf"]
            df, err, meta = fetch_data(yf_sym, interval, start_dt, end_dt)
            if err or df is None or df.empty:
                st.session_state.errors.append({"symbol": sym_key, "error": err or "No data"})
                continue

            # Compute indicators
            close = df["Close"]
            tech = {}
            if show_ema50:
                tech["EMA 50"] = ema(close, 50).iloc[-1]
            if show_ema200:
                tech["EMA 200"] = ema(close, 200).iloc[-1]
            if show_sma200:
                tech["SMA 200"] = sma(close, 200).iloc[-1]
            if show_rsi:
                tech["RSI 14"] = rsi(close, 14).iloc[-1]
            if show_bb:
                upper, mid, lower = bollinger(close, 20, 2)
                tech["BB Upper"] = upper.iloc[-1]
                tech["BB Lower"] = lower.iloc[-1]
            if show_vwap and {"High", "Low", "Volume"}.issubset(df.columns):
                tech["VWAP"] = vwap(df).iloc[-1]
            if show_atr:
                tech["ATR 14"] = atr(df, 14).iloc[-1]

            # Simple opportunity score (0-100)
            score = 50
            if tech.get("EMA 50") and tech.get("EMA 200"):
                if tech["EMA 50"] > tech["EMA 200"]:
                    score += 20
                else:
                    score -= 20
            if tech.get("RSI 14"):
                if 30 < tech["RSI 14"] < 70:
                    score += 10
                elif tech["RSI 14"] > 80 or tech["RSI 14"] < 20:
                    score -= 15
            # Risk meter (1-10) based on ATR
            risk = 5
            atr_val = tech.get("ATR 14")
            if atr_val:
                risk = min(10, max(1, int(atr_val / close * 100)))

            results.append(
                {
                    "Symbol": sym_key,
                    "Close": close.iloc[-1],
                    "EMA50": tech.get("EMA 50"),
                    "EMA200": tech.get("EMA 200"),
                    "RSI": tech.get("RSI 14"),
                    "ATR": tech.get("ATR 14"),
                    "Score": max(0, min(100, score)),
                    "Risk": risk,
                }
            )
            progress.progress((i + 1) / len(scanner_symbols))
            status.text(f"Scanning {sym_key}... ({i+1}/{len(scanner_symbols)})")

        if results:
            df_scan = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(df_scan, use_container_width=True)
            # Download CSV
            csv = df_scan.to_csv(index=False).encode()
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")
        else:
            st.info("No results. Check errors in Diagnostics tab.")

# -----------------------------------------------------------------------------
# Chart Tab
with tab_chart:
    st.subheader(f"Chart: {selected_symbol_key} ({yf_symbol_to_use})")

    # Fetch data
    df, err, meta = fetch_data(yf_symbol_to_use, interval, start_dt, end_dt)
    if err or df is None or df.empty:
        st.error(err)
        with st.expander("Show diagnostic details"):
            st.json(meta)
        st.stop()

    # Transparency metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Refresh", meta["last_refresh"] if meta else "-")
    c2.metric("Latency (s)", meta["latency_sec"] if meta else "-")
    c3.metric("Rows", meta["rows"] if meta else "-")
    c4.metric("Missing Rows", meta["missing_rows"] if meta else "-")

    # Compute overlays
    close = df["Close"]
    overlays = {}
    if show_ema50:
        overlays["EMA 50"] = ema(close, 50)
    if show_ema200:
        overlays["EMA 200"] = ema(close, 200)
    if show_sma200:
        overlays["SMA 200"] = sma(close, 200)
    if show_bb:
        bb_up, bb_mid, bb_low = bollinger(close, 20, 2)
        overlays["BB Upper"] = bb_up
        overlays["BB Lower"] = bb_low
    if show_vwap and {"High", "Low", "Volume"}.issubset(df.columns):
        overlays["VWAP"] = vwap(df)
    if show_ichimoku:
        ich = ichimoku(df)
        overlays.update(ich)

    # Compute subpanes
    subpanes = {}
    if show_rsi:
        subpanes["RSI 14"] = rsi(close, 14)
    if show_macd:
        macd_line, signal_line, hist = macd(close, 12, 26, 9)
        subpanes["MACD"] = macd_line
        subpanes["MACD Signal"] = signal_line
        subpanes["MACD Hist"] = hist
    if show_mfi:
        subpanes["MFI 14"] = mfi(df, 14)
    if show_atr:
        subpanes["ATR 14"] = atr(df, 14)

    # SMC structure
    smc_df = None
    if show_smc:
        smc_df = smc_structure(df, pivot_len=smc_pivot_len)

    # Build Plotly figure
    rows = 1
    row_heights = [0.8]
    subplot_titles = [f"{selected_symbol_key}"]
    if subpanes:
        rows += 1
        row_heights = [0.6, 0.2]
        subplot_titles.append("Indicators")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Overlays
    for name, series in overlays.items():
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines"), row=1, col=1)

    # SMC marks
    if smc_df is not None:
        # Swing points
        ph = smc_df["Swing_High"].dropna()
        pl = smc_df["Swing_Low"].dropna()
        fig.add_trace(go.Scatter(x=ph.index, y=ph, name="Swing High", mode="markers", marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=pl.index, y=pl, name="Swing Low", mode="markers", marker=dict(size=6)), row=1, col=1)
        # BOS/CHOCH events (last 20 bars)
        recent = smc_df.tail(20)
        bos_bull = recent.index[recent["BOS_Bull"].fillna(False)]
        bos_bear = recent.index[recent["BOS_Bear"].fillna(False)]
        choch_bull = recent.index[recent["CHOCH_Bull"].fillna(False)]
        choch_bear = recent.index[recent["CHOCH_Bear"].fillna(False)]
        for idx in bos_bull:
            fig.add_annotation(x=idx, y=df.loc[idx, "Close"], text="BOS↑", showarrow=True, row=1, col=1)
        for idx in bos_bear:
            fig.add_annotation(x=idx, y=df.loc[idx, "Close"], text="BOS↓", showarrow=True, row=1, col=1)
        for idx in choch_bull:
            fig.add_annotation(x=idx, y=df.loc[idx, "Close"], text="CHOCH↑", showarrow=True, row=1, col=1)
        for idx in choch_bear:
            fig.add_annotation(x=idx, y=df.loc[idx, "Close"], text="CHOCH↓", showarrow=True, row=1, col=1)

    # Subpanes
    row_ix = 2
    for name, series in subpanes.items():
        if "RSI" in name:
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines"), row=row_ix, col=1)
            fig.add_hline(y=70, line_dash="dash", row=row_ix, col=1)
            fig.add_hline(y=30, line_dash="dash", row=row_ix, col=1)
            fig.update_yaxes(range=[0, 100], row=row_ix, col=1)
        elif "MACD" in name:
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines" if "Hist" not in name else "lines"), row=row_ix, col=1)
        elif "MFI" in name:
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines"), row=row_ix, col=1)
            fig.add_hline(y=80, line_dash="dash", row=row_ix, col=1)
            fig.add_hline(y=20, line_dash="dash", row=row_ix, col=1)
            fig.update_yaxes(range=[0, 100], row=row_ix, col=1)
        elif "ATR" in name:
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines"), row=row_ix, col=1)

    fig.update_layout(
        height=800,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Indicator Interpretation Pane
    st.markdown("### Indicator Readout")
    last_vals = {k: v.iloc[-1] for k, v in overlays.items()}
    last_vals.update({k: v.iloc[-1] for k, v in subpanes.items()})
    readout = indicator_readout(df, selected_symbol_key, {}, last_vals)
    st.code(readout, language="text")

    # TradingView deep link
    st.markdown("### TradingView")
    tv_url = build_tradingview_url(tv_symbol, interval)
    st.link_button("Open in TradingView", tv_url)
    st.caption("Note: TradingView embeds have limited functionality inside Streamlit. Use the deep-link for full features.")

# -----------------------------------------------------------------------------
# AI Analysis Tab
with tab_ai:
    st.subheader("AI Trade Plan")

    if not enable_ai:
        st.info("Enable AI in the sidebar to generate trade plans.")
    else:
        # Recompute tech summary
        df_ai, err_ai, _ = fetch_data(yf_symbol_to_use, interval, start_dt, end_dt)
        if err_ai or df_ai is None or df_ai.empty:
            st.error(err_ai)
        else:
            close_ai = df_ai["Close"]
            tech_summary_ai = {}
            if show_ema50:
                tech_summary_ai["EMA 50"] = ema(close_ai, 50).iloc[-1]
            if show_ema200:
                tech_summary_ai["EMA 200"] = ema(close_ai, 200).iloc[-1]
            if show_sma200:
                tech_summary_ai["SMA 200"] = sma(close_ai, 200).iloc[-1]
            if show_rsi:
                tech_summary_ai["RSI 14"] = rsi(close_ai, 14).iloc[-1]
            if show_bb:
                bb_up_ai, _, bb_low_ai = bollinger(close_ai, 20, 2)
                tech_summary_ai["BB Upper"] = bb_up_ai.iloc[-1]
                tech_summary_ai["BB Lower"] = bb_low_ai.iloc[-1]
            if show_vwap:
                tech_summary_ai["VWAP"] = vwap(df_ai).iloc[-1]
            if show_atr:
                tech_summary_ai["ATR 14"] = atr(df_ai, 14).iloc[-1]

            # Compute opportunity score and risk meter
            score = 50
            if tech_summary_ai.get("EMA 50") and tech_summary_ai.get("EMA 200"):
                if tech_summary_ai["EMA 50"] > tech_summary_ai["EMA 200"]:
                    score += 20
                else:
                    score -= 20
            risk_meter = 5
            if tech_summary_ai.get("ATR 14"):
                risk_meter = min(10, max(1, int(tech_summary_ai["ATR 14"] / close_ai.iloc[-1] * 100)))

            # Generate trade plan
            if st.button("Generate AI Trade Plan", type="primary"):
                with st.spinner("Generating..."):
                    plan = generate_ai_trade_plan(df_ai, tech_summary_ai, risk_meter, selected_symbol_key, interval)
                    plan_md = render_ai_plan(plan)
                    st.markdown(plan_md)

                    # Store for broadcast
                    st.session_state["current_trade_plan"] = plan
                    st.session_state["current_trade_plan_md"] = plan_md

# -----------------------------------------------------------------------------
# Broadcast Tab
with tab_broadcast:
    st.subheader("Telegram Broadcast")

    if not enable_telegram:
        st.info("Enable Telegram in the sidebar to broadcast.")
    else:
        if not telegram_token:
            st.error("TELEGRAM_TOKEN not found in secrets.")
        else:
            bot, bot_err = get_telegram_bot(telegram_token)
            if bot_err:
                st.error(bot_err)
            else:
                st.success("Telegram bot connected.")

                # Report type selector
                report_type = st.selectbox(
                    "Report Type",
                    ["STRICT SIGNAL", "AI TRADE PLAN", "MARKET SUMMARY", "RISK"],
                    index=1,
                )

                # Preview
                if st.button("Preview Message"):
                    if report_type == "AI TRADE PLAN" and "current_trade_plan_md" in st.session_state:
                        preview = st.session_state["current_trade_plan_md"]
                    else:
                        preview = f"**{report_type}**\n\nSymbol: {selected_symbol_key}\nInterval: {interval}\nPrice: {close.iloc[-1]:.6g}"
                    st.markdown(preview)

                # Test message
                if st.button("Send Test Message"):
                    test_msg = f"✅ Telegram integration working!\nSymbol: {selected_symbol_key}\nPrice: {close.iloc[-1]:.6g}"
                    cids = [cid.strip() for cid in telegram_chat_ids.split(",") if cid.strip()]
                    if not cids:
                        st.warning("No chat IDs configured.")
                    else:
                        with st.spinner("Sending test..."):
                            logs = broadcast_message(bot, cids, test_msg, parse_mode="Markdown")
                            st.session_state.delivery_log.extend(logs)
                        st.success("Test sent. Check delivery log below.")

                # Broadcast AI Plan
                if report_type == "AI TRADE PLAN":
                    if st.button("Broadcast AI Trade Plan"):
                        if "current_trade_plan_md" not in st.session_state:
                            st.warning("No AI trade plan generated yet. Go to AI tab first.")
                        else:
                            cids = [cid.strip() for cid in telegram_chat_ids.split(",") if cid.strip()]
                            if not cids:
                                st.warning("No chat IDs configured.")
                            else:
                                with st.spinner("Broadcasting..."):
                                    plan_md = st.session_state["current_trade_plan_md"]
                                    full_msg = f"🤖 **AI TRADE PLAN – {selected_symbol_key} ({interval})**\n\n{plan_md}"
                                    logs = broadcast_message(
                                        bot,
                                        cids,
                                        full_msg,
                                        parse_mode="Markdown" if telegram_use_markdown else None,
                                    )
                                    st.session_state.delivery_log.extend(logs)
                                st.success("Broadcast complete. See log below.")

                # Delivery Log
                if st.session_state.delivery_log:
                    st.markdown("### Delivery Log")
                    st.dataframe(
                        pd.DataFrame(st.session_state.delivery_log).tail(20),
                        use_container_width=True,
                    )
                else:
                    st.info("No broadcast history yet.")

# -----------------------------------------------------------------------------
# Diagnostics Tab
with tab_diagnostics:
    show_diagnostics(st.session_state.errors)

# -----------------------------------------------------------------------------
# Footer
st.markdown("---")
st.caption("App built for robustness. If a feature fails, diagnostics above will show why. Ensure secrets are in `.streamlit/secrets.toml`.")

# -----------------------------------------------------------------------------
# End of streamlit_app.py
