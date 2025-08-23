# ===================== DAT Dashboard (clean) =====================
# Order: imports -> helpers -> load CSV -> prices -> with_live_fields -> compute_row -> DF -> UI
# ================================================================

import os, re
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# Plotly optional; we’ll fall back to Altair if not installed
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import altair as alt  # Streamlit ships with Altair

APP_DIR = Path(__file__).parent.resolve()
st.set_page_config(page_title="Digital Asset Treasury Dashboard", page_icon="🟠")

# ------------------------------------------------------------------
# Intro / Glossary
#
# Provide a brief explainer for the dashboard and define terms (NAV/MNAV).
st.markdown(
    """
    ### Digital Asset Treasury Dashboard

    This dashboard provides insights into publicly listed companies that hold digital assets.  
    Use the selector below to explore treasury composition, market capitalization versus crypto treasury, valuation & MNAV, and to view a full screener table.  

    **Glossary:**
    - **Net Crypto NAV**: Treasury USD minus total liabilities.
    - **NAV per share**: Net Crypto NAV divided by shares outstanding.
    - **MNAV (x)**: Share price divided by NAV per share (multiple).
    """,
    unsafe_allow_html=False,
)

# Track CSV source and load time for display
LOADED_CSV_NAME: str | None = None  # name of the CSV file loaded for screener
LOADED_TIME: datetime | None = None  # timestamp when the CSV was processed

# Analysis selector (place this in the main content, not in st.sidebar)
analysis_options = {
    "Overview": "overview",
    "Treasury Composition": "treasury",
    "Market Cap vs Treasury": "market_vs_treasury",
    "Valuation & MNAV": "valuation",
    "Table": "table",
}
selected_analysis_label = st.selectbox(
    "Select analysis",
    list(analysis_options.keys()),
    index=0,
    help="Choose which section of the dashboard to view.",
)
analysis_key = analysis_options[selected_analysis_label]

# Note: conditional rendering based on `analysis_key` is defined
# later in the script, after the DataFrame and filters are prepared.



# --------------------------- Helpers ----------------------------

def add_mnav(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Net Crypto NAV, NAV/share, MNAV(x) using columns already in DataFrame."""
    treas_col  = "Treasury USD"
    liab_col   = "Total Liabilities"
    shares_col = "Shares Outstanding"
    price_col  = "Share price USD"

    # Fallbacks if your DataFrame used different headers
    if treas_col not in df.columns and "treasury_usd" in df.columns: treas_col = "treasury_usd"
    if liab_col  not in df.columns and "liabilities"  in df.columns: liab_col  = "liabilities"
    if shares_col not in df.columns and "shares_out"  in df.columns: shares_col = "shares_out"
    if price_col  not in df.columns and "share_price" in df.columns: price_col  = "share_price"

    for c in [treas_col, liab_col, shares_col, price_col]:
        if c not in df.columns: df[c] = 0

    df[treas_col]  = pd.to_numeric(df[treas_col],  errors="coerce").fillna(0.0)
    df[liab_col]   = pd.to_numeric(df[liab_col],   errors="coerce").fillna(0.0)
    df[shares_col] = pd.to_numeric(df[shares_col], errors="coerce").fillna(0.0)
    df[price_col]  = pd.to_numeric(df[price_col],  errors="coerce").fillna(0.0)

    df["Net Crypto NAV"] = df[treas_col] - df[liab_col]
    df["NAV per share"]  = np.where(df[shares_col] > 0, df["Net Crypto NAV"] / df[shares_col], np.nan)
    df["MNAV (x)"]       = np.where(df["NAV per share"] > 0, df[price_col] / df["NAV per share"], np.nan)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def fetch_mcap_yahoo(y_symbol: str) -> float | None:
    """Market cap (USD) via yfinance; returns float or None."""
    try:
        t = yf.Ticker(y_symbol)
        cap = None
        try:
            fi = t.fast_info
            cap = fi.get("market_cap")
        except Exception:
            pass
        if not cap:
            inf = t.info or {}
            cap = inf.get("marketCap")
        cap = float(cap) if cap else None
        return cap if cap and cap > 0 else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_equity_snapshot(y_symbol: str) -> dict:
    """{'price', 'shares', 'market_cap'} with multiple fallbacks."""
    try:
        t = yf.Ticker(y_symbol)
        price = mcap = shares = None

        try:
            fi = t.fast_info
            price = (fi.get("last_price") or fi.get("lastPrice")
                     or fi.get("regular_market_price") or fi.get("last")
                     or fi.get("previous_close"))
            mcap = fi.get("market_cap")
        except Exception:
            pass

        try:
            info = t.info or {}
        except Exception:
            info = {}
        if price is None:
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if mcap is None:
            mcap = info.get("marketCap")
        shares = (info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or info.get("floatShares"))

        if price is None:
            try:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        if shares is None and mcap is not None and price is not None and float(price) != 0:
            try:
                shares = float(mcap) / float(price)
            except Exception:
                pass

        return {
            "price": float(price) if price is not None else None,
            "shares": float(shares) if shares is not None else None,
            "market_cap": float(mcap) if mcap is not None else None,
        }
    except Exception:
        return {"price": None, "shares": None, "market_cap": None}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fx(symbol: str) -> float | None:
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", {}) or {}
        px = (info.get("last_price") or info.get("lastPrice") or info.get("regular_market_price"))
        if not px:
            hist = t.history(period="1d")
            if not hist.empty:
                px = float(hist["Close"].iloc[-1])
        return float(px) if px else None
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fx_map_usd() -> dict:
    """Return FX map quoted as 1 CCY -> USD."""
    return {
        "USD": 1.0,
        "CAD": fetch_fx("CADUSD=X") or 0.75,
        "EUR": fetch_fx("EURUSD=X") or 1.08,
        "JPY": fetch_fx("JPYUSD=X") or 0.0063,
        "GBX": (fetch_fx("GBPUSD=X") or 1.27) / 100.0,  # pence -> USD
        "GBP": fetch_fx("GBPUSD=X") or 1.27,
    }

def fmt_abbrev(v):
    """
    Format large numbers into human‑friendly strings using shortened units.

    Rules:
      • Thousands remain numeric with comma separators (e.g. 1,234).
      • Values ≥1M and <1B are displayed in millions with no decimals (e.g. 235M).
      • Values ≥1B and <1T are displayed in billions with two decimals (e.g. 2.45B).
      • Values ≥1T are displayed in trillions with two decimals (e.g. 1.23T).
    Negative numbers retain the minus sign. Non‑numeric values are returned unchanged.
    """
    try:
        n = float(v)
    except Exception:
        return v
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs >= 1_000_000_000_000:
        return f"{sign}${n_abs/1_000_000_000_000:,.2f}T"
    elif n_abs >= 1_000_000_000:
        return f"{sign}${n_abs/1_000_000_000:,.2f}B"
    elif n_abs >= 1_000_000:
        return f"{sign}${n_abs/1_000_000:,.0f}M"
    else:
        return f"{sign}${n_abs:,.0f}"


def _format_number_no_currency(n: float) -> str:
    """
    Internal helper to format a numeric value without a currency symbol.

    The same rules as fmt_abbrev are applied but without the leading "$".
    Used for axis tick labels and chart text where currency symbols may clutter the visuals.
    """
    try:
        value = float(n)
    except Exception:
        return str(n)
    sign = "-" if value < 0 else ""
    value_abs = abs(value)
    if value_abs >= 1_000_000_000_000:
        return f"{sign}{value_abs/1_000_000_000_000:,.2f}T"
    elif value_abs >= 1_000_000_000:
        return f"{sign}{value_abs/1_000_000_000:,.2f}B"
    elif value_abs >= 1_000_000:
        return f"{sign}{value_abs/1_000_000:,.0f}M"
    else:
        return f"{sign}{value_abs:,.0f}"

EXCHANGE_TO_CCY = {
    "NASDAQ": "USD", "NYSE": "USD",
    "TSXV": "CAD",
    "AQUIS": "GBX", "LON": "GBX",
    "TYO": "JPY", "ETR": "EUR",
}

YAHOO_SUFFIX_BY_EXCHANGE = {
    "NASDAQ": "", "NYSE": "", "AMEX": "",
    "TSX": ".TO", "TSXV": ".V", "CSE": ".CN",
    "LON": ".L", "AQUIS": ".AQ", "ETR": ".DE", "TYO": ".T",
}

def yahoo_symbol_for(ticker: str, exchange: str | None) -> str:
    t = (ticker or "").strip().upper()
    ex = (exchange or "").strip().upper()
    if t in YAHOO_MAP:
        return YAHOO_MAP[t]
    suffix = YAHOO_SUFFIX_BY_EXCHANGE.get(ex, "")
    return f"{t}{suffix}"

# Manual overrides
YAHOO_MAP = {
    "BTCT.V":"BTCT.V","MATA.V":"MATA.V","SWC.AQ":"SWC.AQ","3350.T":"3350.T",
    "CEP":"CEP","SQNS":"SQNS","HOLO":"HOLO","SATS.L":"SATS.L","DJT":"DJT",
    "MSTR":"MSTR","NA":"NA","ADE.DE":"ADE.DE","BMNR":"BMNR","FGNX":"FGNX","SBET":"SBET",
    "SWC": "SWC.AQ",  # SWC ticker → AQUIS suffix
}

NAME_TO_MCAP_CCY = {
    "Bitcoin Treasury Corp": "CAD",
    "Matador Technologies Inc": "CAD",
    "The Smarter Web Company PLC": "GBP",
    "Satsuma Technology": "GBP",
    "Bitcoin Group SE": "EUR",
}

# ----------------- CSV parsing function -----------------

def load_companies_and_holdings(src):
    """Parse CSV or uploaded file-like into (companies list, holdings dict)."""
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["ticker", "name", "exchange", "primary", "btc", "eth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing required columns: {missing}")
        st.stop()

    # numbers
    for col in ["btc", "eth"]:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False)
                              .str.replace("$", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce").fillna(0.0))

    # liabilities
    if "liabilities" not in df.columns:
        df["liabilities"] = 0
    df["liabilities"] = (df["liabilities"].astype(str).str.replace(",", "", regex=False)
                                            .str.replace("$", "", regex=False)
                         .pipe(pd.to_numeric, errors="coerce").fillna(0.0))

    # shares_out / sharesout normalization
    col = ("shares_out" if "shares_out" in df.columns else ("sharesout" if "sharesout" in df.columns else None))
    if col is not None:
        ser = (df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False))
        df["shares_out"] = pd.to_numeric(ser, errors="coerce").fillna(0.0)
    else:
        df["shares_out"] = 0.0

    companies_csv = [
        {
            "ticker":      str(r["ticker"]).strip(),
            "name":        str(r["name"]).strip(),
            "exchange":    str(r["exchange"]).strip(),
            "primary":     str(r["primary"]).strip(),
            "liabilities": float(r["liabilities"] or 0.0),
            "shares_out":  float(r.get("shares_out", 0.0) or 0.0),
            "market_cap_usd": 0.0,  # filled live later
        }
        for _, r in df.iterrows()
    ]

    holdings_csv = {
        r["ticker"].strip(): {
            "btc": float(r["btc"]),
            "eth": float(r["eth"]),
            # add stables later if/when present
        }
        for _, r in df.iterrows()
    }

    return companies_csv, holdings_csv

# -------------------- CSV loader (one place) --------------------

uploaded = st.sidebar.file_uploader("Upload companies_holdings.csv", type=["csv"])
if uploaded is not None:
    companies, HOLDINGS = load_companies_and_holdings(uploaded)
    # Record CSV name and load time when user uploads a file
    LOADED_CSV_NAME = getattr(uploaded, "name", "uploaded.csv")
    LOADED_TIME = datetime.now(timezone.utc)
else:
    env_csv = os.environ.get("DAT_CSV_PATH")
    candidates = [
        Path(env_csv) if env_csv else None,
        APP_DIR / "companies_holdings.csv",
        APP_DIR / "data" / "companies_holdings.csv",
        APP_DIR / "data" / "sample_companies.csv",  # safe demo
    ]
    CSV_PATH = next((p for p in candidates if p and p.exists()), None)
    if CSV_PATH:
        if CSV_PATH.name != "companies_holdings.csv":
            st.info(f"Using demo CSV: {CSV_PATH.name} (upload to override)")
        companies, HOLDINGS = load_companies_and_holdings(CSV_PATH)
        # Record CSV name and load time for display
        LOADED_CSV_NAME = CSV_PATH.name
        LOADED_TIME = datetime.now(timezone.utc)
    else:
        st.warning("No CSV found. Upload a file to continue.")
        st.stop()

st.caption(f"Loaded {len(companies)} companies")

# Build Liabilities lookup now that companies exists
LIABILITIES = {
    (c.get("ticker") or "").strip(): float(c.get("liabilities", 0) or 0)
    for c in companies
}

# -------------------- Day 1: Live Prices -----------------------

st.subheader("Live Prices (USD)")
@st.cache_data(ttl=60, show_spinner=False)
def fetch_prices():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin,ethereum,usd-coin", "vs_currencies": "usd"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return {"BTC": data["bitcoin"]["usd"], "ETH": data["ethereum"]["usd"], "USDC": data["usd-coin"]["usd"]}

# Helper to render live crypto prices below each chart section.
def render_live_prices():
    """Display live crypto prices (BTC, ETH, USDC) as metrics with last update timestamp."""
    # Label for the live prices section
    st.markdown("#### Live Crypto Prices (USD)")
    try:
        prices = fetch_prices()
    except Exception:
        st.error("Error fetching crypto prices.")
        return
    # Create three columns for BTC, ETH, and USDC
    col_btc, col_eth, col_usdc = st.columns(3)
    btc_val = prices.get("BTC")
    eth_val = prices.get("ETH")
    usdc_val = prices.get("USDC")
    # Format and display metrics. Delta omitted since we do not track changes over time.
    col_btc.metric("Bitcoin (BTC)", f"${btc_val:,.2f}" if btc_val is not None else "–")
    col_eth.metric("Ethereum (ETH)", f"${eth_val:,.2f}" if eth_val is not None else "–")
    col_usdc.metric("USD Coin (USDC)", f"${usdc_val:,.2f}" if usdc_val is not None else "–")
    # Caption with the current timestamp for update time
    st.caption(f"Prices updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    prices = fetch_prices()
    c = st.columns(3)
    c[0].metric("BTC", f"${prices['BTC']:,}")
    c[1].metric("ETH", f"${prices['ETH']:,}")
    c[2].metric("USDC", f"${prices['USDC']:,}")
    # Show a green dot to indicate fresh data
    green_dot = "<span style='color:green'>&#9679;</span>"
    st.caption(
        f"{green_dot} Last updated: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        unsafe_allow_html=True,
    )
except Exception as e:
    prices = {"BTC": 0.0, "ETH": 0.0, "USDC": 0.0}
    st.error("Couldn't load prices.")
    st.exception(e)

st.divider()
# Wallet balances section is collapsible to reduce clutter
with st.sidebar.expander("API / Wallet Settings", expanded=False):
    st.subheader("Wallet Settings")
    covalent_api_key = st.text_input("Covalent API Key", type="password")
    eth_address = st.text_input(
        "ETH Address (0x...)",
        placeholder="e.g., 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
    )
    # Only allow fetching balances when both API key and address are provided
    fetch_btn = st.button(
        "Fetch Balances",
        disabled=not bool(covalent_api_key and eth_address),
    )

def fetch_eth_balances_covalent(api_key: str, address: str):
    if not api_key or not address:
        raise ValueError("API key and address are required.")
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.8, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"], raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    url = f"https://api.covalenthq.com/v1/1/address/{address}/balances_v2/"
    r = session.get(url, params={"key": api_key, "nft":"false", "no-nft-fetch":"true", "quote-currency":"USD"}, timeout=45)
    r.raise_for_status()
    items = (r.json().get("data",{}) or {}).get("items",[]) or []
    rows = []
    for it in items:
        try:
            decimals = it.get("contract_decimals", 0) or 0
            human = int(it.get("balance","0") or "0") / (10**decimals)
        except Exception:
            human = None
        if human and abs(human) > 0:
            rows.append({
                "Token": it.get("contract_ticker_symbol",""),
                "Name": it.get("contract_name",""),
                "Amount": human,
                "USD (quote)": it.get("quote"),
                "Explorer": f"https://etherscan.io/token/{it.get('contract_address')}?a={address}" if it.get("contract_address") else f"https://etherscan.io/address/{address}",
            })
    dfw = pd.DataFrame(rows)
    if not dfw.empty and "USD (quote)" in dfw.columns:
        dfw = dfw.sort_values(by=["USD (quote)"], ascending=False, na_position="last")
    return dfw, f"https://etherscan.io/address/{address}"

if fetch_btn:
    with st.spinner("Fetching balances…"):
        try:
            dfw, addr_link = fetch_eth_balances_covalent(covalent_api_key, eth_address)
            st.markdown(f"**Address:** [{eth_address}]({addr_link})")
            if dfw.empty:
                st.info("No non-zero token balances found.")
            else:
                dfw = dfw.copy()
                dfw["Explorer"] = dfw["Explorer"].apply(lambda u: f"[link]({u})")
                st.dataframe(dfw, use_container_width=True)
        except Exception as e:
            st.error("Couldn’t fetch balances.")
            st.exception(e)

# -------------------- Company Screener --------------------------



def with_live_fields(c: dict) -> dict:
    """Attach market cap USD, price USD, holdings, liabilities, and NAV/share."""
    c2 = c.copy()

    # Symbols / metadata
    ysym = yahoo_symbol_for(c.get("ticker"), c.get("exchange"))
    exch = (c.get("exchange") or c.get("Exchange") or "").strip().upper()

    # FX maps and base currency by exchange
    fxm = fx_map_usd()
    ccy = EXCHANGE_TO_CCY.get(exch, "USD")

    # Live snapshot
    snap = {}
    if ysym:
        snap = fetch_equity_snapshot(ysym) or {}

    # --- Share price (normalize to USD) ---
    if (float(c2.get("Share price USD", c2.get("share_price", 0)) or 0) == 0) and (snap.get("price") is not None):
        native_price = float(snap["price"])
        fx_price = fxm.get(ccy, 1.0)  # GBX handled in fx_map_usd
        c2["Share price USD"] = native_price * fx_price

    # --- Market cap (native -> USD) ---
    mcap_ccy = (str(snap.get("currency") or c.get("currency") or "USD")).upper()
    if str(snap.get("quote_ex")).upper() == "GBX":
        mcap_ccy = "GBP"
    comp_name = c.get("name") or c.get("Name")
    if comp_name in NAME_TO_MCAP_CCY:
        mcap_ccy = NAME_TO_MCAP_CCY[comp_name]
    fx_cap = fxm.get(mcap_ccy, 1.0)

    usd_mcap = None
    cap_native = fetch_mcap_yahoo(ysym) if ysym else None
    if cap_native:
        usd_mcap = float(cap_native) * float(fx_cap)
    if usd_mcap is None and (snap.get("market_cap") is not None):
        usd_mcap = float(snap["market_cap"]) * float(fx_cap)
    if usd_mcap is not None:
        c2["market_cap_usd"] = usd_mcap

    # --- Fallback: compute mcap from price × shares ---
    if not c2.get("market_cap_usd"):
        px = float(c2.get("Share price USD", c2.get("share_price", 0)) or 0)
        so = float(c2.get("Shares Outstanding", c2.get("shares_out", 0)) or 0)
        if px > 0 and so > 0:
            c2["market_cap_usd"] = px * so

    # --- Holdings from CSV dict ---
    h = HOLDINGS.get(c.get("ticker"), {})
    c2["holdings_btc"]     = float(h.get("btc", 0) or 0)
    c2["holdings_eth"]     = float(h.get("eth", 0) or 0)
    c2["holdings_stables"] = float(h.get("stables_usd", 0) or 0)

    # --- Liabilities + NAV/share ---
    c2["Liabilities"] = float(h.get("liabilities", c.get("liabilities", 0)) or 0)
    # treasury_usd will be computed in compute_row with live prices
    shares_out = float(c2.get("Shares Outstanding", c2.get("shares_out", 0)) or 0)
    net_nav = 0.0  # temporary until compute_row
    c2["Net NAV (USD)"]    = net_nav
    c2["NAV/ Share (USD)"] = (net_nav / shares_out) if shares_out > 0 else 0.0

    # Canonicalize mcap from price × shares when possible
    px_usd = float(c2.get("Share price USD", 0) or 0)
    so_val = float(c2.get("Shares Outstanding", c2.get("shares_out", 0)) or 0)
    if px_usd > 0 and so_val > 0:
        c2["market_cap_usd"] = px_usd * so_val

    return c2

def compute_row(c: dict, prices: dict) -> dict:
    """Build one numeric row (floats) for the table & charts."""
    btc_usd = (c.get("holdings_btc", 0.0) or 0.0) * prices.get("BTC", 0.0)
    eth_usd = (c.get("holdings_eth", 0.0) or 0.0) * prices.get("ETH", 0.0)
    stables_usd = c.get("holdings_stables", 0.0) or 0.0
    treasury_usd = btc_usd + eth_usd + stables_usd

    liab = float(c.get("Liabilities", c.get("liabilities", 0.0)) or 0.0)
    shares_out = float(c.get("Shares Outstanding", c.get("shares_out", 0.0)) or 0.0)
    share_price = float(c.get("Share price USD", c.get("share_price", 0.0)) or 0.0)

    net_nav_usd = treasury_usd - liab
    nav_per_share = (net_nav_usd / shares_out) if shares_out > 0 else None
    mnav = (share_price / nav_per_share) if nav_per_share and nav_per_share > 0 else None

    mcap_usd = float(c.get("market_cap_usd", 0.0) or 0.0)
    pct_of_mcap = (treasury_usd / mcap_usd) * 100 if mcap_usd > 0 else 0.0

    return {
        "Ticker": c["ticker"],
        "name": c["name"],
        "exchange": c.get("exchange"),
        "Mkt Cap (USD)": mcap_usd,
        "BTC": c.get("holdings_btc", 0.0),
        "ETH": c.get("holdings_eth", 0.0),
        "Treasury USD": treasury_usd,
        "% of Mkt Cap": pct_of_mcap,
        "Total Liabilities": liab,
        "Shares Outstanding": shares_out,
        "Share price USD": share_price,
        "Net Crypto NAV": net_nav_usd,
        "NAV per share": nav_per_share,
        "MNAV (x)": mnav,
    }

# Build enriched companies & numeric DataFrame once
enriched = [with_live_fields(c) for c in companies]
rows = [compute_row(c, prices) for c in enriched]
df = pd.DataFrame(rows)
df = add_mnav(df)
df = df.sort_values(by="% of Mkt Cap", ascending=False).reset_index(drop=True)

# -------------------- Filters / KPIs / Charts -------------------

# Filter panel in the sidebar
st.sidebar.markdown("### Filters")
if "exchange" in df.columns:
    _ex_opts = sorted([e for e in df["exchange"].dropna().unique().tolist() if e != ""])
    _ex_sel = st.sidebar.multiselect("Exchange", _ex_opts, default=_ex_opts or [])
    df_view = df[df["exchange"].isin(_ex_sel)] if _ex_sel else df.copy()
else:
    df_view = df.copy()

_mc_min = float(df_view["Mkt Cap (USD)"].min() if not df_view.empty else 0.0)
_mc_max = float(df_view["Mkt Cap (USD)"].max() if not df_view.empty else 0.0)
_lo, _hi = st.sidebar.slider(
    "Market Cap (USD, millions)",
    min_value=0.0,
    max_value=max(1.0, _mc_max / 1e6),
    value=(0.0, max(1.0, _mc_max / 1e6)),
)
df_view = df_view[(df_view["Mkt Cap (USD)"] >= _lo * 1e6) & (df_view["Mkt Cap (USD)"] <= _hi * 1e6)]

# ------------------------------------------------------------------
# Conditional rendering based on the selected analysis.
# Each branch below corresponds to one of the previous tabs/sub-tabs.
if analysis_key == "overview":
    st.subheader("Overview")
    total_treasury = np.nansum(df_view["Treasury USD"])
    total_liabilities = np.nansum(df_view["Total Liabilities"])
    avg_mnav_series = df_view["MNAV (x)"].replace([np.inf, -np.inf], np.nan).dropna()
    avg_mnav_value = float(avg_mnav_series.mean()) if not avg_mnav_series.empty else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Treasury", fmt_abbrev(total_treasury), help="Sum of BTC, ETH and stablecoin holdings across the filtered companies")
    c2.metric("Total Liabilities", fmt_abbrev(total_liabilities), help="Sum of reported liabilities for the filtered companies")
    c3.metric("Average MNAV", f"{avg_mnav_value:.2f}x" if pd.notnull(avg_mnav_value) else "–", help="Mean of MNAV (market cap / net asset value)")
    st.caption(f"Filtered rows: {len(df_view)} / {len(df)}")
    # Render live crypto prices below the overview metrics
    render_live_prices()
    st.stop()

elif analysis_key == "treasury":
    st.subheader("Top 10 by Treasury (USD)")
    st.caption("Companies with the largest crypto treasury holdings")

    data_top = df_view[["Ticker", "name", "Treasury USD"]].dropna()
    if data_top.empty:
        st.info("No rows available for ranking.")
        st.stop()

    # Top-10 by Treasury
    _top = data_top.sort_values("Treasury USD", ascending=False).head(10).copy()

    if HAS_PLOTLY:
        df_plot = _top.copy()
        df_plot["value_str"] = df_plot["Treasury USD"].apply(_format_number_no_currency)

        max_val_top = df_plot["Treasury USD"].max() if not df_plot["Treasury USD"].empty else 0
        if max_val_top and max_val_top > 0:
            tickvals_top = np.linspace(0, max_val_top, 5)
            ticktext_top = [_format_number_no_currency(v) for v in tickvals_top]
        else:
            tickvals_top = [0]
            ticktext_top = ["0"]

        fig_top = px.bar(
            df_plot,
            x="Ticker",
            y="Treasury USD",
            text="value_str",
            hover_data=["name"],
            title=None,
        )
        fig_top.update_yaxes(
            title="Treasury (USD)",
            type="linear",
            tickmode="array",
            tickvals=tickvals_top,
            ticktext=ticktext_top,
        )
        fig_top.update_traces(
            texttemplate="%{text}",
            hovertemplate="<b>%{x}</b><br>Treasury: %{text}<extra></extra>",
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        chart_top = (
            alt.Chart(_top)
            .mark_bar()
            .encode(
                x="Ticker:N",
                y=alt.Y("Treasury USD:Q", title="Treasury (USD)", scale=alt.Scale(type="linear")),
                tooltip=["Ticker", "name", "Treasury USD"],
            )
            .properties(title=None)
        )
        st.altair_chart(chart_top, use_container_width=True)

    # After showing the Top 10 chart, display live crypto prices
    render_live_prices()
    st.stop()


elif analysis_key == "market_vs_treasury":
    st.subheader("Treasury % of Market Cap vs Market Cap")
    st.caption("Relationship between market cap and the share of crypto treasury")

    _sc = df_view[["Ticker", "name", "Treasury USD", "Mkt Cap (USD)", "% of Mkt Cap"]].dropna()
    if _sc.empty:
        st.info("Not enough data for scatter.")
        st.stop()

    _sc = _sc.assign(pct=_sc["% of Mkt Cap"] / 100.0)

    if HAS_PLOTLY:
        max_val_sc = _sc["Mkt Cap (USD)"].max() if not _sc["Mkt Cap (USD)"].empty else 0
        if max_val_sc and max_val_sc > 0:
            tickvals_sc = np.linspace(0, max_val_sc, 5)
            ticktext_sc = [f"{v/1e9:.2f}B" for v in tickvals_sc]
        else:
            tickvals_sc = [0]
            ticktext_sc = ["0B"]

        fig_sc = px.scatter(
            _sc,
            x="Mkt Cap (USD)",
            y="pct",
            size="Treasury USD",
            hover_data=["Ticker", "name"],
            title=None,
            labels={"pct": "% of Market Cap"},
        )
        fig_sc.update_xaxes(
            title="Market Cap (B USD)",
            type="linear",
            tickmode="array",
            tickvals=tickvals_sc,
            ticktext=ticktext_sc,
        )
        fig_sc.update_yaxes(title="% of Market Cap", tickformat=".2%")
        fig_sc.update_layout(legend=dict(orientation="v", y=1, x=1.02))
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        chart_sc = (
            alt.Chart(_sc)
            .mark_circle()
            .encode(
                x=alt.X("Mkt Cap (USD):Q", title="Market Cap (USD)", scale=alt.Scale(type="linear")),
                y=alt.Y("pct:Q", title="% of Market Cap"),
                size="Treasury USD:Q",
                tooltip=["Ticker", "name", "Treasury USD", "Mkt Cap (USD)", "% of Mkt Cap"],
            )
            .properties(title=None)
        )
        st.altair_chart(chart_sc, use_container_width=True)

    # After showing the scatter chart, display live crypto prices
    render_live_prices()
    st.stop()

elif analysis_key == "valuation":
    st.subheader("Liabilities vs Net Crypto NAV")
    st.caption("Compare total liabilities against net crypto NAV across companies")

    liab_nav_df = df_view[["Ticker", "Total Liabilities", "Net Crypto NAV"]].dropna()
    if liab_nav_df.empty:
        st.info("No data for liabilities vs Net Crypto NAV chart.")
        st.stop()

    liab_nav_df_sorted = liab_nav_df.sort_values("Net Crypto NAV", ascending=False)
    liab_nav_long = liab_nav_df_sorted.melt(
        id_vars=["Ticker"],
        value_vars=["Total Liabilities", "Net Crypto NAV"],
        var_name="Metric",
        value_name="Amount",
    )

    if HAS_PLOTLY:
        liab_nav_long = liab_nav_long.copy()
        liab_nav_long["Amount_B"] = liab_nav_long["Amount"] / 1e9

        max_val_ln = liab_nav_long["Amount"].max() if not liab_nav_long["Amount"].empty else 0
        if max_val_ln and max_val_ln > 0:
            tickvals_ln = np.linspace(0, max_val_ln, 5)
            ticktext_ln = [f"{v/1e9:.2f}B" for v in tickvals_ln]
        else:
            tickvals_ln = [0]
            ticktext_ln = ["0B"]

        fig_ln = px.bar(
            liab_nav_long,
            x="Ticker",
            y="Amount",
            color="Metric",
            barmode="group",
            text="Amount_B",
            title=None,
        )
        fig_ln.update_yaxes(
            title="Amount (B USD)",
            type="linear",
            tickmode="array",
            tickvals=tickvals_ln,
            ticktext=ticktext_ln,
        )
        fig_ln.update_traces(
            texttemplate="%{text:.2f}B",
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{text:.2f}B<extra></extra>",
        )
        fig_ln.update_layout(legend=dict(orientation="v", y=1, x=1.02))
        st.plotly_chart(fig_ln, use_container_width=True)
    else:
        ln_chart = (
            alt.Chart(liab_nav_long)
            .mark_bar()
            .encode(
                x="Ticker:N",
                y=alt.Y("Amount:Q", title="Amount (USD)", scale=alt.Scale(type="linear")),
                color="Metric:N",
                tooltip=["Ticker", "Metric", "Amount"],
            )
            .properties(title=None)
        )
        st.altair_chart(ln_chart, use_container_width=True)

    # After showing the liabilities vs net crypto NAV chart, display live crypto prices
    render_live_prices()
    st.stop()


elif analysis_key == "table":
    st.subheader("Company Screener Table")

    df_display = df_view.copy()
    for col in ["Mkt Cap (USD)", "Treasury USD", "Total Liabilities", "Net Crypto NAV"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(fmt_abbrev)

    if "NAV per share" in df_display.columns:
        df_display["NAV per share"] = df_display["NAV per share"].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "–"
        )
    if "Share price USD" in df_display.columns:
        df_display["Share price USD"] = df_display["Share price USD"].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "–"
        )
    if "% of Mkt Cap" in df_display.columns:
        df_display["% of Mkt Cap"] = df_display["% of Mkt Cap"].apply(lambda x: f"{x:.2f}%")
    if "MNAV (x)" in df_display.columns:
        df_display["MNAV (x)"] = df_display["MNAV (x)"].apply(
            lambda x: f"{x:.2f}x" if pd.notnull(x) else "–"
        )

    st.dataframe(df_display, use_container_width=True)
    # Display live crypto prices below the screener table
    render_live_prices()
    st.stop()





    # ------------------------------------------------------------------
    # Sub‑tab 2: Market Cap vs Treasury scatter
    # ------------------------------------------------------------------
    with sub_tab2:
        st.markdown("#### Treasury % of Market Cap vs Market Cap")
        st.caption("Relationship between market cap and the share of crypto treasury")
        _sc = df_view[["Ticker", "name", "Treasury USD", "Mkt Cap (USD)", "% of Mkt Cap"]].dropna()
        if not _sc.empty:
            _sc = _sc.assign(pct=_sc["% of Mkt Cap"] / 100.0)
            if HAS_PLOTLY:
                max_val_sc = _sc["Mkt Cap (USD)"].max() if not _sc["Mkt Cap (USD)"].empty else 0
                if max_val_sc and max_val_sc > 0:
                    tickvals_sc = np.linspace(0, max_val_sc, 5)
                    ticktext_sc = [f"{v/1e9:.2f}B" for v in tickvals_sc]
                else:
                    tickvals_sc = [0]
                    ticktext_sc = ["0B"]
                fig_sc = px.scatter(
                    _sc,
                    x="Mkt Cap (USD)",
                    y="pct",
                    size="Treasury USD",
                    hover_data=["Ticker", "name"],
                    title=None,
                    labels={"pct": "% of Market Cap"},
                )
                fig_sc.update_xaxes(
                    title="Market Cap (B USD)",
                    type="linear",
                    tickmode="array",
                    tickvals=tickvals_sc,
                    ticktext=ticktext_sc,
                )
                fig_sc.update_yaxes(title="% of Market Cap", tickformat=".2%")
                # Provide a vertical legend if necessary
                fig_sc.update_layout(legend=dict(orientation="v", y=1, x=1.02))
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                chart_sc = (
                    alt.Chart(_sc)
                    .mark_circle()
                    .encode(
                        x=alt.X("Mkt Cap (USD):Q", title="Market Cap (USD)", scale=alt.Scale(type="linear")),
                        y=alt.Y("pct:Q", title="% of Market Cap"),
                        size="Treasury USD:Q",
                        tooltip=["Ticker", "name", "Treasury USD", "Mkt Cap (USD)", "% of Mkt Cap"],
                    )
                    .properties(title=None)
                )
                st.altair_chart(chart_sc, use_container_width=True)
        else:
            st.info("Not enough data for scatter.")

    # ------------------------------------------------------------------
    # Sub‑tab 3: Liabilities vs Net Crypto NAV
    # ------------------------------------------------------------------
    with sub_tab3:
        st.markdown("#### Liabilities vs Net Crypto NAV")
        st.caption("Compare total liabilities against net crypto NAV across companies")
        liab_nav_df = df_view[["Ticker", "Total Liabilities", "Net Crypto NAV"]].dropna()
        if not liab_nav_df.empty:
            # Sort by Net Crypto NAV descending for clearer ordering
            liab_nav_df_sorted = liab_nav_df.sort_values("Net Crypto NAV", ascending=False)
            liab_nav_long = liab_nav_df_sorted.melt(
                id_vars=["Ticker"],
                value_vars=["Total Liabilities", "Net Crypto NAV"],
                var_name="Metric",
                value_name="Amount",
            )
            if HAS_PLOTLY:
                liab_nav_long = liab_nav_long.copy()
                liab_nav_long["Amount_B"] = liab_nav_long["Amount"] / 1e9
                max_val_ln = (
                    liab_nav_long["Amount"].max() if not liab_nav_long["Amount"].empty else 0
                )
                if max_val_ln and max_val_ln > 0:
                    tickvals_ln = np.linspace(0, max_val_ln, 5)
                    ticktext_ln = [f"{v/1e9:.2f}B" for v in tickvals_ln]
                else:
                    tickvals_ln = [0]
                    ticktext_ln = ["0B"]
                fig_ln = px.bar(
                    liab_nav_long,
                    x="Ticker",
                    y="Amount",
                    color="Metric",
                    barmode="group",
                    text="Amount_B",
                    title=None,
                )
                fig_ln.update_yaxes(
                    title="Amount (B USD)",
                    type="linear",
                    tickmode="array",
                    tickvals=tickvals_ln,
                    ticktext=ticktext_ln,
                )
                fig_ln.update_traces(
                    texttemplate="%{text:.2f}B",
                    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{text:.2f}B<extra></extra>",
                )
                # Place legend on the right for long lists
                fig_ln.update_layout(legend=dict(orientation="v", y=1, x=1.02))
                st.plotly_chart(fig_ln, use_container_width=True)
            else:
                # For Altair fallback, show values directly
                ln_chart = (
                    alt.Chart(liab_nav_long)
                    .mark_bar()
                    .encode(
                        x="Ticker:N",
                        y=alt.Y("Amount:Q", title="Amount (USD)", scale=alt.Scale(type="linear")),
                        color="Metric:N",
                        tooltip=["Ticker", "Metric", "Amount"],
                    )
                    .properties(title=None)
                )
                st.altair_chart(ln_chart, use_container_width=True)
        else:
            st.info("No data for liabilities vs Net Crypto NAV chart.")

with tab_table:
    st.subheader("Company Screener Table")
    # Format the display DataFrame with abbreviated numbers
    df_display = df_view.copy()
    for col in ["Mkt Cap (USD)", "Treasury USD", "Total Liabilities", "Net Crypto NAV"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(fmt_abbrev)
    if "NAV per share" in df_display.columns:
        df_display["NAV per share"] = df_display["NAV per share"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "–")
    if "Share price USD" in df_display.columns:
        df_display["Share price USD"] = df_display["Share price USD"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "–")
    if "% of Mkt Cap" in df_display.columns:
        df_display["% of Mkt Cap"] = df_display["% of Mkt Cap"].apply(lambda x: f"{x:.2f}%")
    if "MNAV (x)" in df_display.columns:
        df_display["MNAV (x)"] = df_display["MNAV (x)"].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "–")
    st.dataframe(df_display, use_container_width=True)

# -------------------- Table (formatted) -------------------------
# Data table is rendered within the "Table" tab; see tab_table definition above.
