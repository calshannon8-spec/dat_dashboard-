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
st.set_page_config(page_title="DAT Dashboard", page_icon="🟠")

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

try:
    prices = fetch_prices()
    # Build stress-test sliders in the sidebar for BTC & ETH (exclude USDC since it's pegged)
    with st.sidebar:
        st.markdown("### Stress test: price shocks")
        shock_syms = [sym for sym in prices.keys() if sym.upper() != 'USDC']
        shock_pcts = {}
        for sym in shock_syms:
            shock_pcts[sym] = st.slider(
                f"{sym} shock",
                min_value=-50,
                max_value=200,
                value=0,
                step=5,
                format="%d%%",
                key=f"shock_{sym}"
            )
        # Reset shocks button
        if st.button("Reset shocks"):
            for sym in shock_syms:
                st.session_state[f"shock_{sym}"] = 0
            st.experimental_rerun()
        # Effective prices: apply shocks to BTC/ETH; keep USDC unchanged
        prices_eff = {}
        for sym in prices.keys():
            if sym in shock_pcts:
                prices_eff[sym] = prices[sym] * (1 + shock_pcts[sym] / 100.0)
            else:
                prices_eff[sym] = prices[sym]
    # Display full dollar amounts (no decimals) for BTC and ETH
    cols = st.columns(len([s for s in prices.keys() if s.upper() != 'USDC']))
    # Map symbol to column index
    idx = 0
    for sym in [sym for sym in prices.keys() if sym.upper() != 'USDC']:
        # Use full number formatting with comma separators and no decimals
        cols[idx].metric(f"{sym}", f"${prices_eff[sym]:,.0f}", delta=f"{shock_pcts.get(sym, 0)}%")
        idx += 1
    st.caption(
        "Last updated: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        + " · Calculations below use shocked prices for BTC/ETH"
    )
except Exception as e:
    # Fall back to zero prices and no shocks on failure
    prices = {"BTC": 0.0, "ETH": 0.0, "USDC": 0.0}
    shock_pcts = {"BTC": 0, "ETH": 0}
    prices_eff = prices.copy()
    st.error("Couldn't load prices.")
    st.exception(e)

st.divider()
st.header("Wallet Balances (ETH Mainnet)")
with st.sidebar:
    st.subheader("API / Wallet Settings")
    covalent_api_key = st.text_input("Covalent API Key", type="password")
    eth_address = st.text_input("ETH Address (0x...)", placeholder="e.g., 0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
    fetch_btn = st.button("Fetch Balances")

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

st.divider()
st.header("Company Screener (from CSV)")

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
# Use shocked effective prices (prices_eff) when building the data rows, so treasury and NAV calculations
# reflect any price shocks applied via the sliders. prices_eff falls back to the live prices if no shocks
# are set.
rows = [compute_row(c, prices_eff) for c in enriched]
df = pd.DataFrame(rows)
df = add_mnav(df)
df = df.sort_values(by="% of Mkt Cap", ascending=False).reset_index(drop=True)

# -------------------- Filters / KPIs / Charts -------------------

st.sidebar.markdown("### Filters")
if "exchange" in df.columns:
    _ex_opts = sorted([e for e in df["exchange"].dropna().unique().tolist() if e != ""])
    _ex_sel = st.sidebar.multiselect("Exchange", _ex_opts, default=_ex_opts or [])
    df_view = df[df["exchange"].isin(_ex_sel)] if _ex_sel else df.copy()
else:
    df_view = df.copy()

_mc_min = float(df_view["Mkt Cap (USD)"].min() if not df_view.empty else 0.0)
_mc_max = float(df_view["Mkt Cap (USD)"].max() if not df_view.empty else 0.0)
# Determine an appropriate unit for the Market Cap filter slider based on the maximum value.
if _mc_max >= 1e9:
    _factor = 1e9
    _unit_label = "billions"
elif _mc_max >= 1e6:
    _factor = 1e6
    _unit_label = "millions"
elif _mc_max >= 1e3:
    _factor = 1e3
    _unit_label = "thousands"
else:
    _factor = 1.0
    _unit_label = ""
_slider_label = f"Market Cap (USD{', ' + _unit_label if _unit_label else ''})"
_lo, _hi = st.sidebar.slider(
    _slider_label,
    min_value=0.0,
    max_value=max(1.0, _mc_max / _factor),
    value=(0.0, max(1.0, _mc_max / _factor))
)
df_view = df_view[
    (df_view["Mkt Cap (USD)"] >= _lo * _factor) & (df_view["Mkt Cap (USD)"] <= _hi * _factor)
]

st.subheader("Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Treasury", f"${np.nansum(df_view['Treasury USD']):,.0f}")
c2.metric("Total Liabilities", f"${np.nansum(df_view['Total Liabilities']):,.0f}")
c3.metric("Net NAV (sum)", f"${(np.nansum(df_view['Treasury USD']) - np.nansum(df_view['Total Liabilities'])):,.0f}")

st.subheader("Top Treasuries")
_top = df_view[["Ticker","name","Treasury USD"]].dropna().sort_values("Treasury USD", ascending=False).head(10)
if not _top.empty:
    if HAS_PLOTLY:
        fig = px.bar(_top, x="Ticker", y="Treasury USD", hover_data=["name","Treasury USD"], title="Top 10 by Treasury (USD)")
        fig.update_yaxes(title="Treasury (USD)", tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (alt.Chart(_top).mark_bar()
                 .encode(x="Ticker:N", y=alt.Y("Treasury USD:Q", title="Treasury (USD)"),
                         tooltip=["Ticker","name","Treasury USD"])
                 .properties(title="Top 10 by Treasury (USD)"))
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No rows available for ranking.")

st.subheader("Treasury % of Market Cap vs Market Cap")
_sc = df_view[["Ticker","name","Treasury USD","Mkt Cap (USD)","% of Mkt Cap"]].dropna()
if not _sc.empty:
    # convert % to fraction for plotting as percentage axis
    _sc = _sc.assign(pct=_sc["% of Mkt Cap"] / 100.0)
    if HAS_PLOTLY:
        # Linear scatter plot: x = Market Cap (USD), y = % of Market Cap
        fig2 = px.scatter(
            _sc,
            x="Mkt Cap (USD)",
            y="pct",
            size="Treasury USD",
            hover_data=["Ticker", "name"],
            title="Treasury % of Market Cap vs Market Cap",
        )
        fig2.update_xaxes(title="Market Cap (USD)", tickformat="~s")
        fig2.update_yaxes(title="% of Market Cap", tickformat=".2%")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        chart2 = (
            alt.Chart(_sc).mark_circle()
            .encode(
                x=alt.X("Mkt Cap (USD):Q", title="Market Cap (USD)"),
                y=alt.Y("pct:Q", title="% of Market Cap"),
                size="Treasury USD:Q",
                tooltip=["Ticker", "name", "Treasury USD", "Mkt Cap (USD)", "% of Mkt Cap"],
            )
        )
        st.altair_chart(chart2, use_container_width=True)
else:
    st.info("Not enough data for scatter.")

st.caption(f"Filtered rows: {len(df_view)} / {len(df)}")

# Provide a download button so users can export the currently filtered view as a CSV.
csv_data = df_view.to_csv(index=False)
st.sidebar.download_button(
    label="Download filtered CSV",
    data=csv_data,
    file_name="filtered_companies.csv",
    mime="text/csv",
)

# -------------------- Table (formatted) -------------------------

def fmt_abbrev(v):
    try: n = float(v)
    except Exception: return v
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000_000: return f"{sign}${n/1_000_000_000_000:,.2f}T"
    if n >= 1_000_000_000:     return f"{sign}${n/1_000_000_000:,.2f}B"
    if n >= 1_000_000:         return f"{sign}${n/1_000_000:,.2f}M"
    return f"{sign}${n:,.0f}"

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
