import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="상승초입탐지 v1.0", page_icon="📈", layout="wide")
st.title("📈 상승초입탐지 v1.0")
st.caption("기술적 분석, 거래량·수급, 시장 환경의 관측 가능한 53.33점을 100점으로 정규화합니다.")


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(data, period=14):
    up = data["High"].diff()
    down = -data["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=data.index)
    prev = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di


@st.cache_data(ttl=3600, show_spinner=False)
def download_weekly(ticker, years):
    period_map = {1: "2y", 3: "5y", 5: "10y"}
    data = yf.download(
        ticker,
        period=period_map[years],
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    return data.loc[data.index >= cutoff]


def indicators(data):
    x = data.copy()
    x["EMA4"] = x["Close"].ewm(span=4, adjust=False).mean()
    x["EMA12"] = x["Close"].ewm(span=12, adjust=False).mean()
    x["SMA40"] = x["Close"].rolling(40).mean()
    x["RSI14"] = rsi(x["Close"])
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]
    x["ADX"], x["PLUS_DI"], x["MINUS_DI"] = adx(x)
    mid = x["Close"].rolling(20).mean()
    std = x["Close"].rolling(20).std()
    x["BB_UPPER"] = mid + 2 * std
    x["BB_LOWER"] = mid - 2 * std
    x["BB_WIDTH"] = (x["BB_UPPER"] - x["BB_LOWER"]) / mid
    direction = np.sign(x["Close"].diff()).fillna(0)
    x["OBV"] = (direction * x["Volume"]).cumsum()
    x["OBV_EMA8"] = x["OBV"].ewm(span=8, adjust=False).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["REL_VOL"] = x["Volume"] / x["VOL_MA20"]
    x["HIGH20_PREV"] = x["High"].rolling(20).max().shift(1)
    x["RET12"] = x["Close"].pct_change(12)
    return x


def calculate_scores(stock, benchmark):
    stock = indicators(stock)
    benchmark = indicators(benchmark)

    tech = pd.Series(0.0, index=stock.index)
    tech += np.where(stock["Close"] > stock["EMA4"], 1.5, 0)
    tech += np.where(stock["EMA4"] > stock["EMA12"], 2.0, 0)
    tech += np.where(stock["Close"] > stock["SMA40"], 1.5, 0)
    tech += np.where(stock["EMA4"].diff(3) > 0, 1.0, 0)
    tech += np.where(stock["Close"] > stock["Close"].shift(4), 1.5, 0)
    tech += np.where(stock["Close"] > stock["HIGH20_PREV"], 5.0,
                     np.where(stock["Close"] >= stock["HIGH20_PREV"] * 0.97, 2.5, 0))
    tech += np.where((stock["RSI14"] >= 50) & (stock["RSI14"] <= 70), 2.0, 0)
    tech += np.where((stock["RSI14"] > stock["RSI14"].shift(1)) & (stock["RSI14"] >= 40), 1.5, 0)
    tech -= np.where(stock["RSI14"] > 78, 1.0, 0)
    tech += np.where(stock["MACD"] > stock["MACD_SIGNAL"], 2.0, 0)
    tech += np.where(stock["MACD_HIST"] > stock["MACD_HIST"].shift(1), 1.0, 0)
    tech += np.where(stock["MACD"] > 0, 1.0, 0)
    tech += np.where(stock["PLUS_DI"] > stock["MINUS_DI"], 1.25, 0)
    tech += np.where(stock["ADX"] >= 20, 1.25, 0)
    squeeze = stock["BB_WIDTH"] <= stock["BB_WIDTH"].rolling(20).quantile(0.30)
    tech += np.where(squeeze, 1.0, 0)
    tech += np.where(stock["Close"] > stock["BB_UPPER"], 2.0, 0)
    body = (stock["Close"] - stock["Open"]).abs()
    price_range = (stock["High"] - stock["Low"]).replace(0, np.nan)
    tech += np.where((stock["Close"] > stock["Open"]) & ((body / price_range) >= 0.60), 1.5, 0)

    q_close = benchmark["Close"].reindex(stock.index).ffill()
    q_ret12 = q_close.pct_change(12)
    relative = pd.Series(0.0, index=stock.index)
    relative += np.where(stock["RET12"] > q_ret12, 1.83, 0)
    relative += np.where((stock["RET12"] - q_ret12).diff(2) > 0, 1.0, 0)
    tech = (tech + relative).clip(0, 33.33)

    volume = pd.Series(0.0, index=stock.index)
    volume += np.select(
        [stock["REL_VOL"] >= 2.0, stock["REL_VOL"] >= 1.5,
         stock["REL_VOL"] >= 1.2, stock["REL_VOL"] >= 0.8],
        [3.0, 2.4, 1.8, 1.0], default=0)
    breakout = stock["Close"] > stock["HIGH20_PREV"]
    volume += np.where(breakout & (stock["REL_VOL"] >= 1.5), 3.0, 0)
    volume += np.where(breakout & (stock["REL_VOL"] >= 1.1) & (stock["REL_VOL"] < 1.5), 1.5, 0)
    volume += np.where(stock["OBV"] > stock["OBV_EMA8"], 1.0, 0)
    volume += np.where(stock["OBV"] > stock["OBV"].shift(4), 1.0, 0)
    typical = (stock["High"] + stock["Low"] + stock["Close"]) / 3
    vwap8 = (typical * stock["Volume"]).rolling(8).sum() / stock["Volume"].rolling(8).sum()
    volume += np.where(stock["Close"] > vwap8, 1.5, 0)
    volume += np.where((stock["Close"] > stock["Open"]) & (stock["REL_VOL"] >= 1.5), 1.83, 0)
    volume = volume.clip(0, 13.33)

    q_ema12 = q_close.ewm(span=12, adjust=False).mean()
    q_sma40 = q_close.rolling(40).mean()
    q_rsi = rsi(q_close)
    market = pd.Series(0.0, index=stock.index)
    market += np.where(q_close > q_ema12, 1.50, 0)
    market += np.where(q_close > q_sma40, 1.25, 0)
    market += np.where((q_rsi >= 45) & (q_rsi <= 70), 1.00, 0)
    market += np.where(q_ema12.diff(3) > 0, 0.75, 0)
    market += np.where(q_ret12 > 0, 1.00, 0)
    market += np.where(q_close.pct_change().rolling(4).std() < 0.06, 0.67, 0)
    market += np.where(q_close.pct_change().abs() < 0.08, 0.50, 0)
    market = market.clip(0, 6.67)

    result = stock[["Open", "High", "Low", "Close", "Volume"]].copy()
    result["기술적"] = tech
    result["거래량수급"] = volume
    result["시장환경"] = market
    result["관측원점수"] = tech + volume + market
    result["정규화점수"] = result["관측원점수"] / 53.33 * 100
    result["4주후수익률"] = result["Close"].shift(-4) / result["Close"] - 1
    result["8주후수익률"] = result["Close"].shift(-8) / result["Close"] - 1
    return result.dropna(subset=["정규화점수"])


def grade(score):
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    if score >= 40: return "E"
    return "F"


def signal(score):
    if score >= 80: return "강한 상승초입 후보"
    if score >= 70: return "상승초입 후보"
    if score >= 60: return "관찰 구간"
    return "대기 구간"


with st.sidebar:
    st.header("분석 설정")
    ticker = st.text_input("미국 주식 종목코드", value="TSLA", help="예: TSLA, NVDA, AAPL, BMNR, CRCL").strip().upper()
    years = st.selectbox("분석 기간", [1, 3, 5], index=1, format_func=lambda x: f"{x}년")
    run = st.button("분석 실행", type="primary", use_container_width=True)
    st.divider()
    st.caption("학습·연구용 도구이며 투자 권유가 아닙니다. 데이터 지연·누락 가능성이 있습니다.")

if not run:
    st.info("왼쪽에서 종목코드를 입력하고 **분석 실행** 버튼을 누르세요.")
    st.stop()

if not ticker:
    st.error("종목코드를 입력해 주세요.")
    st.stop()

with st.spinner(f"{ticker} 데이터를 불러오고 점수를 계산하고 있습니다..."):
    try:
        stock = download_weekly(ticker, years)
        qqq = download_weekly("QQQ", years)
        if stock.empty:
            st.error("종목 데이터를 불러오지 못했습니다. 종목코드를 확인해 주세요.")
            st.stop()
        if qqq.empty:
            st.error("시장 기준 데이터(QQQ)를 불러오지 못했습니다.")
            st.stop()
        result = calculate_scores(stock, qqq)
    except Exception as exc:
        st.error("데이터를 불러오는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        with st.expander("오류 내용 보기"):
            st.code(str(exc))
        st.stop()

if result.empty:
    st.warning("분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()

latest = result.iloc[-1]
score = float(latest["정규화점수"])

st.subheader(f"{ticker} 현재 분석 결과")
c1, c2, c3, c4 = st.columns(4)
c1.metric("현재 점수", f"{score:.1f}점")
c2.metric("등급", grade(score))
c3.metric("신호", signal(score))
c4.metric("최근 주간 종가", f"${latest['Close']:,.2f}")

s1, s2, s3 = st.columns(3)
s1.metric("기술적", f"{latest['기술적']:.2f} / 33.33")
s2.metric("거래량·수급", f"{latest['거래량수급']:.2f} / 13.33")
s3.metric("시장 환경", f"{latest['시장환경']:.2f} / 6.67")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.58, 0.42], subplot_titles=(f"{ticker} 주간 가격", "상승초입탐지 점수"))
fig.add_trace(go.Candlestick(x=result.index, open=result["Open"], high=result["High"],
                             low=result["Low"], close=result["Close"], name="주가"), row=1, col=1)
fig.add_trace(go.Bar(x=result.index, y=result["정규화점수"], name="ATS 점수"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", annotation_text="후보 70", row=2, col=1)
fig.add_hline(y=80, line_dash="dash", annotation_text="강한 후보 80", row=2, col=1)
fig.update_yaxes(title_text="가격(USD)", row=1, col=1)
fig.update_yaxes(title_text="점수", range=[0, 100], row=2, col=1)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_layout(height=850, margin=dict(l=20, r=20, t=80, b=20), hovermode="x unified", legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.subheader("과거 신호 성과")
rows = []
for threshold in [70, 80]:
    signals = result[result["정규화점수"] >= threshold]
    for horizon, column in [(4, "4주후수익률"), (8, "8주후수익률")]:
        returns = signals[column].dropna()
        if len(returns):
            rows.append({
                "기준": f"{threshold}점 이상",
                "평가 기간": f"{horizon}주 후",
                "신호 수": len(returns),
                "상승 확률": f"{(returns > 0).mean() * 100:.1f}%",
                "평균 수익률": f"{returns.mean() * 100:.2f}%",
                "중앙값 수익률": f"{returns.median() * 100:.2f}%",
            })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("해당 기간에는 70점 이상의 완결된 과거 신호가 없습니다.")

with st.expander("최근 주간 점수 자세히 보기"):
    table = result[["Close", "기술적", "거래량수급", "시장환경", "정규화점수", "4주후수익률", "8주후수익률"]].tail(20).copy()
    table.index = table.index.strftime("%Y-%m-%d")
    st.dataframe(table.round(2), use_container_width=True)

st.download_button("분석 결과 CSV 내려받기", result.to_csv().encode("utf-8-sig"),
                   file_name=f"{ticker}_ats_v1_result.csv", mime="text/csv")
st.caption("기술적 33.33점 + 거래량·수급 13.33점 + 시장 환경 6.67점 = 53.33점을 100점으로 환산합니다.")
