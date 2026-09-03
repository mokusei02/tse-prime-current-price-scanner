from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).parent


@st.cache_data
def load_companies() -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "prime_companies.csv", dtype={"code": str})


@st.cache_data(ttl=3600, show_spinner=False)
def download_chunk(tickers: tuple[str, ...], start: date) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        start=start,
        end=date.today() + timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        timeout=30,
    )


def ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    if ticker in raw.columns.get_level_values(0):
        return raw[ticker].copy()
    if ticker in raw.columns.get_level_values(1):
        return raw.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def longest_below_streak(
    low: pd.Series, threshold: float, start: date, end: date
) -> tuple[int, date | None, date | None, float | None]:
    values = pd.to_numeric(low, errors="coerce").dropna()
    values = values[
        (values.index.date >= start) & (values.index.date <= end)
    ]
    below = values <= threshold
    if not below.any():
        return 0, None, None, None

    groups = below.ne(below.shift()).cumsum()
    best: tuple[int, date | None, date | None, float | None] = (0, None, None, None)
    for _, segment in values[below].groupby(groups[below]):
        first = pd.Timestamp(segment.index[0]).date()
        last = pd.Timestamp(segment.index[-1]).date()
        days = (last - first).days + 1
        if days > best[0]:
            best = (days, first, last, float(segment.min()))
    return best


def format_date(value: date | None) -> str:
    return value.strftime("%Y年%m月%d日") if value else "—"


def render_search_controls(key_prefix: str, company_count: int):
    start = st.date_input(
        "検索開始日", value=date(2015, 1, 1), key=f"{key_prefix}_start"
    )
    end = st.date_input(
        "検索終了日", value=date.today(), key=f"{key_prefix}_end"
    )
    require_three_years_value = st.checkbox(
        "上場から3年以上", value=True, key=f"{key_prefix}_three_years"
    )
    streak_days = st.selectbox(
        "下落日数",
        options=[30, 60, 90],
        index=0,
        format_func=lambda days: f"{days}日",
        key=f"{key_prefix}_days",
    )
    submitted = st.button(
        "全社を検索する",
        type="primary",
        width="stretch",
        key=f"{key_prefix}_run",
    )
    st.caption(f"対象：東証プライム（内国株式）{company_count:,}社")
    return start, end, require_three_years_value, streak_days, submitted


st.set_page_config(
    page_title="東証プライム下落期間スクリーナー",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    .st-key-mobile_filters { display: none; }
    @media (max-width: 768px) {
        .st-key-mobile_filters { display: block; }
        section[data-testid="stSidebar"] { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
companies = load_companies()

with st.container(key="mobile_filters"):
    st.header("検索条件")
    mobile_values = render_search_controls("mobile", len(companies))

with st.sidebar:
    st.header("検索条件")
    desktop_values = render_search_controls("desktop", len(companies))

if mobile_values[-1]:
    start_date, end_date, require_three_years, max_streak_days, run = mobile_values
else:
    start_date, end_date, require_three_years, max_streak_days, run = desktop_values

title_days = str(max_streak_days)
st.title(f"現在の株価から{title_days}日以上下落しなかった銘柄【東証プライム】")
st.caption(
    f"東証プライム全社から、現在株価以下だった最長の連続期間が"
    f"{title_days}日以内の企業を探します。"
)

if run:
    if start_date > end_date:
        st.error("検索開始日は検索終了日以前にしてください。")
        st.stop()

    progress = st.progress(0, text="検索を開始しています…")
    status = st.empty()
    results: list[dict] = []
    failed: list[str] = []
    too_new_count = 0
    chunk_size = 40
    records = companies.to_dict("records")
    listing_cutoff = (pd.Timestamp(date.today()) - pd.DateOffset(years=3)).date()
    history_start = min(start_date, listing_cutoff - timedelta(days=10))

    for offset in range(0, len(records), chunk_size):
        chunk = records[offset : offset + chunk_size]
        tickers = tuple(f"{item['code']}.T" for item in chunk)
        try:
            raw = download_chunk(tickers, history_start)
        except Exception:
            raw = pd.DataFrame()

        for item, ticker in zip(chunk, tickers):
            frame = ticker_frame(raw, ticker)
            if frame.empty or "Close" not in frame or "Low" not in frame:
                failed.append(item["code"])
                continue

            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if close.empty:
                failed.append(item["code"])
                continue
            if require_three_years and close.index.min().date() > listing_cutoff:
                too_new_count += 1
                continue
            current_price = float(close.iloc[-1])
            days, first, last, minimum = longest_below_streak(
                frame["Low"], current_price, start_date, end_date
            )
            if 1 <= days <= max_streak_days:
                results.append(
                    {
                        "証券コード": item["code"],
                        "企業名": item["name"],
                        "現在株価（円）": round(current_price),
                        "最長日数": days,
                        "開始日": format_date(first),
                        "期間中最安値（円）": round(minimum) if minimum is not None else None,
                    }
                )

        completed = min(offset + len(chunk), len(records))
        progress.progress(
            completed / len(records),
            text=f"{completed:,} / {len(records):,}社を確認しました",
        )
        status.caption(
            f"該当企業：{len(results):,}社　上場3年未満：{too_new_count:,}社　"
            f"取得失敗：{len(failed):,}社"
        )
        time.sleep(0.2)

    progress.empty()
    result = pd.DataFrame(results)
    if result.empty:
        st.warning("条件に一致する企業はありませんでした。")
    else:
        result = result.sort_values(
            ["最長日数", "証券コード"], ascending=[False, True]
        ).reset_index(drop=True)
        result["下落率（%）"] = (
            (result["期間中最安値（円）"] / result["現在株価（円）"] - 1) * 100
        ).round(1)
        st.success(f"条件に一致した企業：{len(result):,}社")
        display_result = result.copy()
        lowest_price_column = "期間中最安値・下落率"
        display_result[lowest_price_column] = result.apply(
            lambda row: (
                f"{int(row['期間中最安値（円）']):,}円"
                f"（{row['下落率（%）']:.1f}%）"
            ),
            axis=1,
        )
        display_result = display_result.drop(
            columns=["期間中最安値（円）", "下落率（%）"]
        )
        cell_styles = pd.DataFrame(
            "", index=display_result.index, columns=display_result.columns
        )
        lowest_price_is_10_percent_lower = (
            result["期間中最安値（円）"] < result["現在株価（円）"] * 0.9
        )
        cell_styles.loc[
            lowest_price_is_10_percent_lower, lowest_price_column
        ] = "color: #DC2626; font-weight: 700;"
        styled_result = display_result.style.apply(lambda _: cell_styles, axis=None)
        st.dataframe(
            styled_result,
            hide_index=True,
            width="stretch",
            height=min(800, 38 * (len(result) + 1) + 4),
            column_config={
                "証券コード": st.column_config.TextColumn(width="small"),
                "企業名": st.column_config.TextColumn(width="medium"),
                "現在株価（円）": st.column_config.NumberColumn(format="%d円"),
                "最長日数": st.column_config.NumberColumn(format="%d日"),
                lowest_price_column: st.column_config.TextColumn(width="medium"),
            },
        )
        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "結果をCSVで保存",
            csv,
            f"tse_prime_under_current_price_{max_streak_days}days.csv",
            "text/csv",
        )

    if failed:
        with st.expander(f"取得できなかった企業（{len(failed):,}社）"):
            st.write("、".join(failed))

st.divider()
st.markdown("制作者：木星在住　[Twitter](https://x.com/mokuseidayo)")

