
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import date, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================
st.set_page_config(
    page_title="StockLens - Dự báo cổ phiếu",
    page_icon="📈",
    layout="wide"
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #312e81);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {font-size: 2.2rem; margin-bottom: 8px;}
.hero p {font-size: 1rem; opacity: 0.9;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.3);
    padding: 15px;
    border-radius: 12px;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}
</style>

<div class="hero">
    <h1>📈 StockLens</h1>
    <p>PHÂN TÍCH VÀ DỰ BÁO GIÁ CỔ PHIẾU BẰNG DEEP LEARNING</p>
    <p>Dự báo giá đóng cửa (Close) bằng mô hình LSTM</p>
</div>
""", unsafe_allow_html=True)


# =========================
# 2. TẢI DỮ LIỆU CỔ PHIẾU
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_stock(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=(end_date + timedelta(days=1)),
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    # Xử lý cả trường hợp Yahoo Finance trả về MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = [
            col for col in data.columns
            if str(col[0]).lower() == "close"
        ]
        if not close_cols:
            return pd.DataFrame()
        close = data[close_cols[0]]
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    }).dropna()

    result.index = pd.to_datetime(result.index)

    try:
        result.index = result.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    return result.sort_index()


# =========================
# 3. TẠO DỮ LIỆU CHUỖI
# =========================
def create_sequences(data, window=60):
    X, y = [], []

    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])

    X = np.array(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.array(y, dtype=np.float32)

    return X, y


# =========================
# 4. KIẾN TRÚC LSTM
# =========================
def build_model(window=60):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# =========================
# 5. HUẤN LUYỆN VÀ DỰ BÁO
# =========================
def run_analysis(df, horizon, epochs):
    window = 60
    values = df[["Close"]].values.astype(np.float32)
    n = len(values)

    if n < 150:
        raise ValueError(
            "Dữ liệu có ít hơn 150 phiên. "
            "Hãy chọn khoảng thời gian dài hơn."
        )

    # Chia theo thời gian, không xáo trộn dữ liệu
    split = int(n * 0.8)

    # Đánh giá trên 20% dữ liệu cuối
    scaler_test = MinMaxScaler()
    scaler_test.fit(values[:split])

    scaled_all = scaler_test.transform(values)
    train_scaled = scaled_all[:split]

    X_train, y_train = create_sequences(train_scaled, window)

    if len(X_train) < 20:
        raise ValueError("Không đủ dữ liệu để huấn luyện LSTM.")

    val_size = max(1, int(len(X_train) * 0.1))

    model_test = build_model(window)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model_test.fit(
        X_train[:-val_size],
        y_train[:-val_size],
        validation_data=(
            X_train[-val_size:],
            y_train[-val_size:]
        ),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Tạo đầu vào kiểm tra với 60 phiên trước ngày bắt đầu test
    test_input = scaled_all[split-window:]
    X_test, y_test = create_sequences(test_input, window)

    test_pred_scaled = model_test.predict(X_test, verbose=0)

    test_pred = scaler_test.inverse_transform(
        test_pred_scaled
    ).ravel()

    actual = scaler_test.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))
    mae = float(mean_absolute_error(actual, test_pred))

    r2 = (
        float(r2_score(actual, test_pred))
        if len(actual) > 1 else float("nan")
    )

    # Tỷ lệ đúng chiều tăng/giảm giữa các phiên liên tiếp
    if len(actual) > 2:
        direction = float(
            np.mean(
                np.sign(np.diff(actual)) ==
                np.sign(np.diff(test_pred))
            ) * 100
        )
    else:
        direction = float("nan")

    # Huấn luyện mô hình dự báo tương lai bằng toàn bộ dữ liệu
    scaler_future = MinMaxScaler()
    scaled = scaler_future.fit_transform(values)

    X_all, y_all = create_sequences(scaled, window)

    val2 = max(1, int(len(X_all) * 0.1))
    model_future = build_model(window)

    early_stop_future = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history_future = model_future.fit(
        X_all[:-val2],
        y_all[:-val2],
        validation_data=(
            X_all[-val2:],
            y_all[-val2:]
        ),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop_future],
        verbose=0
    )

    # Dự báo lần lượt từng phiên kế tiếp
    last_window = scaled[-window:].reshape(1, window, 1)
    future_scaled = []

    for _ in range(horizon):
        next_value = model_future.predict(
            last_window, verbose=0
        )[0, 0]

        future_scaled.append(next_value)

        last_window = np.concatenate([
            last_window[:, 1:, :],
            np.array(next_value, dtype=np.float32).reshape(1, 1, 1)
        ], axis=1)

    future_prices = scaler_future.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).ravel()

    return {
        "actual": actual,
        "test_pred": test_pred,
        "test_dates": df.index[split:],
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "direction": direction,
        "future": future_prices,
        "latest": float(values[-1, 0]),
        "latest_date": df.index[-1],
        "history": history_future.history
    }


# =========================
# 6. GIAO DIỆN NHẬP THÔNG TIN
# =========================
st.subheader("⚙️ Thiết lập dự báo")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        ticker = st.text_input(
            "Mã cổ phiếu",
            value="VNM.VN",
            help="Ví dụ: VNM.VN, FPT.VN, VIC.VN, AAPL"
        ).strip().upper()

        start_date = st.date_input(
            "Ngày bắt đầu lấy dữ liệu",
            value=date(date.today().year - 8, 1, 1),
            max_value=date.today()
        )

    with col2:
        end_date = st.date_input(
            "Ngày kết thúc lấy dữ liệu",
            value=date.today(),
            min_value=start_date,
            max_value=date.today()
        )

        horizon = st.select_slider(
            "Số phiên muốn dự báo",
            options=[5, 10, 15, 20, 30],
            value=10
        )

    epochs = st.select_slider(
        "Số epoch huấn luyện tối đa",
        options=[10, 15, 20, 25, 30],
        value=20
    )

    run_button = st.button(
        "🚀 BẮT ĐẦU PHÂN TÍCH VÀ DỰ BÁO",
        type="primary",
        use_container_width=True
    )


if "stock_result" not in st.session_state:
    st.session_state.stock_result = None


if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu.")
        st.stop()

    try:
        with st.spinner(
            "Đang tải dữ liệu và huấn luyện mô hình LSTM..."
        ):
            df = load_stock(ticker, start_date, end_date)

            if df.empty:
                raise ValueError(
                    "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu."
                )

            result = run_analysis(df, int(horizon), int(epochs))

            st.session_state.stock_result = {
                "df": df,
                "result": result,
                "ticker": ticker,
                "horizon": int(horizon)
            }

    except Exception as e:
        st.error(f"Không thể thực hiện dự báo: {e}")
        st.stop()


# =========================
# 7. HIỂN THỊ KẾT QUẢ
# =========================
if st.session_state.stock_result is None:
    st.info(
        "Nhập mã cổ phiếu, chọn khoảng thời gian và bấm "
        "**Bắt đầu phân tích và dự báo**."
    )
    st.stop()

saved = st.session_state.stock_result
df = saved["df"]
r = saved["result"]
ticker = saved["ticker"]
horizon = saved["horizon"]

st.divider()
st.subheader(f"📌 Tổng quan cổ phiếu {ticker}")

st.caption(
    f"Dữ liệu từ {df.index.min():%d/%m/%Y} "
    f"đến {df.index.max():%d/%m/%Y} · "
    f"{len(df):,} phiên giao dịch"
)

last_price = r["latest"]
future = r["future"]
change_pct = (future[-1] / last_price - 1) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Close gần nhất",
    f"{last_price:,.2f}"
)

c2.metric(
    "Dự báo phiên +1",
    f"{future[0]:,.2f}",
    f"{(future[0]/last_price-1)*100:+.2f}%"
)

c3.metric(
    f"Dự báo phiên +{horizon}",
    f"{future[-1]:,.2f}",
    f"{change_pct:+.2f}%"
)

c4.metric(
    "Số phiên dự báo",
    f"{horizon} phiên"
)


# =========================
# 8. CHỈ SỐ ĐÁNH GIÁ
# =========================
st.subheader("📊 Đánh giá mô hình trên tập kiểm tra")

c1, c2, c3, c4 = st.columns(4)

c1.metric("RMSE", f"{r['rmse']:,.2f}")
c2.metric("MAE", f"{r['mae']:,.2f}")
c3.metric("R² Score", f"{r['r2']:.4f}")
c4.metric(
    "Đúng chiều biến động",
    f"{r['direction']:.1f}%"
)

st.caption(
    "RMSE và MAE đo mức sai số giá; R² đo độ phù hợp trên tập kiểm tra. "
    "Tỷ lệ đúng chiều chỉ so sánh hướng biến động giữa các phiên liên tiếp."
)


# =========================
# 9. BIỂU ĐỒ GIÁ
# =========================
st.subheader(
    f"📈 Giá Close thực tế và dự báo {horizon} phiên tiếp theo"
)

view = st.radio(
    "Phạm vi hiển thị",
    ["180 phiên gần nhất", "Toàn bộ dữ liệu"],
    horizontal=True
)

chart_df = df.tail(180) if view == "180 phiên gần nhất" else df

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(
    chart_df.index,
    chart_df["Close"],
    label="Giá Close thực tế",
    linewidth=1.8
)

test_dates = pd.to_datetime(r["test_dates"])
mask = test_dates >= chart_df.index.min()

if np.any(mask):
    ax.plot(
        test_dates[mask],
        r["test_pred"][mask],
        label="Dự báo trên tập kiểm tra",
        linewidth=1.5
    )

# Ngày làm việc chỉ để minh họa; không phải lịch nghỉ giao dịch chính thức
future_dates = pd.bdate_range(
    start=pd.Timestamp(r["latest_date"]) + pd.offsets.BDay(1),
    periods=horizon
)

ax.plot(
    [pd.Timestamp(r["latest_date"])] + list(future_dates),
    [last_price] + list(future),
    color="red",
    marker="o",
    markersize=4,
    linewidth=2,
    label="Dự báo tương lai"
)

ax.axvline(
    pd.Timestamp(r["latest_date"]),
    linestyle="--",
    alpha=0.5
)

ax.set_xlabel("Ngày")
ax.set_ylabel("Giá Close")
ax.grid(True, alpha=0.25)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# =========================
# 10. BẢNG DỰ BÁO CHI TIẾT
# =========================
st.subheader(f"🗓️ Bảng dự báo {horizon} phiên tiếp theo")

price_changes = np.diff(np.r_[last_price, future])

forecast_table = pd.DataFrame({
    "Phiên": [f"Ngày +{i}" for i in range(1, horizon + 1)],
    "Ngày dự kiến": [
        d.strftime("%d/%m/%Y") for d in future_dates
    ],
    "Giá Close dự báo": np.round(future, 2),
    "Chênh lệch so với phiên trước": np.round(price_changes, 2),
    "Thay đổi so với Close gần nhất (%)": np.round(
        (future / last_price - 1) * 100, 2
    )
})

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Tải bảng dự báo CSV",
    data=forecast_table.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================
# 11. BIỂU ĐỒ LOSS
# =========================
with st.expander("🧪 Chi tiết quá trình huấn luyện LSTM"):
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        r["history"].get("loss", []),
        label="Train Loss"
    )

    ax2.plot(
        r["history"].get("val_loss", []),
        label="Validation Loss"
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()

    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("""
    **Cấu trúc mô hình**
    - Đầu vào: 60 phiên Close gần nhất.
    - LSTM tầng 1: 64 units.
    - LSTM tầng 2: 32 units.
    - Dense: 16 units và đầu ra 1 giá trị Close.
    - Chuẩn hóa: MinMaxScaler.
    - Tối ưu: Adam, hàm mất mát MSE.
    - Early Stopping: dừng sớm khi Validation Loss không cải thiện.
    """)


# =========================
# 12. LƯU Ý
# =========================
st.warning("""
**Lưu ý:** Đây là kết quả ước lượng từ dữ liệu lịch sử, không phải giá chắc chắn
sẽ xảy ra. Mô hình chỉ sử dụng giá Close, chưa xét khối lượng giao dịch,
tin tức, kết quả kinh doanh hoặc các yếu tố thị trường khác.
Dự báo nhiều phiên liên tiếp có thể tích lũy sai số.
""")

st.caption(
    "Nguồn dữ liệu: Yahoo Finance thông qua yfinance. "
    "Ngày dự kiến sử dụng ngày làm việc để minh họa; "
    "lịch giao dịch thực tế có thể khác."
)
