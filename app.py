import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# ==========================
# Load model sekali di awal
# ==========================
@st.cache_resource
def load_models():
    sentiment_clf = joblib.load("sentiment_model.pkl")
    embed_model    = joblib.load("embedding_model.pkl")
    label_encoder  = joblib.load("label_encoder.pkl")
    trend_model    = joblib.load("trend_model.pkl")
    last_trend_date = joblib.load("trend_last_date.pkl")
    return sentiment_clf, embed_model, label_encoder, trend_model, last_trend_date

sentiment_clf, embed_model, label_encoder, trend_model, last_trend_date = load_models()


# ==========================
# Prediksi satu teks
# ==========================
def predict_single_text(text):
    emb = embed_model.encode([text])
    pred = sentiment_clf.predict(emb)
    label = label_encoder.inverse_transform(pred)[0]
    return label


# ==========================
# Preprocessing dataset
# ==========================
def preprocess_data(df):
    required_cols = ["date", "text"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' wajib ada di dataset.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "text"])

    # Hitung sentimen jika belum ada
    if "sentiment_score" not in df.columns:
        text_list = df["text"].astype(str).tolist()
        embeddings = embed_model.encode(text_list)

        y_pred = sentiment_clf.predict(embeddings)
        labels = label_encoder.inverse_transform(y_pred)
        df["sentiment_label"] = labels

        mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
        df["sentiment_score"] = df["sentiment_label"].map(mapping)

    df["sentiment_score"] = df["sentiment_score"].astype(float)
    return df


# ==========================
# Buat TREND + fitur tambahan
# ==========================
def make_trend(df):

    # Interaction feature
    df["interaction_total"] = (
        df["like_count"] + df["retweet_count"] +
        df["reply_count"] + df["quote_count"]
    )
    df["interaction_log"] = np.log1p(df["interaction_total"])

    # Jika tidak ada cluster kolom, beri cluster default 1 (biar tidak error)
    if "cluster" not in df.columns:
        df["cluster"] = 1

    # Grouping harian
    trend = df.groupby(df["date"].dt.date).agg({
        "sentiment_score": "mean",
        "interaction_total": "mean",
        "interaction_log": "mean",
        "cluster": lambda x: x.mode()[0],
    }).reset_index()

    trend["date"] = pd.to_datetime(trend["date"])

    # Fitur waktu
    trend["t"] = trend["date"].map(datetime.toordinal)
    trend["month"] = trend["date"].dt.month
    trend["dayofweek"] = trend["date"].dt.dayofweek
    trend["is_weekend"] = trend["dayofweek"].isin([5,6]).astype(int)

    return trend



# ==========================
# FORECASTING SESUAI FITUR BARU
# ==========================
def forecast_trend_from_model(trend, horizon_days=90):
    import numpy as np
    import pandas as pd

    feature_cols = joblib.load("trend_features.pkl")
    model = trend_model

    # mulai dari data historis lengkap
    df_full = trend.copy().reset_index(drop=True)

    future_rows = []

    for i in range(horizon_days):
        last_row = df_full.iloc[-1]
        next_date = last_row["date"] + timedelta(days=1)

        new_data = {
            "date": next_date,
            "t": next_date.toordinal(),
            "month": next_date.month,
            "dayofweek": next_date.weekday(),
            "is_weekend": int(next_date.weekday() in [5,6]),
            "interaction_total": df_full["interaction_total"].mean(),
            "interaction_log": df_full["interaction_log"].mean(),
            "cluster": df_full["cluster"].mode()[0],
            "rolling_mean_7": df_full["sentiment_score"].tail(7).mean(),
            "rolling_mean_30": df_full["sentiment_score"].tail(30).mean(),
            "lag_1": df_full["sentiment_score"].iloc[-1],
            "lag_7": df_full["sentiment_score"].iloc[-7] if len(df_full) >= 7 else df_full["sentiment_score"].iloc[-1],
        }

        X_future = pd.DataFrame([new_data])[feature_cols]

        pred = model.predict(X_future)[0]

        new_data["sentiment_score"] = pred
        future_rows.append(new_data)

        df_full = pd.concat([df_full, pd.DataFrame([new_data])], ignore_index=True)

    forecast_df = pd.DataFrame(future_rows)
    return forecast_df[["date", "sentiment_score"]].rename(columns={
        "sentiment_score": "predicted_sentiment_score"
    })

# ==========================
# UI Streamlit
# ==========================
st.title("Bitcoin Market Sentiment Analysis (Twitter 2022–2023)")
st.balloons()

st.sidebar.header("Input Data")
option = st.sidebar.radio(
    "Pilih sumber data:",
    ["Gunakan dataset default", "Upload dataset baru"]
)

if option == "Upload dataset baru":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    uploaded_file = None

# Load CSV
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("Dataset baru berhasil diupload ✅")
else:
    df_raw = pd.read_csv("tweets.csv")
    st.info("Menggunakan dataset default: tweets.csv")

st.subheader("Sample Dataset")
st.dataframe(df_raw.head())


# ==========================
# Prediksi Sentimen Input Manual
# ==========================
st.subheader("Uji Sentimen Secara Langsung")

user_text = st.text_area("Masukkan teks tweet:")

if st.button("Prediksi Sentimen Teks"):
    if user_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        label = predict_single_text(user_text)
        st.success(f"Hasil prediksi sentimen: **{label}**")


# ==========================
# Proses Dataset
# ==========================
if st.button("Jalankan Analisis Sentimen & Tren"):
    try:
        df_clean = preprocess_data(df_raw)
        trend = make_trend(df_clean)

        st.session_state["trend"] = trend
        st.success("Analisis dataset selesai! 🎉")

    except Exception as e:
        st.error(f"Terjadi error: {e}")


# ==========================
# Tampilkan Tren + Forecast
# ==========================
if "trend" in st.session_state:

    trend = st.session_state["trend"]

    st.subheader("Tren Sentimen Historis")
    st.line_chart(trend.set_index("date")["sentiment_score"])

    st.write(f"Periode data: {trend['date'].min().date()} — {trend['date'].max().date()}")

    horizon = st.slider("Horizon prediksi (hari)", 30, 180, 90, step=15)

    forecast_df = forecast_trend_from_model(trend, horizon_days=horizon)

    st.subheader(f"Prediksi {horizon} Hari ke Depan")

    all_trend = pd.concat([
        trend[["date", "sentiment_score"]].rename(columns={"sentiment_score": "score"}),
        forecast_df.rename(columns={"predicted_sentiment_score": "score"})
    ], ignore_index=True)

    st.line_chart(all_trend.set_index("date")["score"])

    st.caption("⚠️ Model forecasting hanya prediksi statistik, bukan saran investasi.")
