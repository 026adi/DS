import streamlit as st
import pandas as pd
import joblib
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
# Helper: prediksi sentimen satu teks
# ==========================
def predict_single_text(text):
    emb = embed_model.encode([text])
    pred = sentiment_clf.predict(emb)
    label = label_encoder.inverse_transform(pred)[0]
    return label

# ==========================
# Helper: preprocessing & prediksi dataset
# ==========================
def preprocess_data(df):
    # Pastikan kolom penting ada
    required_cols = ["date", "text"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' wajib ada di dataset.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "text"])

    # Kalau belum ada sentiment_score / sentiment_label, kita hitung sendiri
    if "sentiment_score" not in df.columns or "sentiment_label" not in df.columns:
        text_list = df["text"].astype(str).tolist()
        embeddings = embed_model.encode(text_list)

        # Prediksi label
        y_pred = sentiment_clf.predict(embeddings)
        labels = label_encoder.inverse_transform(y_pred)
        df["sentiment_label"] = labels

        mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
        df["sentiment_score"] = df["sentiment_label"].map(mapping)
    else:
        df["sentiment_score"] = df["sentiment_score"].astype(float)

    return df

def make_trend(df):
    trend = df.groupby(df["date"].dt.date)["sentiment_score"].mean().reset_index()
    trend["date"] = pd.to_datetime(trend["date"])
    return trend

def forecast_trend_from_model(trend, horizon_days=90):
    from datetime import datetime

    last_date_user = trend["date"].max()
    future_dates = pd.date_range(last_date_user + timedelta(days=1),
                                 periods=horizon_days, freq="D")

    t_future = future_dates.map(datetime.toordinal).to_frame(name="t")
    y_future = trend_model.predict(t_future)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_sentiment_score": y_future
    })
    return forecast_df

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

# Load data
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("Dataset baru berhasil diupload ✅")
else:
    df_raw = pd.read_csv("tweets.csv")
    st.info("Menggunakan dataset default: tweets.csv")

st.subheader("Sample Dataset")
st.dataframe(df_raw.head())

# ==========================
# 🔍 FITUR: Prediksi Sentimen Input Manual
# ==========================
st.subheader("Uji Sentimen Secara Langsung")

user_text = st.text_area("Masukkan teks tweet atau opini tentang Bitcoin:")

if st.button("Prediksi Sentimen Teks"):
    if user_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        label = predict_single_text(user_text)
        st.success(f"Hasil prediksi sentimen: **{label}**")

# ==========================
# 🚀 Proses Dataset
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
# 📈 TAMPILKAN HASIL JIKA SUDAH DIANALISIS
# ==========================
if "trend" in st.session_state:
    trend = st.session_state["trend"]

    st.subheader("Tren Sentimen Historis")
    st.line_chart(trend.set_index("date")["sentiment_score"])
    st.write(f"Periode data: {trend['date'].min().date()} s.d. {trend['date'].max().date()}")

    horizon = st.slider("Horizon prediksi (hari ke depan)", 30, 120, 90, step=15)

    forecast_df = forecast_trend_from_model(trend, horizon_days=horizon)

    st.subheader(f"Prediksi Tren Sentimen {horizon} Hari ke Depan")

    all_trend = pd.concat([
        trend[["date", "sentiment_score"]].rename(columns={"sentiment_score": "score"}),
        forecast_df.rename(columns={"predicted_sentiment_score": "score"})
    ], ignore_index=True)

    st.line_chart(all_trend.set_index("date")["score"])
    st.caption("⚠️ Model forecasting berbasis pola historis, bukan saran investasi.")
