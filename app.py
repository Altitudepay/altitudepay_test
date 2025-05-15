import streamlit as st
import pandas as pd
import pickle
import json
import os
import base64
from datetime import datetime, timedelta
import altair as alt
from azure.storage.blob import BlobServiceClient
import tempfile
from dotenv import load_dotenv
from retrain_model import run_retraining_pipeline
load_dotenv()
from streamlit_autorefresh import st_autorefresh
from db_utils import run_bin_query, run_processor_query
# -------------------------
# 🎯 Page Configuration
# -------------------------
st.set_page_config(page_title="BIN Predictor Dashboard", layout="wide", page_icon="altitudepay.svg")

# -------------------------
# 🖼️ Web-style Brand Header with Logo
# -------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("altitudepaylogo.png")

st.markdown(f"""
    <style>
    .custom-header {{
        display: flex;
        align-items: center;
        padding: 12px 20px;
        background-color: #ffffff;
        border-bottom: 1px solid #eee;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        position: sticky;
        top: 0;
        z-index: 999;
    }}
    .custom-header img {{
        height: 40px;
    }}
    </style>
    <div class="custom-header">
        <img src="data:image/png;base64,{logo_base64}" alt="AltitudePay Logo">
    </div>
""", unsafe_allow_html=True)

# -------------------------
# 🧠 Load Model & Stats
# -------------------------
@st.cache_data(ttl=3600)
def get_last_month_bins():
    return run_bin_query()

last_month_bin_list = get_last_month_bins()
last_month_bin_list = [int(b) for b in last_month_bin_list]

@st.cache_data(ttl=3600)
def get_last_month_processors():
    return run_processor_query()
last_month_processor_list = get_last_month_processors()
last_month_processor_list = [str(p) for p in last_month_processor_list]

@st.cache_resource
def load_artifacts():
    # Always load the latest model
    model_path = "models/model_latest.pkl"
    stats_path = "stats/processor_success_stats_latest.pkl"
    processor_map_path = "processor_name_mapping.json"

    with open(model_path, "rb") as f_model, \
         open(stats_path, "rb") as f_stats, \
         open(processor_map_path, "r") as f_map:

        model = pickle.load(f_model)
        stats = pickle.load(f_stats)
        processor_name_map = json.load(f_map)
        reverse_map = {v: k for k, v in processor_name_map.items()}

    return model, stats, processor_name_map, reverse_map

model, stats, processor_name_map, reverse_processor_map = load_artifacts()
external_processors = {"TWP", "TWP (US)", "Fin - MID 01", "Npay", "Dreamzpay - Altitudepay"}

# -----------------------------
# Compute Global Fallback Averages
# -----------------------------
# Flatten stats to DataFrame for easy averaging
bin_proc_df = pd.DataFrame.from_dict(stats["bin_proc_stats"], orient="index")

global_fallback = {
    "bin_tx_count": pd.DataFrame.from_dict(stats["bin_tx"], orient="index")["bin_tx_count"].mean(),
    "bin_success_rate": pd.DataFrame.from_dict(stats["bin_success"], orient="index")["bin_success_rate"].mean(),
    "bin_processor_tx_count": bin_proc_df["bin_processor_tx_count"].mean(),
    "bin_processor_success_count": bin_proc_df["bin_processor_success_count"].mean(),
    "bin_processor_success_rate": bin_proc_df["bin_processor_success_rate"].mean()
}
# -------------------------
# 🔍 Prediction Logic
# -------------------------
def predict_top_processors(bin_number, is_3d_encoded, top_n=5, threshold=0.80):
    bin_prefix = bin_number // 1000
    bin_suffix = bin_number % 1000
    bin_known = bin_number in stats["bin_tx"]

    rows = []
    for processor_id in [pid for pid in stats["all_processors"] if reverse_processor_map.get(pid) in last_month_processor_list]:
    # for processor_id in stats["all_processors"]:
        print(reverse_processor_map.get(processor_id), processor_id)
        if bin_known:
            bin_stats = stats["bin_tx"].get(bin_number, {})
            bin_success = stats["bin_success"].get(bin_number, {}).get("bin_success_rate", 0.0)
            proc_success = stats["proc_success"].get(processor_id, {}).get("processor_success_rate", 0.0)
            bin_proc_stats = stats["bin_proc_stats"].get((bin_number, processor_id), {})
        else:
            bin_stats = {"bin_tx_count": global_fallback["bin_tx_count"]}
            bin_success = global_fallback["bin_success_rate"]
            bin_proc_stats = {
                "bin_processor_tx_count": global_fallback["bin_processor_tx_count"],
                "bin_processor_success_count": global_fallback["bin_processor_success_count"],
                "bin_processor_success_rate": global_fallback["bin_processor_success_rate"]
            }
            proc_success = stats["proc_success"].get(processor_id, {}).get("processor_success_rate", 0.0)
        row = {
            "bin": bin_number,
            "bin_prefix": bin_prefix,
            "bin_suffix": bin_suffix,
            "is_3d_encoded": is_3d_encoded,
            "bin_tx_count": bin_stats.get("bin_tx_count", global_fallback["bin_tx_count"]),
            "bin_success_rate": bin_success,
            "processor_success_rate": proc_success,
            "bin_processor_tx_count": bin_proc_stats.get("bin_processor_tx_count", global_fallback["bin_processor_tx_count"]),
            "bin_processor_success_count": bin_proc_stats.get("bin_processor_success_count", global_fallback["bin_processor_success_count"]),
            "bin_processor_success_rate": bin_proc_stats.get("bin_processor_success_rate", global_fallback["bin_processor_success_rate"])
        }
        rows.append((processor_id, row))

    if not rows:
        return [], False

    df_pred = pd.DataFrame([r[1] for r in rows])
    probs = model.predict_proba(df_pred)[:, 1]

    results = sorted(
        [ {
            "processor": reverse_processor_map.get(r[0], f"Unknown ID {r[0]}"),
            "predicted_success": round(prob * 100, 2)
        } for r, prob in zip(rows, probs)],
        key=lambda x: x["predicted_success"],
        reverse=True
    )

    internal = [r for r in results if r["processor"] not in external_processors and r["predicted_success"] >= threshold * 100]
    external = [r for r in results if r["processor"] in external_processors]
    fallback = False if internal else True
    return (internal if internal else external)[:top_n], fallback

tab1, tab2 = st.tabs(["🔮 Predict Processors", "📉 Poor Processors"])
with tab1:
    # -------------------------
    # 💡 Title Section
    # -------------------------
    st.markdown("<h1 style='margin-top: 20px;'>BIN-based Processor Success Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Use this tool to predict top-performing processors for any BIN using AI-powered success rates.")

    # -------------------------
    # 📝 Input Section
    # -------------------------
    with st.container():
        st.markdown("<div style='background: #f9f9f9; padding: 1.5rem; border-radius: 12px;'>", unsafe_allow_html=True)
        input_col1, input_col2 = st.columns([1, 2])

        with input_col1:
            input_method = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

        bin_set = set()
        invalid_bin_count = 0
        duplicate_bin_count = 0

        with input_col2:
            if input_method == "Manual Entry":
                bin_input = st.text_input("Enter BINs (comma-separated):", "510123, 462263")
                raw_bins = [b.strip() for b in bin_input.split(",")]
                for b in raw_bins:
                    if b.isdigit() and len(b) == 6:
                        b_int = int(b)
                        if b_int in bin_set:
                            duplicate_bin_count += 1
                        else:
                            bin_set.add(b_int)
                    else:
                        invalid_bin_count += 1
            else:
                uploaded_file = st.file_uploader("Upload a CSV with a 'BIN' column up to **10 MB**", type=["csv"])
                if uploaded_file:
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > 10:
                        st.error(f"❌ File too large: {file_size_mb:.2f} MB. Please upload a file smaller than 10 MB.")
                    else:
                        try:
                            df_uploaded = pd.read_csv(uploaded_file)
                            if "BIN" in df_uploaded.columns:
                                for b in df_uploaded["BIN"]:
                                    try:
                                        b_int = int(b)
                                        if len(str(b_int)) == 6:
                                            if b_int in bin_set:
                                                duplicate_bin_count += 1
                                            else:
                                                bin_set.add(b_int)
                                        else:
                                            invalid_bin_count += 1
                                    except:
                                        invalid_bin_count += 1
                            else:
                                st.error("❌ CSV must contain a 'BIN' column.")
                        except Exception as e:
                            st.error(f"❌ Error reading CSV: {str(e)}")

        bin_list = list(bin_set) 
        
        is_3d = st.selectbox("Is 3D Secure Enabled?", options=[0, 1], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

    predict = st.button("Predict Processors")

    # -------------------------
    # 📊 Results Section
    # -------------------------
    if predict:
        all_results = []
        if not bin_list:
            st.error("❌ Please enter at least one valid 6-digit BIN before predicting.")
        else:
            if invalid_bin_count > 0:
                st.warning(f"⚠️ Skipped {invalid_bin_count} invalid BIN(s). Only valid integers with 6 digits were accepted.")
            if duplicate_bin_count > 0:
                st.warning(f"ℹ️ Removed {duplicate_bin_count} duplicate BIN(s).")

            for bin_no in bin_list:
                top_processors, _ = predict_top_processors(bin_no, is_3d)
                for rank, proc in enumerate(top_processors, 1):
                    fallback_used = "Yes" if proc["processor"] in external_processors else "No"
                    all_results.append({
                        "BIN": bin_no,
                        "Processor": proc["processor"],
                        "Predicted Success %": proc["predicted_success"],
                        "Rank": rank,
                        "Fallback External": fallback_used
                    })

        if all_results:
            df_result = pd.DataFrame(all_results)
            st.success("✅ Prediction Complete!")
            # st.dataframe(df_result, use_container_width=True)
            def highlight_fallback(row):
                return ['background-color: #ffe6e6' if row["Fallback External"] == "Yes" else '' for _ in row]

            st.dataframe(df_result.style.apply(highlight_fallback, axis=1), use_container_width=True)

            # ✅ FIX: Group predictions by Processor and average success %
            df_plot = df_result.groupby(["Processor", "Fallback External"], as_index=False)["Predicted Success %"].mean()

            # 📊 Processor Success Distribution (Averaged)
            st.markdown("### 📊 Processor Success Distribution")
            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X("Processor:N", sort="-y"),
                y="Predicted Success %:Q",
                color=alt.Color("Fallback External:N", scale=alt.Scale(scheme='blues')),
                tooltip=["Processor", "Predicted Success %", "Fallback External"]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

            # 📊 Fallback Count Chart (optional)
            fallback_df = df_result[df_result["Fallback External"] == "Yes"]
            if not fallback_df.empty:
                fallback_counts = fallback_df["Processor"].value_counts().reset_index()
                fallback_counts.columns = ["Processor", "Count"]
                fallback_chart = alt.Chart(fallback_counts).mark_bar().encode(
                    x="Processor:N", y="Count:Q"
                ).properties(height=300)
                st.altair_chart(fallback_chart, use_container_width=True)

            # 📋 Summary Stats
            avg_internal = df_result[df_result["Fallback External"] == "No"]["Predicted Success %"].mean()
            avg_external = df_result[df_result["Fallback External"] == "Yes"]["Predicted Success %"].mean()

            avg_internal = 0.0 if pd.isna(avg_internal) else avg_internal
            avg_external = 0.0 if pd.isna(avg_external) else avg_external

            st.markdown("### 🔍 Summary Statistics")
            col1, col2 = st.columns(2)
            col1.metric("Average Internal Processor Success", f"{avg_internal:.2f}%")
            col2.metric("Average External Processor Success", f"{avg_external:.2f}%")

            # 💾 Save and Download
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/prediction_log_{timestamp}.csv"
            df_result.to_csv(log_path, index=False)
            csv_data = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Report as CSV", csv_data, "processor_predictions.csv", "text/csv")
            st.info(f"📝 Log saved as: `{log_path}`")
        else:
            st.error("❌ No predictions could be made.")
    # ----------- Retraining Trigger ------------
    AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZ_CONTAINER = os.getenv("BLOB_CONTAINER_NAME")
    blob_service = BlobServiceClient.from_connection_string(AZ_CONN)
    container_client = blob_service.get_container_client(AZ_CONTAINER)

    LAST_RUN_BLOB = "last_run.txt"
    CRON_HISTORY_BLOB = "cron_history.txt"
    TEN_DAYS = timedelta(days=10)
    REFRESH_INTERVAL_MS = 5 * 60 * 1000
    # Replace file-based helpers with Azure Blob-backed ones
    def get_last_run():
        try:
            blob = container_client.get_blob_client(LAST_RUN_BLOB)
            data = blob.download_blob().readall().decode()
            return datetime.fromisoformat(data)
        except Exception:
            return None

    def update_last_run():
        blob = container_client.get_blob_client(LAST_RUN_BLOB)
        blob.upload_blob(datetime.utcnow().isoformat(), overwrite=True)

    def get_refresh_time():
        try:
            blob = container_client.get_blob_client(CRON_HISTORY_BLOB)
            data = blob.download_blob().readall().decode()
            return datetime.fromisoformat(data)
        except Exception:
            return None

    def update_refresh_time():
        blob = container_client.get_blob_client(CRON_HISTORY_BLOB)
        blob.upload_blob(datetime.utcnow().isoformat(), overwrite=True)

    def download_blob_to_file(blob_name, download_path):
        AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)

        with open(download_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        st.success(f"✅ Downloaded blob: {blob_name} to {download_path}")

    # Client-side auto-refresh and retraining logic remains the same
    count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="cron_refresh")
    if count > 0:
        last_run = get_last_run()
        now = datetime.utcnow()
        update_refresh_time()
        st.write("🔁 Auto-refresh detected, checking for job trigger...")
        if last_run is None or (now - last_run) >= TEN_DAYS:
            download_blob_to_file("transaction.csv", "./transaction.csv")
            with st.spinner("Running retraining pipeline..."):
                msg, old_acc, new_acc = run_retraining_pipeline()
            if new_acc is not None:
                update_last_run()
                st.success("✅ Retraining successful!")
                st.text(msg)
            else:
                st.error("❌ Retraining failed.")
                st.text(msg)
        else:
            days_left = TEN_DAYS.days - (now - last_run).days
            st.info(f"⏳ Next update in {days_left} day(s).")

    # Show status
    last = get_last_run()
    if last:
        st.info(f"🕒 Last retraining: {last.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        st.warning("⚠️ No retraining has been done yet.")

    ref = get_refresh_time()
    if ref:
        st.info(f"🕒 Last refresh: {ref.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        st.warning("⚠️ No refresh time found yet.")

with tab2:
    st.markdown("### 🔻 Poor Performing Processors")

    with st.expander("🔧 Adjust Detection Criteria"):
        min_txns = st.slider("Minimum Transactions (per BIN-Processor)", min_value=100, max_value=2000, value=500, step=100)
        approval_threshold = st.slider("Approval Rate Threshold (%)", min_value=30, max_value=90, value=60, step=5)

    def get_poor_processors_df(min_txns=500, approval_threshold=60.0):
        bin_proc_df = pd.DataFrame.from_dict(stats["bin_proc_stats"], orient="index").copy()

        # Extract BIN and Processor ID
        bin_proc_df["bin"] = bin_proc_df.index.map(lambda x: x[0])
        bin_proc_df["processor_id"] = bin_proc_df.index.map(lambda x: x[1])

        # Map processor ID ➝ Name using reversed JSON
        bin_proc_df["processor_name"] = bin_proc_df["processor_id"].map(reverse_processor_map).fillna(bin_proc_df["processor_id"])

        # Approval %
        bin_proc_df["approval_rate_percent"] = bin_proc_df["bin_processor_success_rate"] * 100

        # Flag poor performers
        bin_proc_df["is_poor"] = (
            (bin_proc_df["approval_rate_percent"] < approval_threshold) &
            (bin_proc_df["bin_processor_tx_count"] >= min_txns)
        )

        # Final filtered DataFrame
        poor_df = bin_proc_df[bin_proc_df["is_poor"]].copy()
        poor_df = poor_df[["bin", "processor_id", "processor_name", "bin_processor_tx_count", "bin_processor_success_count", "approval_rate_percent"]]
        return poor_df.sort_values(by="approval_rate_percent")

    poor_processor_df = get_poor_processors_df(min_txns, approval_threshold)
    poor_processor_df = poor_processor_df[poor_processor_df["bin"].isin(last_month_bin_list)]
    if not poor_processor_df.empty:
        st.success(f"🚩 Found {len(poor_processor_df)} poor-performing BIN-Processor combinations.")
        st.dataframe(poor_processor_df, use_container_width=True)

        st.markdown("#### 📉 Poor Processor Approval Rates")
        poor_chart = alt.Chart(poor_processor_df).mark_bar().encode(
            x=alt.X("processor_name:N", sort="-y", title="Processor"),
            y=alt.Y("approval_rate_percent:Q", title="Approval Rate (%)"),
            tooltip=[
                alt.Tooltip("bin", title="BIN"),
                # alt.Tooltip("processor_id", title="Processor ID"),
                alt.Tooltip("processor_name", title="Processor Name"),
                alt.Tooltip("bin_processor_tx_count", title="Tx Count"),
                alt.Tooltip("approval_rate_percent", title="Approval Rate (%)", format=".2f")
            ]
        ).properties(height=350)
        st.altair_chart(poor_chart, use_container_width=True)

        csv_poor = poor_processor_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Poor Processors CSV", csv_poor, "poor_processors.csv", "text/csv")
    else:
        st.success("🎉 No poor-performing processors found based on current threshold.")

# FastAPI endpoint for external scheduler
from fastapi import FastAPI
import threading
import uvicorn

api = FastAPI()

@api.post("/_cron_retrain")
def cron_retrain():
    last_run = get_last_run()
    now = datetime.utcnow()
    update_refresh_time()
    if last_run is None or (now - last_run) >= TEN_DAYS:
        download_blob_to_file("transaction.csv", "./transaction.csv")
        msg, _, new_acc = run_retraining_pipeline()
        if new_acc is not None:
            update_last_run()
            return {"status": "success", "message": msg}
    return {"status": "skipped", "message": "Not due yet."}


def _serve_api():
    uvicorn.run(api, host="0.0.0.0", port=8001)

threading.Thread(target=_serve_api, daemon=True).start()