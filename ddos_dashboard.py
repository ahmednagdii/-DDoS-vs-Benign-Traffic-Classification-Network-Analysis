
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Set page config
st.set_page_config(page_title="DDoS Detection & Analysis", layout="wide")

# Load models and data
@st.cache_resource
def load_model():
    model = joblib.load("GradientBoosting.pkl")
    all_columns = joblib.load("columns.pkl")
    return model, all_columns

@st.cache_data
def load_data():
    return pd.read_csv('SampleToDashboard.csv')

model, all_columns = load_model()


# Define pages
PAGES = {
    "🔍 DDoS Detection": "detection",
    "📊 DDoS Analysis": "analysis"
}

# ----------------------------
# 🔍 DDoS Detection Page
# ----------------------------
def detection_page():
    st.title("🛡️ DDoS Attack Detection Dashboard")
    
    with st.expander("ℹ️ About this tool", expanded=False):
        st.write("""
        This tool helps detect potential DDoS attacks by analyzing network traffic features. 
        Adjust the sliders to match your network traffic characteristics and click 'Predict Now' 
        to get a real-time assessment.
        """)
    
    # Show feature importance explanation BEFORE the input form
    with st.expander("📊 Understanding the Features", expanded=True):
        st.markdown("""
        **Key Features and Their Significance :**
        
           **Fwd Packet Length Mean:** Average size of packets in forward direction 
           **Bwd Packet Length Mean:** Average size of packets in backward direction 
           **Flow IAT Mean:** Average time between packets in the flow 
           **Subflow Fwd Bytes:** Number of bytes transferred in subflow forward direction 
           **Init Win Bytes Forward:** Initial window size in bytes for forward direction 
           **Init Win Bytes Backward:** Initial window size in bytes for backward direction 
           **Total Packets:** Total packets in the flow 
           **Byte Rate Ratio:** Ratio of byte rates in different directions 
        
        
        """)
    with st.form("input_form"):
        st.subheader("📊 Input Network Traffic Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            fwd_packet_length_mean = st.slider("Fwd Packet Length Mean", 0.0, 4000.0, 10.0)
            bwd_packet_length_mean = st.slider("Bwd Packet Length Mean", 0.0, 6000.0, 10.0)
            flow_iat_mean = st.slider("Flow IAT Mean", 0.0, 1e8, 1000.0, step=10000.0)

        with col2:
            subflow_fwd_bytes = st.slider("Subflow Fwd Bytes", 0.0, 200000.0, 1000.0)
            init_win_bytes_forward = st.slider("Init Win Bytes Forward", -1.0, 65535.0, 256.0)
            init_win_bytes_backward = st.slider("Init Win Bytes Backward", -1.0, 65535.0, 256.0)

        with col3:
            total_packets = st.slider("Total Packets", 2, 5000, 10)
            byte_rate_ratio = st.slider("Byte Rate Ratio", 0.0, 100.0, 1.0)

        submitted = st.form_submit_button("🔍 Predict Now")

    if submitted:
        user_input = {
            'Fwd_Packet_Length_Mean': fwd_packet_length_mean,
            'Bwd_Packet_Length_Mean': bwd_packet_length_mean,
            'Flow_IAT_Mean': flow_iat_mean,
            'Subflow_Fwd_Bytes': subflow_fwd_bytes,
            'Init_Win_bytes_forward': init_win_bytes_forward,
            'Init_Win_bytes_backward': init_win_bytes_backward,
            'Total_Packets': total_packets,
            'Byte_Rate_Ratio': byte_rate_ratio
        }

        input_filled = {col: user_input.get(col, 0.0) for col in all_columns}
        input_df = pd.DataFrame([input_filled])

        prediction = model.predict(input_df)[0]

        st.subheader("🎯 Prediction Result")
        if prediction == "DDoS":
            st.error("⚠️ Warning: DDoS Attack Detected!")
            st.markdown("""
            **Recommended Actions:**
            - Investigate the source of traffic
            - Check for unusual patterns in network logs
            - Consider implementing rate limiting
            - Contact your security team
            """)
        else:
            st.success("✅ Normal Traffic (Benign)")

# ----------------------------
# 📊 DDoS Analysis Pages
# ----------------------------
def analysis_page():
    st.sidebar.title("Analysis Sections")
    
    analysis_pages = [
        "🧭 Traffic Overview",
        "📦 Packet Statistics",
        "📊 Flow Metrics",
        "🧠 Time-Based IAT",
        "🚩 Flag Analysis",
        "🔁 Flow Structure",
        "📶 Bulk Data Transfer",
        "🛡️ Window Behavior"
    ]
    
    page = st.sidebar.selectbox("Select Analysis Section", analysis_pages)
    
    st.title(f"{page} Analysis")
    
    if page == "🧭 Traffic Overview":
        traffic_overview()
    elif page == "📦 Packet Statistics":
        packet_statistics()
    elif page == "📊 Flow Metrics":
        flow_metrics()
    elif page == "🧠 Time-Based IAT":
        time_based_iat()
    elif page == "🚩 Flag Analysis":
        flag_analysis()
    elif page == "🔁 Flow Structure":
        flow_structure()
    elif page == "📶 Bulk Data Transfer":
        bulk_data_transfer()
    elif page == "🛡️ Window Behavior":
        window_behavior()

# Analysis page functions
def traffic_overview():
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="Protocol", color="Label", barmode="group",
                            title="Distribution of Protocols by Label")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Hour", color="Label", barmode="overlay",
                            title="Traffic Distribution by Hour")
        fig2.update_layout(bargap=0.1)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x="Flow Duration", color="Label",
                            title="Flow Duration by Label")
        st.plotly_chart(fig3, use_container_width=True)

        label_counts = df["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        fig4 = px.pie(label_counts, names="Label", values="Count",
                      title="Overall Traffic Distribution")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Traffic Trend by Hour")
    hourly = df.groupby(["Hour", "Label"]).size().reset_index(name="Count")
    fig5 = px.line(hourly, x="Hour", y="Count", color="Label", markers=True,
                   title="Traffic Trend by Hour")
    st.plotly_chart(fig5, use_container_width=True)

def packet_statistics():
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="Label", y="Total Packets",
                            title="Total Packets per Flow by Label",
                            log_y=True)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Label", y="Pkt Size Ratio",
                            title="Packet Size Ratio by Label",
                            log_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x="Average Packet Size", color="Label",
                            nbins=100,
                            title="Histogram of Average Packet Size")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.histogram(df, x="Total Packets", color="Label",
                            nbins=60,
                            title="Distribution of Total Packets per Flow")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Average Packet Size by Hour")
    hourly_stats = df.groupby("Hour")["Average Packet Size"].mean().reset_index()
    fig5 = px.line(hourly_stats, x="Hour", y="Average Packet Size",
                   title="Average Packet Size per Hour")
    st.plotly_chart(fig5, use_container_width=True)

def flow_metrics():
    fig1 = px.histogram(df, x="Byte Rate Ratio", color="Label",
                        nbins=10,
                        title="Byte Rate Ratio Histogram (DDoS vs BENIGN)")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 📋 Average Metrics by Label")
    metrics = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Byte Rate Ratio", "Flow Efficiency"]
    grouped = df.groupby("Label")[metrics].mean().round(2)
    st.dataframe(grouped)

    fig3 = px.histogram(df, x="Flow Bytes/s", color="Label",
                        nbins=100, log_y=True,
                        title="Histogram of Flow Bytes/s by Label")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x="Flow Packets/s", color="Label",
                        nbins=100, log_y=True,
                        title="Histogram of Flow Packets/s by Label")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📊 Byte Rate Ratio Quantile-Based Distribution")
    df["Byte Ratio Category"] = pd.qcut(df["Byte Rate Ratio"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    pie_data = df["Byte Ratio Category"].value_counts().reset_index()
    pie_data.columns = ["Category", "Count"]
    fig5 = px.pie(pie_data, names="Category", values="Count",
                  title="Byte Rate Ratio Distribution (Quantile-Based)")
    st.plotly_chart(fig5, use_container_width=True)

def time_based_iat():
    st.markdown("### 📋 IAT Averages by Label")
    columns = ["Flow IAT Mean", "Fwd IAT Total", "Fwd IAT Mean", 
               "Bwd IAT Total", "Bwd IAT Mean", "Active Mean", "Idle Mean"]
    iat_grouped = df.groupby("Label")[columns].mean().round(2)
    st.dataframe(iat_grouped)

    st.markdown("### 📊 IAT Distributions by Label")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="Flow IAT Mean", color="Label", 
                            nbins=100, log_y=True,
                            title="Histogram of Flow IAT Mean by Label")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Idle Mean", color="Label", 
                            nbins=100, log_y=True,
                            title="Histogram of Idle Mean by Label")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x="Active Mean", color="Label", 
                            nbins=100, log_y=True,
                            title="Histogram of Active Mean by Label")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### ⏱️ Hourly IAT Trends (Forward & Backward)")
    iat_hour = df.groupby(["Hour", "Label"])[["Fwd IAT Mean", "Bwd IAT Mean"]].mean().reset_index()
    st.dataframe(iat_hour)

def flag_analysis():
    flag_cols = [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", 
        "ACK Flag Count", "URG Flag Count", "ECE Flag Count"
    ]

    fig1 = px.histogram(df, x="Flag_Sum", color="Label", 
                        nbins=6, barmode="overlay",
                        title="Flag Sum Distribution by Label")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🛑 Most Frequent Flags in DDoS Traffic")
    ddos_flags = df[df["Label"] == "DDoS"][flag_cols].sum().sort_values(ascending=False).reset_index()
    ddos_flags.columns = ["Flag Type", "Total Count"]

    fig2 = px.pie(ddos_flags.head(7), names="Flag Type", values="Total Count",
                  title="Top Used Flags in DDoS Traffic")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="RST Flag Count", color="Label", 
                        nbins=4, barmode="overlay", log_y=True,
                        title="Distribution of RST Flag Count by Label")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x="SYN Flag Count", color="Label", 
                        nbins=4, log_y=True,
                        title="SYN Flag Count Distribution by Label")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### ✅ Flag Usage in BENIGN Traffic")
    benign_flags = df[df["Label"] == "BENIGN"][flag_cols].sum().sort_values(ascending=False).reset_index()
    benign_flags.columns = ["Flag Type", "Total Count"]

    fig5 = px.pie(benign_flags, names="Flag Type", values="Total Count",
                  title="Flag Usage in BENIGN Traffic")
    st.plotly_chart(fig5, use_container_width=True)

def flow_structure():
    cols = [
        "Subflow Fwd Packets", "Subflow Fwd Bytes", 
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Fwd Header Length", "Bwd Header Length", "Header Length Diff",
        "min_seg_size_forward"
    ]

    st.markdown("### 📋 Average Subflow & Header Metrics by Label")
    subflow_means = df.groupby("Label")[cols].mean().round(2)
    st.dataframe(subflow_means)

    fig1 = px.histogram(df, x="Header Length Diff", color="Label",
                        nbins=50, barmode="overlay", log_y=True,
                        title="Header Length Difference Distribution by Label")
    st.plotly_chart(fig1, use_container_width=True)


def bulk_data_transfer():
    bulk_cols = [
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
    ]

    st.markdown("### 🧮 Zero Values in Bulk Transfer Features")
    bulk_zero = (df[bulk_cols] == 0).sum().reset_index()
    bulk_zero.columns = ["Feature", "Zero Count"]
    st.dataframe(bulk_zero.sort_values("Zero Count", ascending=False))

    df["Zero Bulk"] = ((df["Fwd Avg Bulk Rate"] == 0) & (df["Bwd Avg Bulk Rate"] == 0)).astype(int)

    st.markdown("### 📊 Percentage of Sessions with Zero Bulk Transfer")
    zero_bulk_ratio = df.groupby("Label")["Zero Bulk"].mean().reset_index()
    zero_bulk_ratio["Zero Bulk (%)"] = (zero_bulk_ratio["Zero Bulk"] * 100).round(2)

    fig = px.pie(zero_bulk_ratio, names="Label", values="Zero Bulk",
                 title="Percentage of Sessions with Zero Bulk Transfer")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Average Bulk Metrics by Label")
    avg_bulk = df.groupby("Label")[bulk_cols].mean().round(2)
    st.dataframe(avg_bulk)

    st.markdown("### 📈 Bulk Rate Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="Fwd Avg Bulk Rate", color="Label",
                            nbins=60, log_y=True,
                            title="Fwd Avg Bulk Rate by Label")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="Bwd Avg Bulk Rate", color="Label",
                            nbins=60, log_y=True,
                            title="Bwd Avg Bulk Rate by Label")
        st.plotly_chart(fig2, use_container_width=True)

def window_behavior():
    fig1 = px.histogram(df, x="Init_Win_bytes_forward", color="Label", 
                        nbins=60, barmode="overlay", log_y=True,
                        title="Initial Window Bytes (Forward) by Label")
    st.plotly_chart(fig1, use_container_width=True)

    df["No_Active_Data_Pkt"] = (df["act_data_pkt_fwd"] == 0).astype(int)
    no_data_pie = df.groupby("Label")["No_Active_Data_Pkt"].mean().reset_index()
    no_data_pie["No_Active_Data_Pkt (%)"] = (no_data_pie["No_Active_Data_Pkt"] * 100).round(2)
    fig2 = px.pie(no_data_pie, names="Label", values="No_Active_Data_Pkt (%)",
                  title="Sessions with 0 Active Data Packets Forward (%)")
    st.plotly_chart(fig2, use_container_width=True)

    df["Zero_Init_Forward"] = (df["Init_Win_bytes_forward"] == 0).astype(int)
    zero_win_pie = df.groupby("Label")["Zero_Init_Forward"].mean().reset_index()
    zero_win_pie["Zero_Init_Forward (%)"] = (zero_win_pie["Zero_Init_Forward"] * 100).round(2)
    fig3 = px.pie(zero_win_pie, names="Label", values="Zero_Init_Forward (%)",
                  title="Sessions with Init_Win_bytes_forward = 0 (%)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📋 Down/Up Ratio Mean by Label")
    du_stats = df.groupby("Label")["Down/Up Ratio"].mean().round(2).reset_index()
    st.dataframe(du_stats)

    df_grouped = df.groupby(["Hour", "Label"])["Init_Win_bytes_forward"].mean().reset_index()
    fig4 = px.line(df_grouped, x="Hour", y="Init_Win_bytes_forward", color="Label",
                   title="Avg Init_Win_bytes_forward by Hour & Label")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(df[df["Label"] == "DDoS"], x="Down/Up Ratio",
                        nbins=25,
                        title="Down/Up Ratio Distribution in DDoS Sessions")
    st.plotly_chart(fig5, use_container_width=True)

# ----------------------------
# Main App
# ----------------------------
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    if PAGES[selection] == "detection":
        detection_page()
    elif PAGES[selection] == "analysis":
        analysis_page()

if __name__ == "__main__":
    main()
