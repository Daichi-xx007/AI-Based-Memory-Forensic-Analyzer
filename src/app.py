import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from data_preprocessing import DataPreprocessor
from search_algo import FeatureSelector, DataSearcher
from base_models import BaseModelTrainer
from advanced_models import AdvancedModelTrainer
import shap
import streamlit.components.v1 as components
import base64
import json
from datetime import datetime

# --- Custom CSS for Premium "Hacker" Look (UI 2.0) ---
st.set_page_config(page_title="AI Memory Forensics Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* Dark Theme & Glassmorphism */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #FAFAFA;
    }
    
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.5);
    }

    h1 {
        color: #00f260;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 0 0 10px rgba(0, 242, 96, 0.5);
    }
    
    h2, h3 {
        color: #0575E6;
         font-family: 'Segoe UI', sans-serif;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00f260 0%, #0575E6 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 242, 96, 0.5);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #00f260;
    }
</style>
""", unsafe_allow_html=True)

# --- Hepler Functions ---
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        dp = DataPreprocessor(path)
        df = dp.load_data()
        df = dp.clean_and_encode()
        train_X, test_X, train_y, test_y, train_mal_y, test_mal_y = dp.split_data()
        return dp, df, train_X, test_X, train_y, test_y, train_mal_y, test_mal_y
    return None, None, None, None, None, None, None, None

def save_scan_history(record):
    history_file = 'scan_history.json'
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try: history = json.load(f)
            except: pass
    history.append(record)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

def load_scan_history():
    history_file = 'scan_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try: return json.load(f)
            except: return []
    return []

# --- Main App ---

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=120)
st.sidebar.markdown("## 🛡️ CyberSentinel AI")
data_path = st.sidebar.text_input("Dataset Path", "malmem.csv")

dp, df, X_train, X_test, y_train, y_test, y_mal_train, y_mal_test = load_data(data_path)

if df is None:
    st.error(f"File not found: {data_path}")
    st.stop()

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Dashboard & Viz", "Scan Dump", "Advanced Analysis", "History"])

if page == "Dashboard & Viz":
    st.title("🛡️ Threat Intelligence Dashboard")
    
    # 1. Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Samples", df.shape[0])
    with col2: st.metric("Malware Ratio", f"{df['Class'].mean()*100:.1f}%")
    with col3: st.metric("Features", df.shape[1])
    with col4: st.metric("System Status", "SECURE", delta="Active")
    
    st.markdown("---")
    
    # 2. 3D Cluster Visualization
    st.subheader("🧬 3D Memory Space Visualization (PCA)")
    with st.spinner("Generating 3D Plot..."):
        # Reduce to 3 components for 3D plot
        pca = PCA(n_components=3)
        # Take a subset for speed
        subset_size = 2000
        X_subset = X_train.iloc[:subset_size]
        y_subset = y_train.iloc[:subset_size]
        X_pca = pca.fit_transform(X_subset)
        
        viz_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        viz_df['Class'] = y_subset.map({0: 'Benign', 1: 'Malware'}).values
        
        fig = px.scatter_3d(viz_df, x='PC1', y='PC2', z='PC3', color='Class', 
                            color_discrete_map={'Benign': '#00f260', 'Malware': '#ff4b1f'},
                            opacity=0.7, title="Memory Dumplings Clustering")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Scan Dump":
    st.title("🔍 Advanced Threat Scanner")
    uploaded_file = st.file_uploader("Upload Memory Dump (CSV)", type="csv")
    
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        
        if st.button("Run Deep Scan"):
            with st.spinner("Initializing Deep Scan Protocol..."):
                try:
                    # Preprocess
                    scan_df = input_df.copy()
                    if 'Class' in scan_df.columns: scan_df = scan_df.drop(columns=['Class'])
                    if 'Category' in scan_df.columns: scan_df = scan_df.drop(columns=['Category'])
                    scan_df = scan_df[X_train.columns] # alignment
                    scan_scaled = dp.scaler.transform(scan_df)
                    
                    # Load Models
                    ensemble = joblib.load('models/ensemble.pkl')
                    anomaly_detector = joblib.load('models/anomaly_detector.pkl')
                    multiclass = joblib.load('models/mlp_multiclass.pkl')
                    
                    # Predictions
                    preds = ensemble.predict(scan_scaled)
                    probs = ensemble.predict_proba(scan_scaled)
                    anomaly_scores = anomaly_detector.decision_function(scan_scaled) # neg score = anomaly
                    
                    # Display Results
                    st.success("Scan Completed.")
                    
                    for i, pred in enumerate(preds):
                        is_malware = pred == 1
                        confidence = max(probs[i]) * 100
                        is_anomaly = anomaly_scores[i] < 0 
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if is_malware:
                                st.error(f"Sample {i}: MALWARE DETECTED")
                            else:
                                st.success(f"Sample {i}: BENIGN")
                        
                        with col2:
                            # Advanced Details
                            mal_type = "N/A"
                            if is_malware:
                                mal_type_idx = multiclass.predict([scan_scaled[i]])[0]
                                mal_type = dp.malware_encoder.inverse_transform([mal_type_idx])[0]
                            
                            st.write(f"**Type:** {mal_type} | **Confidence:** {confidence:.2f}% | **Anomaly Score:** {anomaly_scores[i]:.4f}")
                            if is_anomaly:
                                st.warning("⚠️ Abnormal pattern detected (possible Zero-Day)")
                                
                            # Save to History
                            record = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sample_id": i,
                                "status": "Malware" if is_malware else "Benign",
                                "type": mal_type,
                                "confidence": confidence,
                                "anomaly_score": anomaly_scores[i]
                            }
                            save_scan_history(record)
                            
                except Exception as e:
                    st.error(f"Scan Error: {e}")

elif page == "Advanced Analysis":
    st.title("🧠 Neural Core Training")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Train the Ensemble (Super Learner)")
        if st.button("Train Ensemble"):
            with st.spinner("Training..."):
                adv = AdvancedModelTrainer(X_train, y_train, X_test, y_test, y_mal_train, y_mal_test)
                adv.train_ensemble_model()
                adv.train_anomaly_detector()
                adv.save_models()
                st.success("Ensemble & Anomaly Models Trained!")
                
    with col2:
        st.info("Train Deep Learning Network")
        if st.button("Train Deep Network"):
            # Call existing logic if needed
            st.warning("Use the other tab for standard MLP training or re-run pipeline.")
            
elif page == "History":
    st.title("📜 Scan Logs")
    history = load_scan_history()
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df.style.applymap(lambda v: 'color: red' if v == 'Malware' else 'color: green', subset=['status']))
        
        if st.button("Clear History"):
            if os.path.exists('scan_history.json'):
                os.remove('scan_history.json')
                st.experimental_rerun()
    else:
        st.info("No scan history found.")
