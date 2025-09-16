import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import json

# Configure page
st.set_page_config(
    page_title="Smart Grid Predictive Maintenance | AKS",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
if 'ai_assistant_visible' not in st.session_state:
    st.session_state.ai_assistant_visible = True
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = []
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None

# Dark mode CSS with modern design
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    .ai-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(14, 17, 23, 0.95);
        backdrop-filter: blur(12px);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .ai-assistant {
        background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
        border: 2px solid #ff6b35;
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 550px;
        text-align: center;
        box-shadow: 0 25px 50px rgba(255, 107, 53, 0.3);
        animation: aiFloat 4s ease-in-out infinite;
    }
    
    @keyframes aiFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(1deg); }
    }
    
    .ai-avatar {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        margin: 0 auto 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        animation: pulse 2.5s infinite;
        box-shadow: 0 0 30px rgba(255, 107, 53, 0.5);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card:hover {
        border-color: #ff6b35;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.2);
        transform: translateY(-5px);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
    }
    
    .risk-critical {
        border-left: 5px solid #ff4757;
        background: linear-gradient(135deg, #2d1b1b 0%, #1a1a2e 100%);
    }
    
    .risk-high {
        border-left: 5px solid #ffa502;
        background: linear-gradient(135deg, #2d2b1b 0%, #1a1a2e 100%);
    }
    
    .risk-medium {
        border-left: 5px solid #2ed573;
        background: linear-gradient(135deg, #1b2d1b 0%, #1a1a2e 100%);
    }
    
    .ml-metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .ml-metric-card:hover {
        border-color: #ff6b35;
        transform: scale(1.05);
    }
    
    .model-performance {
        background: #2d1b69;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #11998e;
    }
    
    .crew-optimization {
        background: #1b2d2d;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
    }
    
    .cost-savings {
        background: #2d2d1b;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f7931e;
    }
    
    .architecture-node {
        background: #1a1a2e;
        border: 2px solid #ff6b35;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .architecture-node:hover {
        background: #2d1b69;
        transform: scale(1.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
    }
    
    .feature-importance {
        background: #16213e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff6b35;
    }
    
    .footer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-top: 2px solid #ff6b35;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Mock data for 9,000+ monitored assets
def generate_asset_data():
    asset_types = ["Transformer", "Circuit Breaker", "Relay", "Capacitor Bank", "Switch", "Cable", "Bus"]
    voltage_levels = ["4.16kV", "12.47kV", "25kV", "69kV", "138kV", "230kV", "500kV"]
    locations = ["Substation Alpha", "Substation Beta", "Substation Gamma", "Substation Delta", 
                "Distribution Hub North", "Distribution Hub South", "Transmission Yard West"]
    
    assets = []
    for i in range(150):  # Show subset of 9,000+
        failure_prob = np.random.beta(2, 10)  # Most assets low risk, few high risk
        risk_level = "CRITICAL" if failure_prob > 0.15 else "HIGH" if failure_prob > 0.08 else "MEDIUM"
        
        asset = {
            "asset_id": f"AST-{i+1:04d}",
            "asset_type": random.choice(asset_types),
            "voltage_level": random.choice(voltage_levels),
            "location": random.choice(locations),
            "failure_probability": failure_prob,
            "risk_level": risk_level,
            "days_to_failure": max(1, int(np.random.exponential(45) * (1 - failure_prob))),
            "maintenance_cost": random.randint(5000, 85000),
            "replacement_cost": random.randint(50000, 1200000),
            "criticality_score": random.uniform(0.1, 1.0),
            "last_maintenance": datetime.now() - timedelta(days=random.randint(30, 800))
        }
        assets.append(asset)
    
    return sorted(assets, key=lambda x: x["failure_probability"], reverse=True)

# ML Model Features and Performance Data
ML_FEATURES = {
    "Electrical Parameters": [
        "Voltage THD (%)", "Current Unbalance (%)", "Power Factor", "Load Factor (%)",
        "Harmonics 3rd/5th/7th", "Neutral Current (A)", "Insulation Resistance (MΩ)"
    ],
    "Thermal Characteristics": [
        "Operating Temperature (°C)", "Temperature Rise Rate (°C/hr)", "Hot Spot Temperature",
        "Ambient Temperature Delta", "Cooling System Efficiency (%)", "Oil Temperature (transformers)"
    ],
    "Mechanical Indicators": [
        "Vibration Level (mm/s)", "Contact Resistance (μΩ)", "Operating Time (ms)",
        "SF6 Gas Pressure (bar)", "Mechanism Travel Time", "Contact Wear Pattern"
    ],
    "Environmental Factors": [
        "Humidity Level (%)", "Contamination Index", "UV Exposure Hours",
        "Salt Deposit Density", "Wind Loading Factor", "Seismic Activity Level"
    ],
    "Operational History": [
        "Fault Count (last 12 months)", "Operating Cycles", "Maintenance Intervals",
        "Load History Variance", "Emergency Operations", "Manufacturer Age (years)"
    ]
}

MODEL_PERFORMANCE = {
    "XGBoost Ensemble": {"precision": 0.94, "recall": 1.00, "f1_score": 0.97, "accuracy": 0.96},
    "TensorFlow Neural Net": {"precision": 0.91, "recall": 0.98, "f1_score": 0.94, "accuracy": 0.93},
    "scikit-learn Random Forest": {"precision": 0.89, "recall": 0.96, "f1_score": 0.92, "accuracy": 0.91}
}

BUSINESS_IMPACT = {
    "total_assets_monitored": 9247,
    "high_risk_identified": 146,
    "percentage_high_risk": 1.6,
    "potential_savings": 2300000,
    "prevented_outages": 23,
    "crew_efficiency_gain": 34,
    "alert_fatigue_reduction": 78
}

# Simple AI Assistant without complex overlay - just show message and buttons
if st.session_state.ai_assistant_visible:
    # Create a centered welcome screen
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; 
                min-height: 80vh; background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);">
        <div style="background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
                    border: 3px solid #ff6b35; border-radius: 25px; padding: 3rem;
                    max-width: 600px; text-align: center; box-shadow: 0 25px 50px rgba(255, 107, 53, 0.4);">
            
            <div style="width: 100px; height: 100px; border-radius: 50%;
                        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
                        margin: 0 auto 2rem; display: flex; align-items: center; justify-content: center;
                        font-size: 3rem; animation: pulse 2s infinite;">
                🤖
            </div>
            
            <h1 style="color: #ff6b35; margin-bottom: 1.5rem; font-size: 2.5rem;">
                ML Engineering Assistant
            </h1>
            
            <p style="font-size: 1.3em; line-height: 1.6; margin-bottom: 1.5rem; color: white;">
                Hey there! I'm your AI-powered predictive maintenance assistant. I've been trained on 
                9,000+ grid assets using XGBoost, TensorFlow, and scikit-learn ensemble models. 
                Ready to prevent failures and save millions? Let's dive into the data! 📊
            </p>
            
            <p style="font-size: 1.1em; color: #ccc; margin-bottom: 2.5rem;">
                <strong>Current Status:</strong> 146 high-risk assets identified • $2.3M potential savings
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Now add the buttons below the welcome message - these will be visible
    st.markdown("---")
    
    # Center the buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 Enter Website", 
                    key="enter_main_button", 
                    help="Click to enter the ML platform",
                    use_container_width=True):
            st.session_state.ai_assistant_visible = False
            st.success("Welcome to the Smart Grid Predictive Maintenance Platform!")
            st.rerun()
    
    with col2:
        if st.button("🔊 Test Voice", 
                    key="test_voice_button",
                    help="Test voice synthesis",
                    use_container_width=True):
            st.markdown("""
            <script>
            if ('speechSynthesis' in window) {
                const msg = new SpeechSynthesisUtterance('Hello! Welcome to our Smart Grid platform. I am your ML assistant ready to help!');
                msg.rate = 0.8;
                speechSynthesis.speak(msg);
            } else {
                alert('Voice not supported in your browser');
            }
            </script>
            """, unsafe_allow_html=True)
            st.info("🔊 Voice test initiated! You should hear a welcome message.")
    
    with col3:
        if st.button("ℹ️ Skip Intro", 
                    key="skip_intro_button",
                    help="Skip directly to the platform",
                    use_container_width=True):
            st.session_state.ai_assistant_visible = False
            st.rerun()
    
    # Instructions
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; 
                background: rgba(255, 107, 53, 0.1); border-radius: 10px;">
    <h4 style="color: #ff6b35;">Quick Start Instructions:</h4>
    <p style="color: #ccc;">
    • Click <strong>"🚀 Enter Website"</strong> to start exploring the ML platform<br>
    • Try <strong>"🔊 Test Voice"</strong> to check if voice synthesis works in your browser<br>
    • Use <strong>"ℹ️ Skip Intro"</strong> to go directly to the dashboard
    </p>
    </div>
    """, unsafe_allow_html=True)

# Main content (only show when overlay is dismissed)
if not st.session_state.ai_assistant_visible:
    # Main header
    st.markdown('<div class="main-header">🔧 Smart Grid Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2rem;'>ML-Powered Asset Management on Azure Kubernetes Service</p>", unsafe_allow_html=True)

    # Generate asset data
    if 'assets_data' not in st.session_state:
        st.session_state.assets_data = generate_asset_data()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 ML System Status")
        st.markdown("**Platform:** Azure Kubernetes Service")
        st.markdown("**ML Stack:** XGBoost + TensorFlow + scikit-learn")
        st.markdown("**Orchestration:** Apache Airflow")
        st.markdown("**Monitoring:** Grafana + Prometheus")
        st.markdown("**Model Accuracy:** 96.4%")
        st.markdown("**Recall Rate:** 100%")
        
        st.markdown("### 📊 Business Metrics")
        st.metric("Assets Monitored", "9,247", "+127")
        st.metric("High-Risk Assets", "146", "-12")
        st.metric("Potential Savings", "$2.3M", "+$340K")
        st.metric("Crew Efficiency", "34%", "+8%")

    # Navigation
    page = st.sidebar.radio("Navigate:", [
        "📊 Predictive Dashboard",
        "🤖 ML Model Performance", 
        "👥 Crew Optimization",
        "🏗️ AKS Architecture",
        "💰 Business Impact Analysis"
    ])

    if page == "📊 Predictive Dashboard":
        st.markdown("### 🎯 Real-Time Asset Risk Assessment")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="ml-metric-card">
            <h4 style="color: #ff4757;">Critical Risk</h4>
            <h2>23</h2>
            <small>Immediate action required</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="ml-metric-card">
            <h4 style="color: #ffa502;">High Risk</h4>
            <h2>123</h2>
            <small>Schedule within 30 days</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="ml-metric-card">
            <h4 style="color: #2ed573;">Medium Risk</h4>
            <h2>1,847</h2>
            <small>Routine monitoring</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="ml-metric-card">
            <h4 style="color: #70a1ff;">Low Risk</h4>
            <h2>7,254</h2>
            <small>Normal operation</small>
            </div>
            """, unsafe_allow_html=True)

        # Asset predictions table
        st.markdown("### 🔮 Top Priority Asset Predictions")
        
        # Filter top 20 highest risk assets
        high_risk_assets = [asset for asset in st.session_state.assets_data 
                          if asset["risk_level"] in ["CRITICAL", "HIGH"]][:20]
        
        for asset in high_risk_assets:
            risk_class = f"risk-{asset['risk_level'].lower()}"
            probability_percent = f"{asset['failure_probability']*100:.1f}%"
            
            st.markdown(f"""
            <div class="prediction-card {risk_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{asset['asset_id']} - {asset['asset_type']}</h4>
                    <p><strong>Location:</strong> {asset['location']} | <strong>Voltage:</strong> {asset['voltage_level']}</p>
                    <p><strong>Failure Probability:</strong> {probability_percent} | <strong>Est. Days to Failure:</strong> {asset['days_to_failure']}</p>
                </div>
                <div style="text-align: right;">
                    <h3 style="color: #ff6b35;">{asset['risk_level']}</h3>
                    <p><strong>Maintenance Cost:</strong> ${asset['maintenance_cost']:,}</p>
                    <p><strong>Replacement Cost:</strong> ${asset['replacement_cost']:,}</p>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Interactive prediction demo
        st.markdown("### 🎯 Interactive Asset Analysis")
        
        selected_asset_id = st.selectbox("Select Asset for Detailed Analysis:", 
                                       [asset['asset_id'] for asset in high_risk_assets])
        
        if st.button("Run ML Prediction Analysis"):
            selected_asset = next(asset for asset in high_risk_assets 
                                if asset['asset_id'] == selected_asset_id)
            
            with st.spinner("Running ensemble ML models (XGBoost + TensorFlow + scikit-learn)..."):
                time.sleep(2.5)
            
            # Generate detailed prediction
            st.markdown(f"""
            <div class="model-performance">
            <h4>🧠 ML Model Prediction Results for {selected_asset['asset_id']}</h4>
            <p><strong>Asset Type:</strong> {selected_asset['asset_type']} at {selected_asset['location']}</p>
            <p><strong>XGBoost Prediction:</strong> {selected_asset['failure_probability']*100:.2f}% failure probability</p>
            <p><strong>TensorFlow Neural Net:</strong> {(selected_asset['failure_probability']*0.97)*100:.2f}% failure probability</p>
            <p><strong>Random Forest:</strong> {(selected_asset['failure_probability']*1.03)*100:.2f}% failure probability</p>
            <p><strong>Ensemble Average:</strong> {selected_asset['failure_probability']*100:.1f}% (High Confidence)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance
            st.markdown("### 📈 Top Contributing Risk Factors")
            feature_importance = [
                ("Operating Temperature Anomaly", 0.23),
                ("Vibration Pattern Change", 0.19),
                ("Load Factor Deviation", 0.16),
                ("Maintenance Interval Overdue", 0.14),
                ("Contact Resistance Increase", 0.12),
                ("Environmental Stress Index", 0.09),
                ("Historical Fault Pattern", 0.07)
            ]
            
            for feature, importance in feature_importance:
                st.markdown(f"""
                <div class="feature-importance">
                <strong>{feature}:</strong> {importance:.0%} contribution to risk score
                </div>
                """, unsafe_allow_html=True)

    elif page == "🤖 ML Model Performance":
        st.markdown("### 🧠 Ensemble ML Model Performance")
        
        # Model performance comparison
        st.markdown("#### 📊 Model Accuracy Metrics")
        
        for model_name, metrics in MODEL_PERFORMANCE.items():
            st.markdown(f"""
            <div class="model-performance">
            <h4>{model_name}</h4>
            <div style="display: flex; justify-content: space-between;">
                <div><strong>Precision:</strong> {metrics['precision']:.1%}</div>
                <div><strong>Recall:</strong> {metrics['recall']:.1%}</div>
                <div><strong>F1-Score:</strong> {metrics['f1_score']:.1%}</div>
                <div><strong>Accuracy:</strong> {metrics['accuracy']:.1%}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Feature categories
        st.markdown("#### 🔧 Model Features by Category")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for category in list(ML_FEATURES.keys())[:3]:
                with st.expander(f"📊 {category}"):
                    for feature in ML_FEATURES[category]:
                        st.write(f"• {feature}")
        
        with col2:
            for category in list(ML_FEATURES.keys())[3:]:
                with st.expander(f"📊 {category}"):
                    for feature in ML_FEATURES[category]:
                        st.write(f"• {feature}")

        # Model training pipeline
        st.markdown("#### 🔄 Automated ML Pipeline (Airflow DAGs)")
        
        st.markdown("""
        <div class="model-performance">
        <h4>Daily Retraining Pipeline</h4>
        <p><strong>1. Data Ingestion:</strong> Collect 24-hour SCADA/sensor data from 9,247 assets</p>
        <p><strong>2. Feature Engineering:</strong> Calculate rolling statistics, anomaly scores, trend analysis</p>
        <p><strong>3. Model Training:</strong> Retrain ensemble models with new data + historical context</p>
        <p><strong>4. Validation:</strong> Rolling window validation with time-series cross-validation</p>
        <p><strong>5. Deployment:</strong> A/B test new models, gradual rollout to production</p>
        <p><strong>6. Monitoring:</strong> Track prediction accuracy, drift detection, business impact</p>
        </div>
        """, unsafe_allow_html=True)

        # Real-time model monitoring
        st.markdown("#### 📈 Real-Time Model Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Drift Score", "0.03", "-0.01")
            st.metric("Prediction Latency", "47ms", "-3ms")
        
        with col2:
            st.metric("False Positive Rate", "3.2%", "-0.8%")
            st.metric("Feature Stability", "98.7%", "+0.3%")
        
        with col3:
            st.metric("Data Quality Score", "99.1%", "+0.2%")
            st.metric("Model Confidence", "96.4%", "+1.1%")

    elif page == "👥 Crew Optimization":
        st.markdown("### 👥 Intelligent Crew Scheduling & Capacity Optimization")
        
        # Crew capacity metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Crews", "23", "+2")
        with col2:
            st.metric("Scheduled Work Orders", "67", "+12")
        with col3:
            st.metric("Crew Utilization", "87%", "+15%")
        with col4:
            st.metric("Emergency Availability", "4 crews", "")

        # Optimization algorithm results
        st.markdown("#### 🎯 ML-Optimized Crew Assignments")
        
        crew_assignments = [
            {
                "crew_id": "CREW-A01",
                "specialization": "Transmission Maintenance",
                "assigned_assets": ["AST-0001", "AST-0047", "AST-0089"],
                "total_work_hours": 32,
                "travel_time_optimized": "2.3 hours saved",
                "priority_score": 0.94
            },
            {
                "crew_id": "CREW-B03",
                "specialization": "Distribution Repair",
                "assigned_assets": ["AST-0012", "AST-0034", "AST-0156"],
                "total_work_hours": 28,
                "travel_time_optimized": "1.8 hours saved",
                "priority_score": 0.87
            },
            {
                "crew_id": "CREW-C07",
                "specialization": "Protection Systems",
                "assigned_assets": ["AST-0023", "AST-0078"],
                "total_work_hours": 24,
                "travel_time_optimized": "3.1 hours saved",
                "priority_score": 0.91
            }
        ]
        
        for crew in crew_assignments:
            st.markdown(f"""
            <div class="crew-optimization">
            <h4>{crew['crew_id']} - {crew['specialization']}</h4>
            <p><strong>Assigned Assets:</strong> {', '.join(crew['assigned_assets'])}</p>
            <p><strong>Total Work Hours:</strong> {crew['total_work_hours']} hours | <strong>Travel Optimization:</strong> {crew['travel_time_optimized']}</p>
            <p><strong>Priority Score:</strong> {crew['priority_score']:.0%} (ML-calculated based on asset criticality + location)</p>
            </div>
            """, unsafe_allow_html=True)

        # Resource constraint optimization
        st.markdown("#### ⚙️ Resource Constraint Optimization")
        
        st.markdown("""
        <div class="crew-optimization">
        <h4>Optimization Algorithm Results</h4>
        <p><strong>Objective Function:</strong> Minimize (failure_risk × replacement_cost) + travel_time + crew_overtime</p>
        <p><strong>Constraints:</strong> Crew specialization matching, work hour limits, geographic proximity</p>
        <p><strong>Solution Method:</strong> Integer Linear Programming with genetic algorithm refinement</p>
        <p><strong>Optimization Results:</strong> 34% improvement in crew efficiency, 67% reduction in emergency calls</p>
        </div>
        """, unsafe_allow_html=True)

        # Interactive crew scheduling
        st.markdown("#### 📅 Interactive Crew Scheduling")
        
        if st.button("Generate Optimized Weekly Schedule"):
            with st.spinner("Running crew optimization algorithm..."):
                time.sleep(2)
                
            st.success("✅ Optimized schedule generated! 23 crews assigned to 67 high-priority assets")
            st.markdown("""
            <div class="crew-optimization">
            <h4>Weekly Schedule Optimization Results</h4>
            <p><strong>Total Assets Covered:</strong> 67 high-risk assets</p>
            <p><strong>Estimated Completion:</strong> 5.2 days (vs 7.8 days unoptimized)</p>
            <p><strong>Travel Distance Reduced:</strong> 340 miles saved across all crews</p>
            <p><strong>Emergency Response Capacity:</strong> 4 crews maintained for urgent failures</p>
            <p><strong>Cost Efficiency:</strong> $47,000 saved in overtime and travel expenses</p>
            </div>
            """, unsafe_allow_html=True)

    elif page == "🏗️ AKS Architecture":
        st.markdown("### 🏗️ Azure Kubernetes Service Architecture")
        
        # Architecture diagram
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    border: 2px solid #ff6b35; border-radius: 15px; padding: 2rem; text-align: center; margin: 2rem 0;">
        <h4 style="color: #ff6b35;">ML Pipeline Data Flow on AKS</h4>
        <div style="font-size: 1.1em; line-height: 2; margin-top: 1rem;">
        <strong>SCADA Data</strong> → <strong>Kafka Ingestion</strong> → <strong>Feature Store</strong> → 
        <strong>ML Training Pods</strong> → <strong>Model Registry</strong> → <strong>Inference Service</strong> → 
        <strong>Grafana Dashboard</strong>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Kubernetes components
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 ML Training Infrastructure")
            
            components = [
                {"name": "XGBoost Training Pods", "replicas": "3", "cpu": "4 cores", "memory": "16GB"},
                {"name": "TensorFlow GPU Pods", "replicas": "2", "cpu": "8 cores + GPU", "memory": "32GB"},
                {"name": "Feature Engineering", "replicas": "5", "cpu": "2 cores", "memory": "8GB"},
                {"name": "Data Validation", "replicas": "2", "cpu": "2 cores", "memory": "4GB"}
            ]
            
            for comp in components:
                st.markdown(f"""
                <div class="architecture-node">
                <h5>{comp['name']}</h5>
                <p>Replicas: {comp['replicas']} | CPU: {comp['cpu']} | RAM: {comp['memory']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⚙️ Production Infrastructure")
            
            prod_components = [
                {"name": "Inference API Gateway", "replicas": "4", "cpu": "2 cores", "memory": "4GB"},
                {"name": "Model Serving (MLflow)", "replicas": "3", "cpu": "4 cores", "memory": "8GB"},
                {"name": "Airflow Workers", "replicas": "6", "cpu": "2 cores", "memory": "6GB"},
                {"name": "Grafana + Prometheus", "replicas": "2", "cpu": "2 cores", "memory": "8GB"}
            ]
            
            for comp in prod_components:
                st.markdown(f"""
                <div class="architecture-node">
                <h5>{comp['name']}</h5>
                <p>Replicas: {comp['replicas']} | CPU: {comp['cpu']} | RAM: {comp['memory']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Airflow DAG workflow
        st.markdown("#### 🔄 Apache Airflow DAG Architecture")
        
        st.markdown("""
        <div class="model-performance">
        <h4>Automated ML Pipeline DAGs</h4>
        <p><strong>daily_model_retrain.py:</strong> Scheduled daily at 2 AM, pulls 24h data, retrains ensemble models</p>
        <p><strong>feature_engineering.py:</strong> Runs every 4 hours, calculates rolling statistics and anomaly scores</p>
        <p><strong>model_validation.py:</strong> Weekly validation against hold-out test set, drift detection</p>
        <p><strong>prediction_batch.py:</strong> Hourly batch predictions for all 9,247 assets</p>
        <p><strong>alert_generation.py:</strong> Real-time alert processing when failure probability > threshold</p>
        </div>
        """, unsafe_allow_html=True)

        # System performance metrics
        st.markdown("#### 📊 AKS Cluster Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pod Uptime", "99.7%", "+0.2%")
        with col2:
            st.metric("CPU Utilization", "67%", "+5%")
        with col3:
            st.metric("Memory Usage", "73%", "+3%")
        with col4:
            st.metric("Storage IOPS", "12.5K", "+1.2K")

        # Cost optimization
        st.markdown("#### 💰 AKS Cost Optimization")
        
        st.markdown("""
        <div class="cost-savings">
        <h4>Kubernetes Cost Optimization Results</h4>
        <p><strong>Auto-scaling:</strong> Cluster scales from 12-45 nodes based on ML training demand</p>
        <p><strong>Spot Instances:</strong> 70% of training workload runs on Azure Spot VMs (-80% cost)</p>
        <p><strong>Resource Rightsizing:</strong> ML model optimization reduced memory usage by 40%</p>
        <p><strong>Monthly Infrastructure Cost:</strong> $23,400 (vs $67,000 on-premise equivalent)</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "💰 Business Impact Analysis":
        st.markdown("### 💰 Business Impact & ROI Analysis")
        
        # Key business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="ml-metric-card">
            <h4 style="color: #2ed573;">Potential Savings</h4>
            <h2>${BUSINESS_IMPACT['potential_savings']:,}</h2>
            <small>Annual projected savings</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="ml-metric-card">
            <h4 style="color: #70a1ff;">Prevented Outages</h4>
            <h2>{BUSINESS_IMPACT['prevented_outages']}</h2>
            <small>Major failures avoided</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="ml-metric-card">
            <h4 style="color: #ffa502;">Crew Efficiency</h4>
            <h2>+{BUSINESS_IMPACT['crew_efficiency_gain']}%</h2>
            <small>Productivity improvement</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="ml-metric-card">
            <h4 style="color: #ff6b35;">Alert Reduction</h4>
            <h2>-{BUSINESS_IMPACT['alert_fatigue_reduction']}%</h2>
            <small>False positive elimination</small>
            </div>
            """, unsafe_allow_html=True)

        # Cost breakdown analysis
        st.markdown("#### 📈 Detailed Cost-Benefit Analysis")
        
        cost_categories = [
            {"category": "Emergency Repair Costs Avoided", "amount": 1850000, "description": "Prevented catastrophic failures"},
            {"category": "Reduced Overtime Labor", "amount": 280000, "description": "Optimized crew scheduling"},
            {"category": "Parts Inventory Optimization", "amount": 170000, "description": "Predictive parts ordering"},
            {"category": "Customer Outage Cost Reduction", "amount": 320000, "description": "Improved reliability metrics"},
            {"category": "Regulatory Compliance Savings", "amount": 95000, "description": "Avoided NERC penalties"}
        ]
        
        for cost in cost_categories:
            st.markdown(f"""
            <div class="cost-savings">
            <h4>{cost['category']}: ${cost['amount']:,}</h4>
            <p>{cost['description']}</p>
            </div>
            """, unsafe_allow_html=True)

        # ROI calculation
        st.markdown("#### 💡 Return on Investment Calculation")
        
        implementation_cost = 680000
        annual_savings = BUSINESS_IMPACT['potential_savings']
        roi_percentage = ((annual_savings - implementation_cost) / implementation_cost) * 100
        payback_months = (implementation_cost / annual_savings) * 12
        
        st.markdown(f"""
        <div class="cost-savings">
        <h4>ROI Analysis Summary</h4>
        <p><strong>Implementation Cost:</strong> ${implementation_cost:,} (AKS infrastructure + ML development)</p>
        <p><strong>Annual Savings:</strong> ${annual_savings:,}</p>
        <p><strong>ROI:</strong> {roi_percentage:.0f}% in first year</p>
        <p><strong>Payback Period:</strong> {payback_months:.1f} months</p>
        <p><strong>3-Year NPV:</strong> ${(annual_savings * 3 - implementation_cost):,}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key learnings and success factors
        st.markdown("#### 🎯 Key Learnings & Success Factors")
        
        learnings = [
            "Feature calibration with domain expert input increased model accuracy by 23%",
            "Business cost framing reduced alert fatigue from 340 to 75 daily alerts",
            "Rolling window validation prevented overfitting to seasonal patterns",
            "Ensemble approach improved robustness compared to single model deployment",
            "Automated retraining pipeline maintains 96%+ accuracy despite data drift"
        ]
        
        for learning in learnings:
            st.markdown(f"""
            <div class="model-performance">
            <p>✅ <strong>Key Learning:</strong> {learning}</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
    <h4 style="color: #ff6b35;">🔧 Smart Grid Predictive Maintenance</h4>
    <p><strong>ML-Powered Asset Management on Azure Kubernetes Service</strong></p>
    <p><em>XGBoost + TensorFlow + scikit-learn • Airflow Orchestration • $2.3M Annual Savings</em></p>
    <br>
    <p style="font-size: 1.1em;"><strong>🎯 146 High-Risk Assets Identified</strong> • <strong>⚡ 100% Recall Rate</strong> • <strong>🚀 34% Crew Efficiency Gain</strong></p>
    </div>
    """, unsafe_allow_html=True)
