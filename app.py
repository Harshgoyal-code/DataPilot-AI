"""
app.py - Main Streamlit Dashboard for DataPilot AI

This is the entry point for our Streamlit web application. It handles the user interface,
layout structure, and displays the exploratory data analysis (EDA) results using functions
imported from our backend helper file eda.py.
"""

import streamlit as st
import pandas as pd
import eda  # Import our helper functions from eda.py
import google.generativeai as genai
import os

# --- Page Configuration ---
# We set the page title, icon, and layout width. This must be the first Streamlit command.
st.set_page_config(
    page_title="DataPilot AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
# Streamlit allows us to inject custom CSS to style elements for a more premium design.
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--text-color, #1E3A8A);
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: var(--text-color, #4B5563);
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1));
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid var(--primary-color, #4F46E5);
        color: var(--text-color, inherit);
    }
    .metric-card h4 {
        color: var(--text-color, inherit) !important;
        opacity: 0.8;
        margin: 0;
    }
    .metric-card h2 {
        color: var(--text-color, inherit) !important;
        margin: 5px 0 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Component ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/airplane-take-off.png", width=120)
    st.title("DataPilot AI ✈️")
    st.markdown("Upload your dataset and get AI-powered data analysis, interactive visualizations, and machine learning recommendations.")
    st.markdown("---")
    
    st.header("1. Upload Dataset")
    # File uploader widget. It restricts input to CSV files.
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    
    st.markdown("---")
    st.header("2. Quick Demo")
    # A button to load a built-in mock dataset so users can explore the app instantly.
    demo_button = st.button("Load Sample Dataset")

    st.markdown("---")
    st.header("3. AI Analyst Key")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Required only for the '🤖 AI Analyst' tab. Get a free key from Google AI Studio.")

# --- Sample Dataset Generation ---
# Initialize session state keys if not already present
if "dataset_source" not in st.session_state:
    st.session_state["dataset_source"] = None
if "demo_df" not in st.session_state:
    st.session_state["demo_df"] = None
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

# If a new file is uploaded or file uploader is cleared, detect the change
if uploaded_file != st.session_state["last_uploaded_file"]:
    st.session_state["last_uploaded_file"] = uploaded_file
    if uploaded_file is not None:
        st.session_state["dataset_source"] = "uploaded"
    else:
        # If the file was cleared and we were using the uploaded file, reset source
        if st.session_state["dataset_source"] == "uploaded":
            st.session_state["dataset_source"] = None

# Determine the active dataset based on the current user action or saved state
if demo_button:
    # Creating a sample DataFrame about employee performance for visualization
    demo_data = {
        "Employee ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Age": [28, 34, 45, 23, 31, 38, 52, 40, 27, 36],
        "Years of Experience": [3, 9, 18, 1, 6, 11, 25, 14, 2, 8],
        "Performance Score": [78, 85, 92, 60, 74, 88, 95, 82, 69, 81],
        "Department": ["Sales", "Engineering", "HR", "Sales", "Engineering", "Engineering", "Executive", "HR", "Sales", "Engineering"],
        "Remote Worker": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No", "No", "Yes", "No"]
    }
    df = pd.DataFrame(demo_data)
    st.session_state["demo_df"] = df
    st.session_state["dataset_source"] = "demo"
    st.sidebar.success("Loaded Employee Performance demo dataset!")
elif st.session_state["dataset_source"] == "uploaded" and uploaded_file is not None:
    # If the user uploaded a file, we pass it to the loader function from eda.py
    df = eda.load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("Successfully loaded CSV file!")
    else:
        st.sidebar.error("Error loading CSV. Make sure it is a valid, non-empty file.")
        df = None
        st.session_state["dataset_source"] = None
elif st.session_state["dataset_source"] == "demo" and st.session_state["demo_df"] is not None:
    df = st.session_state["demo_df"]
else:
    df = None

# --- Helper for AI Analyst Prompt ---
def prepare_dataset_summary_prompt(df, summary):
    """
    Constructs a textual summary of the dataset for the Gemini model.
    """
    dtypes_str = summary["data_types"].to_string(index=False)
    missing_str = summary["missing_values"].to_string(index=False)
    if len(summary["numeric_cols"]) > 0:
        desc_stats_str = df.describe().to_string()
    else:
        desc_stats_str = "No numeric columns available."
    sample_str = df.head(3).to_string()
    
    return f"""You are an expert AI Data Analyst. You are assisting a user in analyzing their dataset.
Here is the detailed summary of the active dataset:

--- DATASET DIMENSIONS ---
- Number of rows: {summary["num_rows"]}
- Number of columns: {summary["num_cols"]}

--- COLUMNS & DATA TYPES ---
{dtypes_str}

--- MISSING VALUES ---
{missing_str}

--- DESCRIPTIVE STATISTICS (NUMERIC) ---
{desc_stats_str}

--- SAMPLE ROWS (FIRST 3 ROWS) ---
{sample_str}

When answering, analyze the context of this dataset, provide helpful answers, identify patterns, and answer questions accurately.
Your response MUST strictly adhere to these rules:
- Maximum 6 bullet points in the entire response.
- Maximum 180 words total.
- Give practical recommendations and focus on actionable insights.
- Avoid long paragraphs completely.
"""

# --- Main Dashboard Area ---
if df is not None:
    # Title Section
    st.markdown('<h1 class="main-title">DataPilot AI Dashboard ✈️</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <span style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); color: var(--text-color, inherit); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-right: 8px; border: 1px solid rgba(128,128,128,0.2); display: inline-block;">🤖 Powered by Gemini AI</span>
            <span style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); color: var(--text-color, inherit); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; border: 1px solid rgba(128,128,128,0.2); display: inline-block;">📊 Automated Exploratory Data Analysis</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Navigate your data journey with smooth summaries and beautiful charts.</p>', unsafe_allow_html=True)
    
    # Calculate dataset summaries using eda.py
    summary = eda.get_basic_summary(df)
    
    # --- Top KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Rows Count</h4><h2>{summary["num_rows"]}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Columns Count</h4><h2>{summary["num_cols"]}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>Numeric Columns</h4><h2>{len(summary["numeric_cols"])}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4>Categorical Columns</h4><h2>{len(summary["categorical_cols"])}</h2></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Interactive Tabs Interface ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview & Statistics", "📊 Visualizations", "💡 Data Pilot Insights", "🤖 AI Analyst"])
    
    # --- TAB 1: OVERVIEW & DATA PREVIEW ---
    with tab1:
        st.header("Dataset Overview")
        
        # Display dataset preview
        st.subheader("Data Preview (Head)")
        num_preview_rows = st.slider("Select number of rows to preview", min_value=5, max_value=50, value=10)
        st.dataframe(df.head(num_preview_rows), use_container_width=True)
        
        # Split layout for data types and missing values
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Data Types Details")
            st.dataframe(summary["data_types"], use_container_width=True)
            
        with col_right:
            st.subheader("Missing Values Count")
            st.dataframe(summary["missing_values"], use_container_width=True)
            
        # Display descriptive statistics for numeric columns
        st.subheader("Descriptive Statistics")
        if len(summary["numeric_cols"]) > 0:
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns found in this dataset to show descriptive statistics.")

    # --- TAB 2: INTERACTIVE VISUALIZATIONS ---
    with tab2:
        st.header("Interactive Data Plots")
        st.write("Generate beautiful plots dynamically by selecting variables below.")
        
        plot_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution of a Single Column", "Scatter Plot (Relationship Between Columns)", "Correlation Matrix Heatmap"]
        )
        
        # 1. Distribution Plot
        if plot_type == "Distribution of a Single Column":
            if len(summary["numeric_cols"]) > 0:
                selected_col = st.selectbox("Select Numeric Column to Analyze", summary["numeric_cols"])
                # Generate distribution plot using eda.py helper
                fig = eda.generate_distribution_plot(df, selected_col)
                st.pyplot(fig)
            else:
                st.warning("You need at least one numeric column to draw a distribution plot.")
                
        # 2. Scatter Plot
        elif plot_type == "Scatter Plot (Relationship Between Columns)":
            if len(summary["numeric_cols"]) >= 2:
                col_x = st.selectbox("Select X-Axis Column (Numerical)", summary["numeric_cols"], index=0)
                col_y = st.selectbox("Select Y-Axis Column (Numerical)", summary["numeric_cols"], index=min(1, len(summary["numeric_cols"])-1))
                
                # Generate scatter plot using eda.py helper
                fig = eda.generate_scatter_plot(df, col_x, col_y)
                st.pyplot(fig)
            else:
                st.warning("You need at least two numeric columns to draw a scatter plot.")
                
        # 3. Correlation Heatmap
        elif plot_type == "Correlation Matrix Heatmap":
            fig = eda.generate_correlation_heatmap(df)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("You need at least two numeric columns to draw a correlation matrix heatmap.")

    # --- TAB 3: DATA PILOT INSIGHTS ---
    with tab3:
        st.header("Data Pilot AI Automated Insights 💡")
        st.write("Here are some automatically generated insights and warnings regarding your dataset:")
        
        # Insight 1: Missing Data Alert
        missing_count_total = df.isnull().sum().sum()
        if missing_count_total > 0:
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            st.warning(f"⚠️ **Missing Values Alert**: Your dataset contains a total of **{missing_count_total}** missing values across columns: `{', '.join(cols_with_missing)}`.")
            st.info("💡 **Recommendation**: Consider checking these columns. You can fill missing numeric data with the median/mean values or drop records with missing targets.")
        else:
            st.success("✅ **No Missing Values**: Clean flight ahead! No missing entries detected in any columns.")
            
        # Insight 2: High Correlation Detection
        if len(summary["numeric_cols"]) >= 2:
            corr = df[summary["numeric_cols"]].corr()
            high_corr_pairs = []
            
            # Extract correlation pairs that are highly correlated (> 0.7 or < -0.7) and not self-correlated
            for i in range(len(corr.columns)):
                for j in range(i):
                    val = corr.iloc[i, j]
                    if abs(val) >= 0.7:
                        high_corr_pairs.append((corr.columns[i], corr.columns[j], val))
            
            if len(high_corr_pairs) > 0:
                st.info("📈 **Strong Linear Relationships Detected**:")
                for c1, c2, val in high_corr_pairs:
                    st.write(f"- `{c1}` and `{c2}` have a correlation coefficient of **{val:.2f}**.")
                st.info("💡 **Recommendation**: High correlation can lead to multicollinearity in machine learning. Consider dropping or combining one of these columns if you plan to build linear predictive models.")
            else:
                st.success("✅ **Independent Numeric Features**: No extremely strong correlations (r >= 0.7) detected. Features seem highly independent.")
        else:
            st.info("📊 Correlation insights require at least 2 numerical columns.")
            
        # Insight 3: Imbalance check for categorical columns
        if len(summary["categorical_cols"]) > 0:
            st.subheader("Categorical Features Breakdown")
            for col in summary["categorical_cols"][:3]:  # Display top 3 categorical columns at most to avoid clutter
                unique_vals = df[col].nunique()
                most_freq = df[col].mode()[0]
                pct_most_freq = (df[col] == most_freq).sum() / len(df) * 100
                
                st.markdown(f"**Column: `{col}`**")
                st.write(f"- Unique Categories: **{unique_vals}**")
                st.write(f"- Most common category: **'{most_freq}'** (appears in **{pct_most_freq:.1f}%** of rows)")
                
                if pct_most_freq > 80:
                    st.warning(f"⚠️ **High Imbalance**: `{col}` is dominated by **'{most_freq}'** ({pct_most_freq:.1f}% of data). Models training on this column might struggle to learn about other categories.")
        else:
            st.info("ℹ️ No categorical features found to analyze class balance.")

    # --- TAB 4: AI ANALYST ---
    with tab4:
        st.header("🤖 AI Data Analyst")
        st.markdown("""
        💡 **How to use this tool:**
        Select one of the analysis tasks below to query Gemini AI. The model will analyze your dataset summary and answer based on the following types of analytical questions:
        - *Which feature column has the highest rate of missing values?*
        - *Are there any extreme outliers or data type errors in this table?*
        - *What machine learning models fit best for classifying remote vs office workers?*
        - *How should I clean and normalize numeric features for training?*
        """)
        
        if not api_key:
            st.warning("⚠️ **Gemini API Key Missing**: Please enter your Gemini API key in the sidebar to use the AI Analyst features.")
            st.info("You can get a free API Key from [Google AI Studio](https://aistudio.google.com/).")
        else:
            st.write("Click a button below to generate instant analysis insights using Gemini AI:")
            
            # Create columns for the 4 action buttons
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            
            query_type = None
            prompt_suffix = ""
            
            with col_b1:
                if st.button("🔍 Find Data Problems", use_container_width=True):
                    query_type = "Find Data Problems"
                    prompt_suffix = "Identify any potential problems in the dataset, such as high missingness, outlier columns, class imbalance, or suspicious data types. Give a summary list of problems."
                    
            with col_b2:
                if st.button("🤖 Suggest Best ML Model", use_container_width=True):
                    query_type = "Suggest Best ML Model"
                    prompt_suffix = "Suggest suitable machine learning model types for this dataset depending on different possible target columns (regression vs classification). Recommend a few models and explain why."
                    
            with col_b3:
                if st.button("🧹 Recommend Data Cleaning", use_container_width=True):
                    query_type = "Recommend Data Cleaning Steps"
                    prompt_suffix = "Provide a specific step-by-step checklist to clean and prepare this dataset for analysis or machine learning, focusing on handling missing values, encoding categoricals, scaling, etc."
                    
            with col_b4:
                if st.button("📊 Summarize Dataset", use_container_width=True):
                    query_type = "Summarize Dataset"
                    prompt_suffix = "Provide a high-level executive summary of this dataset, highlighting its structure, main variables, interesting statistics, and overall health."
            
            if query_type and prompt_suffix:
                st.markdown("---")
                st.subheader(f"Analysis Results: {query_type}")
                with st.spinner("Analyzing dataset with Gemini..."):
                    try:
                        # Configure API key dynamically
                        genai.configure(api_key=api_key)
                        
                        # Prepare prompt context
                        sys_instruction = prepare_dataset_summary_prompt(df, summary)
                        
                        # Initialize model
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        
                        # Call API with single-shot prompt
                        response = model.generate_content(f"{sys_instruction}\n\nTask: {prompt_suffix}")
                        
                        # Display response
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"❌ **Error calling Gemini API**: {str(e)}")
                        st.info("Please double-check your API key or network connection.")


else:
    # Landing Page if no file is uploaded yet
    st.markdown('<h1 class="main-title">Welcome to DataPilot AI ✈️</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <span style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); color: var(--text-color, inherit); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-right: 8px; border: 1px solid rgba(128,128,128,0.2); display: inline-block;">🤖 Powered by Gemini AI</span>
            <span style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); color: var(--text-color, inherit); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; border: 1px solid rgba(128,128,128,0.2); display: inline-block;">📊 Automated Exploratory Data Analysis</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An automated data visualization and exploratory analysis tool.</p>', unsafe_allow_html=True)
    
    st.info("👉 **Get Started**: Upload a CSV file in the sidebar to begin or click **'Load Sample Dataset'** in the sidebar to test with a built-in dataset!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧭 Interactive Workflow")
    
    col_wf1, col_wf2, col_wf3, col_wf4 = st.columns(4)
    with col_wf1:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); border-radius: 8px; padding: 15px; text-align: center; border-bottom: 4px solid #4F46E5; height: 110px;">
            <h4 style="margin: 0; color: var(--text-color, inherit);">1. Upload CSV 📂</h4>
            <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; color: var(--text-color, inherit);">Drop your dataset in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    with col_wf2:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); border-radius: 8px; padding: 15px; text-align: center; border-bottom: 4px solid #06B6D4; height: 110px;">
            <h4 style="margin: 0; color: var(--text-color, inherit);">2. Analyze Dataset 📊</h4>
            <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; color: var(--text-color, inherit);">Explore stats and auto-generated plots</p>
        </div>
        """, unsafe_allow_html=True)
    with col_wf3:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); border-radius: 8px; padding: 15px; text-align: center; border-bottom: 4px solid #10B981; height: 110px;">
            <h4 style="margin: 0; color: var(--text-color, inherit);">3. Ask AI 🤖</h4>
            <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; color: var(--text-color, inherit);">Query the Gemini AI Data Analyst</p>
        </div>
        """, unsafe_allow_html=True)
    with col_wf4:
        st.markdown("""
        <div style="background-color: var(--secondary-background-color, rgba(128,128,128,0.1)); border-radius: 8px; padding: 15px; text-align: center; border-bottom: 4px solid #F59E0B; height: 110px;">
            <h4 style="margin: 0; color: var(--text-color, inherit);">4. Get ML Recommendations 💡</h4>
            <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; color: var(--text-color, inherit);">Get model and prep tips from the AI</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature list grid
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown("### 📋 1. Instant Data Summary")
        st.write("Easily view missing values, data types, dimensions, and descriptive statistics.")
    with col_feat2:
        st.markdown("### 📊 2. Dynamic Charts")
        st.write("Generate distributions, scatter plots, and correlation heatmaps with simple click controls.")
    with col_feat3:
        st.markdown("### 💡 3. Automated Insights")
        st.write("Identify data quality issues, collinearity warnings, and categorical imbalances instantly.")
        
    st.markdown("---")
    st.write("""
DataPilot AI helps users analyze datasets, generate visualizations and receive AI-powered insights for faster exploratory data analysis.""")

# --- Global Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: var(--text-color, inherit); opacity: 0.7; font-size: 0.9rem; padding: 15px 0;">
        <p style="margin: 0; font-weight: 600;">Developed by Harsh Goyal</p>
        <p style="margin: 5px 0 0 0; font-size: 0.8rem;">Built with Python • Pandas • Streamlit • Gemini AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
