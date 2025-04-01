import streamlit as st
import pandas as pd
from pca_analysis import perform_pca_analysis_streamlit



# Page configuration (you can switch to "centered" if you prefer)
st.set_page_config(page_title="SARON Swap Rate PCA Analysis", layout="wide")

# Inject custom CSS to limit the container width
st.markdown(
    """
    <style>
    div.plot-container {
        display: flex;
        justify-content: center;
    }
    
    div.plot-container img {
        width: 1000px !important;  /* Adjust width */
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# File path for the dataset
data_file_path = "chf_hist_cleaned.csv"

# Load the dataset
df = pd.read_csv(data_file_path)

# Streamlit App
st.title("SARON Swap Rate PCA Analysis")

st.markdown("### In Short")
st.markdown("""
#### About the Project
This platform is a prototype for analyizing interest rate data - starting with a focus on SARON swap rates. At this stage, the tool 
uses a single dataset, which cannot be shared or downloaded due to copyright restricitons. However, the borader vision is to expand the tool's
functionality to allow users to upload and analyze any interst rate dataset of their choice. Alongside the platform, I'm currently working"
            "on a non-scientific paper that explors the SAROn swap market and presents insights form the analyses you see here. The paper will be amde available
            "on this site once finalized. Future versions of this platform aim to support customizable research and automated time series anylsis. Stay tuned-Mathieu

#### What is Principal Component Analysis?

Principal Component Analysis (PCA) is a statistical technique used to simplify complex datasets by reducing their dimensionality while retaining the most important information. It transforms the original features into a new set of variables called "principal components," which are ordered by the amount of variance they capture from the data.

PCA helps in:
- **Identifying patterns** in data by highlighting correlations between variables.
- **Reducing dimensionality**, making it easier to visualize and analyze large datasets.
- **Improving performance** in machine learning by eliminating noise and redundant information.

In essence, PCA extracts the essential features from the data, allowing you to work with a smaller set of variables without losing critical information.
""")

st.markdown("""
### 📊 Visualizations in the PCA Analysis App

The app provides various interactive and static visualizations to help explore and understand the Principal Component Analysis (PCA) results:

#### 0. Data Characteristics
- **0.1 Original Values Over Time**  
  A visualization of the selected SARON swap returns over time.

- **0.2 Scatter Plot of The Original Values In The PC Space**  
  A scatter plot showing how the original values relate to the first two principal components.

#### 1. Variance
- **1.1 Explained Variance Table**  
  A table displaying how much variance is explained by each principal component.

- **1.2 Scree Plot**  
  The Scree Plot visualizes the explained variance of each principal component in the PCA analysis. The bars represent the individual explained variance for each component, while the step line indicates the cumulative variance. This helps determine the optimal number of principal components to retain.

#### 2. Variable Contributions
- **2.1 Feature Contributions**  
  A table showing how each original variable contributes to the principal components.

- **2.2 PCA vs. SARON Maturity**  
  A line chart visualizing the relationship between the PCA loadings and the selected SARON maturities.

#### 3. Time Analysis
- **3.1 Feature Contributions Over Time**
  - **3.1.1 PC1**  
    A heatmap illustrating the contribution of each original feature to Principal Component 1 over time.
  
  - **3.1.2 PC2**  
    A heatmap illustrating the contribution of each original feature to Principal Component 2 over time.
  
  - **3.1.3 PC3**  
    A heatmap illustrating the contribution of each original feature to Principal Component 3 over time.
""")

st.divider()

# Normalize column names
df.columns = df.columns.str.strip()          # Remove leading/trailing whitespace
df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
df.columns = df.columns.str.replace('-', '_')  # Replace dashes with underscores
df.columns = df.columns.str.lower()            # Convert to lowercase for consistency



st.subheader("Return Period")


st.text("Return = absolute change to the last metric. E.g. 30d equals the absolute change of the chosen metric"
        "compared to the same metric 30 days ago")


# Dropdown for selecting return period
periods = ['1d', '10d', '30d', '90d']
selected_period = st.selectbox("Please select:", periods)

# Iterate through columns to filter based on the selected period
period_columns = [col for col in df.columns if f"{selected_period}" in col]

# Filter SARON columns and extract unique maturities
saron_columns = [col for col in period_columns if "swr" in col]  # Only SARON-related columns
saron_maturities = sorted(
    set([col.split("_")[2] for col in saron_columns]),
    key=lambda x: (x[-1], int(x[:-1]))  # Sort by type first ('m' before 'y'), then numerically
)

st.divider()

st.subheader("SARON Maturity")

st.text("Default maturities are already selected. Click the selection box below to select other "
        "maturities (1 Month - 30 Years")

# Multiselect for SARON maturities with default set to ["1m", "3m", "6m", "9m"]
selected_maturities = st.multiselect(
    "Please select",
    options=["All"] + saron_maturities,  # Allow 'All' option to include all maturities
    default=["All"]  # Default to these maturities
)

# Filter SARON columns by selected maturities
if "All" not in selected_maturities:
    saron_columns = [
        col for col in saron_columns if any(f"_{maturity}_" in col for maturity in selected_maturities)
    ]

# Sort SARON columns by maturity for consistent ordering
saron_columns = sorted(
    saron_columns,
    key=lambda col: (col.split("_")[2][-1], int(col.split("_")[2][:-1]))  # Ensures 'm_' maturities come before 'y_'
)


# Update the list of period columns to include filtered SARON columns
period_columns = [
                     col for col in period_columns if "swr" not in col
                 ] + saron_columns

st.divider()

st.subheader("Additional Features")

st.text("If wanted, select additional features below:")

# Define categories for further filtering
feature_categories = {
    "SARON Swap": [col for col in period_columns if "swr" in col],
    "Gov Bond": [col for col in period_columns if "bundesobli" in col],
    "2Y vs. 10Y": [col for col in period_columns if "2y vs" in col or "spread" in col],
    "Kassasatz": [col for col in period_columns if "kassasatz" in col],
}

# Multiselect for feature category selection with default set to ["SARON Swap"]
selected_categories = st.multiselect(
    "Select feature categories to include:",
    list(feature_categories.keys()),
    default=["SARON Swap"]  # Default to only SARON Swap category
)

# Filter features based on the selected categories
filtered_features = [
    feature for category in selected_categories for feature in feature_categories[category]
]

st.divider()

st.write("Selected Features:", filtered_features)

# Initialize session state for selected features
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

st.divider()

# Update session state
st.session_state.selected_features = filtered_features

# Perform PCA analysis if features are selected
if st.session_state.selected_features:
    perform_pca_analysis_streamlit(df, st.session_state.selected_features)
else:
    st.error("Please select at least one feature for PCA analysis.")

st.markdown("""
    <style>
        .footer {
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #333;
        }
    </style>
    <div class="footer">
        <p>© 2025 Mathieu Bitz</p>
    </div>
""", unsafe_allow_html=True)
