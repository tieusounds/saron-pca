# **SARON Swap Rate PCA Analysis**

## 🚀 Overview

This platform is a prototype for analyizing interest rate data - starting with a focus on SARON swap rates. At this stage, the tooluses a single dataset, which cannot be shared or downloaded due to 
copyright restricitons. However, the borader vision is to expand the tool'sfunctionality to allow users to upload and analyze any interst rate dataset of their choice. Alongside the platform, 
I'm currently workingon a non-scientific paper that explors the SAROn swap market and presents insights form the analyses you see here. The paper will be amde availableon this site once finalized. 
Future versions of this platform aim to support customizable research and automated time series anylsis. Stay tuned-Mathieu

## 🎯 Features

- **Interactive 3D Plots**: Visualize your PCA results in a fully interactive 3D environment.
- **Heatmaps**: Beautiful and dynamic heatmaps show the relationship between features and principal components over time.
- **Scree Plot**: Easily identify the number of principal components to retain using an intuitive scree plot.
- **PCA Loadings**: Deep dive into the loadings of each principal component with clean, readable tables.
- **Downloadable Visualizations**: Save and download high-quality visualizations for your reports or presentations.

## 🌍 Why This Project?

First of all, I’ve always had challenges fully understanding the complexities of interest rates, especially how various economic factors interact and affect them over time. 
The abstract nature of financial data often made it difficult for me to grasp the deeper relationships between variables like maturities, dates, and principal components.

Secondly, I’ve found that visual learning is the most effective for me. Numbers and equations are often abstract, but when they’re brought to life through visualizations, 
it’s much easier to understand how everything fits together. That’s why I created these PCA visualizations.

By combining PCA with interactive, visually intuitive plots, this dashboard allows me to explore and understand the underlying patterns in interest rates, and hopefully, 
it can do the same for anyone else facing similar challenges. It’s all about transforming complex data into something that’s easy to interpret and act upon, giving us the power to make smarter decisions with clearer insights.

## 📊 Installation & Setup (Running the app locally)

To get this project up and running locally, follow these simple steps:

### 🛠 Requirements

- Python 3.7+
- `pip` (Package manager)

### 💻 Steps

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/tieusounds/saron-pca.git
    cd saron-pca
    ```

2. Install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    streamlit run pca1.py
    ```

4. Open your browser and navigate to `http://localhost:8501` to explore the interactive dashboard.

## 🔥 How to Use

1. **Choose From Datasets Available**: Currently, onyl the defualt SARON Swap Rate dataset is available.
2. **Select Your Variables**: Choose the features you want to analyze and the return period (1 day, 10 days, etc.).
3. **Explore Visualizations**: Dive into the scree plot, PCA loadings, and 3D surface plots. Rotate, zoom, and explore your data interactively.
4. **Download Visuals**: Need a high-res plot for your reports? Download them directly from the dashboard.

## 🎨 Customization

- Adjust the camera angles and zoom levels for the 3D plots.
- Choose color schemes to fit your branding or style.
- Customize axis labels, font sizes, and more!

## 🧑‍💻 Technologies Used

- **Python**: The backbone of the app, leveraging powerful libraries.
- **Streamlit**: The simplest way to create beautiful data apps with Python.
- **Plotly**: For interactive 3D surface plots and other visualizations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For traditional static visualizations and heatmaps.
- **Scikit-learn**: For PCA and other data analysis techniques.

## Thank You and Happy Coding!
