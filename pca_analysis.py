
#from io import BytesIO
#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib.pyplot import ticklabel_format
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.colors import TwoSlopeNorm
#import plotly.graph_objects as go



#################################### STREAMLIT ######################################
# Function: Convert Figures to Image for Streamlit
def fig_to_image(fig, dpi=300, width=650):
    #buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)  # Adjust DPI for sharpness
    buf.seek(0)
    return buf

#################################### PCA COMPUTATION #########################################

# Function: Compute PCA and return results
def compute_pca(df, selected_features):
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        df_features = df[selected_features]
    else:
        st.error("The 'date' column is missing. Ensure it exists for plotting.")
        return None, None, None, None

    # Standardize Data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # PCA Loadings
    loadings = pd.DataFrame(
        pca.components_,
        columns=selected_features,
        index=[f"PC{i+1}" for i in range(len(pca.components_))]
    )


    return pca, pca_result, loadings, explained_variance, cumulative_variance, df_features, dates

################################ PLOTS #######################################


# Function: Plot Scree Plot
def plot_scree(pca):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xlabel('Principal Component Index')
    ax.set_title('Scree Plot')
    ax.legend(loc='best')

    st.image(fig_to_image(fig,1500), caption="The Scree Plot visualizes the explained variance of each principal"
                                        " component in the PCA analysis. The bars represent the individual"
                                        " explained variance for each component, while the step line indicates"
                                        " the cumulative variance. This helps determine the optimal number"
                                        " of principal components to retain.", width=650)


# Function: Plot PCA Biplot
def plot_biplot(pca, pca_result, df_features):
    pc_scores = pca_result[:, :2]
    pc_loadings = pca.components_[:2, :]
    scaling_factor = np.max(np.abs(pc_scores)) / np.max(np.abs(pc_loadings))
    pc_loadings_scaled = pc_loadings * scaling_factor

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.5, label='Data Projections')

    for i, var in enumerate(df_features.columns):
        ax.arrow(0, 0, pc_loadings_scaled[0, i], pc_loadings_scaled[1, i], color='red', alpha=0.75,
                 head_width=0.02 * scaling_factor, head_length=0.05 * scaling_factor)
        ax.text(pc_loadings_scaled[0, i] * 1.1, pc_loadings_scaled[1, i] * 1.1, var, color='darkblue', fontsize=8)

    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_title('PCA Biplot')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(alpha=0.3)
    ax.legend()

    st.image(fig_to_image(fig,1500), caption="The PCA Biplot displays the first two principal components,"
                                        " showing both the data projections (scatter points)"
                                        " and the feature loadings (red arrows). The arrows "
                                        "indicate how strongly each feature contributes to th"
                                        "e principal components, helping to interpret feature "
                                        "relationships in the reduced space.", width=650)


# Function: Plot Heatmaps of Feature Contributions
# def plot_heatmaps(pca, pca_result, df_features, dates):
#
#     contributions_df = pd.DataFrame(
#         pca_result,
#         columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
#         index=dates
#     )
#
#     def plot_pc_contribution_heatmap(pc_number, title):
#         fig, ax = plt.subplots(figsize=(7, 15))
#         formatted_dates = dates.dt.date
#         heatmap_data = pd.DataFrame(
#             np.outer(contributions_df[f'PC{pc_number}'], pca.components_[pc_number - 1]),
#             index=formatted_dates,
#             columns=df_features.columns
#         )
#         sns.heatmap(
#             heatmap_data,
#             cmap='Spectral',
#             cbar=True,
#             # vmin=-10,
#             # vmax=10,
#             ax=ax
#         )
#         ax.set_title(f'{title} (PC{pc_number})', fontsize=10)
#         ax.set_xlabel('Features')
#         ax.set_ylabel('Dates')
#
#         st.image(fig_to_image(fig,1000), caption=f"This heatmap illustrates the contribution of each original feature"
#                                             " to the " f" {title} " "over time. Red and blue colors"
#                                             " indicate strong positive and negative contributions, respectively, "
#                                             "helping to identify dominant features influencing " f" {title} "".", width=650)
#
#     plot_pc_contribution_heatmap(1, 'Feature Contributions to Principal Component 1')
#     plot_pc_contribution_heatmap(2, 'Feature Contributions to Principal Component 2')
#     plot_pc_contribution_heatmap(3, 'Feature Contributions to Principal Component 3')


def plot_heatmap(pca, pca_result, df_features, dates, pc_number):
    """
    Plots a heatmap for a single Principal Component (PC).

    Parameters:
    - pca: Trained PCA model
    - pca_result: Transformed PCA scores
    - df_features: Original feature DataFrame before PCA transformation
    - dates: Date index for the heatmap
    - pc_number: Principal Component number (1, 2, 3, ...)
    """

    # Compute contributions
    contributions_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=dates
    )

    # Create heatmap data
    fig, ax = plt.subplots(figsize=(4, 4))
    formatted_dates = dates.dt.date
    heatmap_data = pd.DataFrame(
        np.outer(contributions_df[f'PC{pc_number}'], pca.components_[pc_number - 1]),
        index=formatted_dates,
        columns=df_features.columns
    )

    sns.heatmap(
        heatmap_data,
        cmap='Spectral',
        cbar=True,
        ax=ax,
        cbar_kws={'label': 'Contribution Magnitude', 'shrink': 0.5}  # Adjust colorbar size and label
    )
    ax.collections[0].colorbar.ax.tick_params(labelsize=5)  # Adjust colorbar tick font size
    ax.collections[0].colorbar.set_label("Contribution Magnitude", fontsize=8)  # Colorbar label font size


    ax.set_title(f'Feature Contributions to Principal Component {pc_number}', fontsize=8)
    ax.set_xlabel('Features', fontsize=8)
    ax.set_ylabel('Dates', fontsize=8)
    ax.tick_params(labelsize=5)

    # Display in Streamlit
    st.image(fig_to_image(fig, 1000), caption=f"This heatmap illustrates the contribution of each original feature "
                                              f"to Principal Component {pc_number} over time.", width=650)



def plot_pca_loadings_linechart(loadings):
    """
    Plots PCA loadings for the first three principal components as a line chart.
    X-axis: Extracted maturities (sorted within 'm' and 'y')
    Y-axis: Loadings
    """



    # Plot the first three principal components
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(3):  # PC1, PC2, PC3
        ax.plot(loadings.columns, loadings.iloc[i], marker='o', label=f'PC{i+1}')

    ax.set_title("PCA Loadings by Maturity")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Loading Value")
    ax.legend()
    ax.grid(True)

    ax.set_xticklabels(loadings.columns, rotation=90)

    st.image(fig_to_image(fig,1500), caption="", width=650)

# def plot_z0_intersection(ax, X, Y, Z, color='black'):
#     """
#     Plots the intersection of Z=0 with the surface (X,Y,Z) as a set of points in 3D.
#     X, Y, Z are 2D arrays from np.meshgrid.
#     """
#
#     nrows, ncols = Z.shape
#     xs, ys, zs = [], [], []
#
#     # Check horizontal edges
#     for i in range(nrows):
#         for j in range(ncols - 1):
#             z1, z2 = Z[i, j], Z[i, j+1]
#             if z1 * z2 < 0:  # sign change => crossing z=0
#                 # Linear interpolation factor
#                 t = abs(z1) / (abs(z1) + abs(z2))
#                 x = X[i, j] + t*(X[i, j+1] - X[i, j])
#                 y = Y[i, j] + t*(Y[i, j+1] - Y[i, j])
#                 xs.append(x)
#                 ys.append(y)
#                 zs.append(0)
#
#     # Check vertical edges
#     for j in range(ncols):
#         for i in range(nrows - 1):
#             z1, z2 = Z[i, j], Z[i+1, j]
#             if z1 * z2 < 0:
#                 t = abs(z1) / (abs(z1) + abs(z2))
#                 x = X[i, j] + t*(X[i+1, j] - X[i, j])
#                 y = Y[i, j] + t*(Y[i+1, j] - Y[i, j])
#                 xs.append(x)
#                 ys.append(y)
#                 zs.append(0)
#
#     # Plot as a small scatter (or lines if you want to connect them)
#     ax.scatter(xs, ys, zs, c=color, s=10)


def plot_pca_3d_surface(contributions_df, dates, df_features, loadings, pc_number):

    """
    Creates a 3D surface plot representing the feature contributions for a specific Principal Component.

    Parameters:
    - contributions_df: DataFrame containing feature contributions for each principal component.
    - dates: Date index for the y-axis.
    - features: Feature names (maturities) for the x-axis.
    - pc_number: Integer representing the Principal Component (1, 2, 3, ...).
    """

    fig = plt.figure(figsize=(16, 9))  # Set to landscape aspect ratio
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(np.arange(len(df_features)), np.arange(len(dates)))


    heatmap_data = pd.DataFrame(
        np.outer(contributions_df[f'PC{pc_number}'], loadings.loc[f'PC{pc_number}']),
        index=dates,
        columns=df_features
    )

    Z = heatmap_data.values  # Use correctly shaped array

    # Plot surface with transparency
    ls = LightSource(270, 45)
    # rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

    # Suppose your data roughly ranges between -3 and +3
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='Spectral',
        norm=norm,         # Use the TwoSlopeNorm
        rstride=1,
        cstride=1,
        linewidth=0.03,
        antialiased=True,
        shade=False,
        alpha=1,
        edgecolor='black'
    )

    ##### Activating rstride enables a color jump in both the x and y direction, as data is downsampled. Makes sense
    ##### for higher visibility

    # Projection of the loadings down on the x/y surfae (= z.min)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='Spectral', linewidths=1, alpha=1)

    # Set labels
    # ax.set_xlabel("Features (Maturities)")
    # ax.set_ylabel("Time (Dates)")
    ax.set_zlabel("Contribution Magnitude")
    ax.set_title(f"3D Feature Contributions to Principal Component {pc_number}")

    # Rotate for a better frontal view of the date axis
    ax.view_init(elev=10, azim=20)

    # Set aspect ratio to make the date axis twice as long
    ax.set_box_aspect([4, 6, 2])

    # Adjust axis ticks
    ax.set_xticks(np.arange(len(df_features))[::max(1, len(df_features) // 10)])
    ax.set_xticklabels(df_features[::max(1, len(df_features) // 10)], fontsize=6, rotation=-45)
    ax.set_yticks(np.arange(len(dates))[::max(1, len(dates) // 30)])
    ax.set_yticklabels(pd.to_datetime(dates[::max(1, len(dates) // 30)]), fontsize=6, rotation=90)


    # Add color bar
    # fig.colorbar(surf, shrink=0.3, aspect=10, label="Contribution Magnitude", )

    # Optionally, allow downloading
    st.image(fig_to_image(fig, 1000), caption=f"3D Feature Contributions to Principal Component {pc_number}", width=1250)



def plot_pca_3d_surface_interactive(contributions_df, dates, df_features, loadings, pc_number, flip_axis=False):
    """
    Creates an interactive 3D surface plot in Plotly for a specific Principal Component (PC).

    Parameters:
    ----------
    contributions_df : pd.DataFrame
        DataFrame with columns ['PC1', 'PC2', ...], indexed by dates (rows).
        Example: contributions_df['PC1'] = PCA scores for the first PC.
    dates : pd.Series or list-like
        Date index for the Y-axis.
    df_features : pd.DataFrame
        The original feature DataFrame (or at least something that has .columns in the
        correct short-to-long maturity order).
    loadings : pd.DataFrame
        PCA loadings with index like ['PC1', 'PC2', ...] and columns matching your feature names.
    pc_number : int
        Which principal component to plot (1 for PC1, 2 for PC2, etc.).
    flip_axis : bool
        If True, reverses the X-axis order (long maturities to left, short to right).

    Returns:
    -------
    None (displays the plot in Streamlit).
    """

    # 1) Align loadings columns with df_features columns
    #    so they match the 2D approach exactly.
    #    (Assumes loadings has the same columns as df_features.)
    matching_cols = df_features.columns
    my_loadings = loadings.loc[f'PC{pc_number}', matching_cols]

    # 2) Create the same outer product as 2D heatmap:
    #    outer(scores, loadings) => contributions over time × features
    heatmap_data = pd.DataFrame(
        np.outer(contributions_df[f'PC{pc_number}'], my_loadings),
        index=dates,
        columns=matching_cols
    )

    # 3) Optionally flip the X-axis if desired
    if flip_axis:
        heatmap_data = heatmap_data[heatmap_data.columns[::-1]]

    # Convert to arrays for Plotly
    Z = heatmap_data.values
    xvals = np.arange(len(heatmap_data.columns))  # 0,1,2,... for features
    yvals = np.arange(len(heatmap_data.index))    # 0,1,2,... for dates

    # Optional color range:
    cmin, cmax = -2, 2  # Adjust as needed

    # 4) Build the 3D surface
    fig = go.Figure(data=[
        go.Surface(
            x=xvals,
            y=yvals,
            z=Z,
            colorscale="Spectral",
            cmin=cmin,
            cmax=cmax,
            showscale=True
        )
    ])

    # 5) Customize layout
    fig.update_layout(
        title=f"3D Feature Contributions to Principal Component {pc_number}",
        width=1000,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="lightgrey",
        plot_bgcolor="white",
        font=dict(size=11, color="black"),
        scene=dict(
            aspectratio=dict(x=1, y=2.5, z=1),
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis=dict(
                title="Features",
                tickmode="array",
                tickvals=xvals,
                ticktext=heatmap_data.columns  # matches the same order as your 2D heatmap
            ),
            yaxis=dict(
                title="Dates",
                tickmode="array",
                # Sample a subset of date ticks if you have many rows
                tickvals=yvals[::max(1, len(yvals)//30)],
                ticktext=[str(d.date()) for d in dates[::max(1, len(yvals)//30)]]
            ),
            zaxis=dict(title="Contribution Magnitude"),
        )
    )

    # 6) Display in Streamlit
    st.plotly_chart(fig, use_container_width=False)


import numpy as np
import plotly.graph_objects as go
import streamlit as st

def plot_pca_3d_scatter(pca_result, dates=None):
    """
    Plots a 3D scatter of data points projected into the first three principal components (PC1, PC2, PC3).

    Parameters
    ----------
    pca_result : np.ndarray
        A 2D array of shape (n_samples, n_components). The first three columns are assumed
        to be the PC1, PC2, PC3 scores for each data point.
    dates : array-like, optional
        A list/array of date labels (or any string labels) for hover tooltips. Must have length n_samples
        if provided. Defaults to None.
    """

    # Ensure we have at least three components
    if pca_result.shape[1] < 3:
        st.error("Your PCA result has fewer than 3 components. Please compute at least 3 to plot 3D scatter.")
        return

    # Extract the three principal component scores
    pc1_scores = pca_result[:, 0]
    pc2_scores = pca_result[:, 1]
    pc3_scores = pca_result[:, 2]

    # Prepare hover texts
    if dates is not None:
        # Convert each date (or other label) to a string for better display in Plotly's hover
        hover_texts = [str(d) for d in dates]
    else:
        hover_texts = [f"Point {i}" for i in range(len(pc1_scores))]

    # Create a 3D scatter trace
    scatter_3d = go.Scatter3d(
        x=pc1_scores,
        y=pc2_scores,
        z=pc3_scores,
        mode='markers',
        marker=dict(
            size=5,
            color=pc1_scores,  # color by PC1 scores, or choose another dimension
            colorscale='Spectral',
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text'
    )

    # Build the Plotly figure
    fig = go.Figure(data=[scatter_3d])

    # Customize the layout
    fig.update_layout(
        title="Data Points in PC1-PC2-PC3 Space",
        width=800,
        height=600,
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        margin=dict(l=10, r=10, b=10, t=30)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=False)



def plot_original_time_3d_surface_interactive(dates, df_features, flip_axis=False):
    """
    Creates an interactive 3D surface plot in Plotly for original (non-absolute-change) values
    over time, given:
      - 'dates': a sequence aligned with the rows of 'df_features'
      - 'df_features': a DataFrame of numeric columns (one for each maturity),
        EXCLUDING the Date column itself.

    Parameters
    ----------
    dates : pd.Series or list-like
        A sequence of dates (or datetimes) that correspond 1:1 to the rows of df_features.
    df_features : pd.DataFrame
        A DataFrame of only numeric columns (the selected features/maturities),
        with one row per date in 'dates'.

    Returns
    -------
    None (displays the Plotly figure in Streamlit).

    Usage
    -----
    1) Up in your pca1.py (or other script), filter your main 'df' to just the rows/columns you want:
       - Align 'dates' = df_filtered["Date"]
       - Align 'df_features' = df_filtered.drop("Date", axis=1)
       (Ensure the row order matches exactly.)

    2) Call:
       plot_original_time_3d_surface_interactive(dates, df_features)
    """

    # Basic checks
    if len(dates) != len(df_features):
        st.error("Length mismatch: 'dates' and 'df_features' must have the same number of rows.")
        return

    if df_features.select_dtypes(include=[np.number]).shape[1] == 0:
        st.warning("No numeric columns in df_features to plot.")
        return

    # 1) Convert 'dates' to a list of strings for labeling
    #    (If they're already strings, this has no effect.)
    date_strings = [str(d) for d in dates]

    if flip_axis:
        df_features = df_features[df_features.columns[::-1]]

    # 2) Build the Z array: shape (n_rows, n_features)
    Z = df_features.values  # Each row corresponds to one date, each column is one feature
    n_rows, n_cols = Z.shape

    # 3) X-axis = columns of df_features, Y-axis = each row (date)
    xvals = np.arange(n_cols)
    yvals = np.arange(n_rows)



    # 4) Create the Plotly surface
    fig = go.Figure(data=[
        go.Surface(
            x=xvals,
            y=yvals,
            z=Z,
            colorscale="Spectral",
            showscale=True
        )
    ])

    # 5) Configure axis ticks
    # Subsample date ticks if there are many rows
    max_ticklabels = 20
    step_rows = max(1, n_rows // max_ticklabels)

    fig.update_layout(
        title="3D Original Values Over Time",
        width=1000,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=11, color="black"),
        scene=dict(
            aspectratio=dict(x=1, y=2.5, z=1),
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis=dict(
                title="Selected Features",
                tickmode="array",
                tickvals=xvals,
                ticktext=df_features.columns  # e.g. ["SARON_SWR_1M", "SARON_SWR_3M", ...]
            ),
            yaxis=dict(
                title="Dates",
                tickmode="array",
                tickvals=yvals[::step_rows],
                ticktext=date_strings[::step_rows]
            ),
            zaxis=dict(title="Rate (original)"),
        )
    )

    # 6) Display in Streamlit
    st.plotly_chart(fig, use_container_width=False)


######################### TABLES ###########################

def display_explained_variance_table(explained_variance, cumulative_variance):
    """
    Displays the explained variance of each principal component as a DataFrame in Streamlit.

    Args:
        explained_variance (array-like): Explained variance ratio of each principal component.
        cumulative_variance (array-like): Cumulative explained variance ratio.

    Returns:
        pd.DataFrame: A DataFrame containing the explained variance and cumulative variance.
    """
    # Convert to DataFrame and ensure numeric values
    variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance": pd.to_numeric(explained_variance, errors='coerce'),
        "Cumulative Variance": pd.to_numeric(cumulative_variance, errors='coerce')
    })

    # Display in Streamlit
    st.write("This table shows how much variance is explained by each principal component.")
    st.dataframe(variance_df.style.format({"Explained Variance": "{:.4f}", "Cumulative Variance": "{:.4f}"}))

    return variance_df  # Returns DataFrame for further processing if needed



def display_pca_loadings(pca, df_features):
    """
    Displays the PCA loadings as a table in Streamlit.

    Args:
        pca (PCA object): Fitted PCA model containing component loadings.
        df_features (DataFrame): Original feature DataFrame before PCA transformation.
    """
    loadings = pd.DataFrame(
        pca.components_,
        columns=df_features.columns,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )

    st.write("This table shows how each original variable contributes to the principal components.")
    st.dataframe(loadings.style.format("{:.4f}"))  # Display with 4 decimal places for readability


###################### MAIN FUNCTION ###############################


# **Main Function: Calls Other Functions**
def perform_pca_analysis_streamlit(df, selected_features):
    st.markdown("## PCA Results")

    pca, pca_result, loadings, explained_variance, cumulative_variance, df_features, dates = compute_pca(df, selected_features)

    contributions_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=dates
    )

    if pca is None:
        return  # Stop execution if PCA computation failed


    st.markdown("### 0 Data Characteristics")

    st.markdown("#### 0.1 Original Values Over Time")
    st.markdown(f"Visualization of the selected saron swap returns")
    plot_original_time_3d_surface_interactive(dates, df_features, flip_axis=True)

    st.markdown('#### 0.2 Scatter Plot of The Original Values In The PC Space')
    plot_pca_3d_scatter(pca_result, dates=dates)

    st.divider()

    st.markdown("### 1 Variance")

    st.markdown("#### 1.1 Explained Variance Table")
    display_explained_variance_table(explained_variance, cumulative_variance)

    st.markdown("#### 1.2 Scree Plot")
    plot_scree(pca)

    st.divider()

    st.markdown("### 2 Variable Contributions")

    st.markdown("#### 2.1 Feature Contributions")
    display_pca_loadings(pca, df_features)

    st.markdown("#### 2.2 Biplot")
    plot_biplot(pca, pca_result, df_features)

    st.markdown("#### 2.3 PCA vs. SARON Maturity")
    plot_pca_loadings_linechart(loadings)

    st.divider()

    st.markdown("### 3 Time Analysis")
    st.markdown("#### 3.1 Feature Contributions over time")
    st.markdown("##### 3.1.1 PC1")

    plot_heatmap(pca, pca_result, df_features, dates, pc_number=1)
    plot_pca_3d_surface_interactive(contributions_df, dates, df_features, loadings, pc_number=1, flip_axis=True)

    st.markdown("##### 3.1.2 PC2")

    plot_heatmap(pca, pca_result, df_features, dates, pc_number=2)
    plot_pca_3d_surface_interactive(contributions_df, dates, df_features, loadings, pc_number=2, flip_axis=True)

    st.markdown("##### 3.1.3 PC3")

    plot_heatmap(pca, pca_result, df_features, dates, pc_number=3)
    plot_pca_3d_surface_interactive(contributions_df, dates, df_features, loadings, pc_number=3, flip_axis=True)





