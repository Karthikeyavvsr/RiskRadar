# riskradar/utils/risk_clusterer.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Import centralized data fetchers
from .data_fetcher import get_historical_data, get_stock_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------
def get_price_matrix_and_metadata(tickers: list, period: str = "3mo"):
    """
    Fetches historical closing prices and metadata (sector, type) for clustering.
    Uses the centralized, cached data_fetcher.
    """
    price_data = {}
    metadata = []
    logging.info(f"Fetching price matrix and metadata for clustering: {tickers}")

    for ticker in tickers:
        # Fetch historical data
        hist = get_historical_data(ticker, period=period)

        # Fetch stock info (sector, type)
        info = get_stock_info(ticker)
        sector = info.get("sector", "Unknown")
        # Use quoteType for broader classification (EQUITY, ETF, MUTUALFUND, etc.)
        asset_type = info.get("quoteType", "Unknown")

        if not hist.empty and 'Close' in hist.columns:
            price_data[ticker] = hist['Close']
            metadata.append({"ticker": ticker, "sector": sector, "type": asset_type})
            logging.debug(f"Successfully fetched data for {ticker}")
        else:
            logging.warning(f"No historical data found for {ticker} in get_price_matrix. Skipping.")
            # Still add metadata even if price data failed, might be useful later
            metadata.append({"ticker": ticker, "sector": sector, "type": asset_type})

    if not price_data:
         logging.error("Failed to fetch price data for ALL tickers for clustering.")
         return pd.DataFrame(), pd.DataFrame(metadata)

    # Combine price data, ensuring consistent index alignment
    price_df = pd.concat(price_data, axis=1).sort_index()
    # Handle potential missing values introduced by concat if indices aren't perfectly aligned
    price_df = price_df.ffill().bfill()

    metadata_df = pd.DataFrame(metadata)
    return price_df, metadata_df


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily percentage returns."""
    if price_df.empty or len(price_df) < 2:
        logging.warning("Price DataFrame empty or too short for return calculation.")
        return pd.DataFrame()
    returns = price_df.pct_change().dropna(axis=0, how='all') # Drop rows where all values are NaN
    return returns


def cluster_assets(returns_df: pd.DataFrame, metadata_df: pd.DataFrame, n_clusters: int = 3):
    """
    Clusters assets based on the correlation of their daily returns using PCA and K-Means.
    """
    if returns_df.empty or returns_df.shape[1] < n_clusters:
        logging.warning(f"Returns DataFrame is empty or has fewer columns ({returns_df.shape[1]}) than n_clusters ({n_clusters}). Cannot cluster.")
        # Return dataframe with tickers from metadata if available
        if not metadata_df.empty:
             return metadata_df.assign(cluster=np.nan, x=np.nan, y=np.nan)
        return pd.DataFrame()

    # Calculate correlation matrix, handle NaNs robustly
    # Fill NaNs that might occur if a stock had zero variance (returns are all 0)
    correlation_matrix = returns_df.corr().fillna(0)

    # Ensure matrix is square and suitable for PCA
    if correlation_matrix.empty or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
         logging.error("Correlation matrix is empty or non-square. Cannot perform PCA.")
         if not metadata_df.empty:
             return metadata_df.assign(cluster=np.nan, x=np.nan, y=np.nan)
         return pd.DataFrame()


    # --- PCA for dimensionality reduction ---
    # Standardize the correlation matrix before PCA? Or apply PCA directly?
    # Applying directly to correlation matrix is common for this type of analysis.
    n_components = min(2, correlation_matrix.shape[1]) # Max 2 components for 2D plot, ensure less than num features
    if n_components < 1:
         logging.error("Cannot perform PCA with less than 1 component.")
         if not metadata_df.empty:
             return metadata_df.assign(cluster=np.nan, x=np.nan, y=np.nan)
         return pd.DataFrame()

    try:
        pca = PCA(n_components=n_components)
        # Fit PCA on the correlation matrix (or transpose if features should be rows)
        # Usually fit on data where rows=samples, cols=features. Here assets are features.
        # We want to reduce the dimensionality of the assets based on their correlation structure.
        # Let's use the correlation matrix directly as input, representing asset relationships.
        reduced_features = pca.fit_transform(correlation_matrix)
    except Exception as e:
        logging.error(f"PCA failed: {e}")
        if not metadata_df.empty:
             return metadata_df.assign(cluster=np.nan, x=np.nan, y=np.nan)
        return pd.DataFrame()


    # --- K-Means Clustering ---
    try:
        # Use k-means++ for better initialization, increase n_init
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_features)
    except Exception as e:
        logging.error(f"K-Means clustering failed: {e}")
        if not metadata_df.empty:
             # Assign NaN cluster if K-Means fails but PCA worked
             pca_df = pd.DataFrame(reduced_features, index=correlation_matrix.index, columns=[f'pca_{i+1}' for i in range(n_components)])
             merged = metadata_df.merge(pca_df.reset_index().rename(columns={'index':'ticker'}), on='ticker', how='left')
             merged['cluster'] = np.nan
             merged = merged.rename(columns={'pca_1':'x', 'pca_2':'y'} if n_components==2 else {})
             return merged
        return pd.DataFrame()


    # --- Combine Results ---
    # Create DataFrame from PCA results
    pca_cols = [f'pca_{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(reduced_features, index=correlation_matrix.index, columns=pca_cols)
    pca_df['cluster'] = cluster_labels
    pca_df = pca_df.reset_index().rename(columns={'index':'ticker'}) # Add ticker column from index

    # Merge with metadata
    clustered_df = metadata_df.merge(pca_df, on='ticker', how='left')

    # Standardize column names for plotting
    if 'pca_1' in clustered_df.columns:
        clustered_df = clustered_df.rename(columns={'pca_1': 'x'})
    else:
         clustered_df['x'] = 0 # Add placeholder if only 1 component

    if 'pca_2' in clustered_df.columns:
        clustered_df = clustered_df.rename(columns={'pca_2': 'y'})
    else:
         clustered_df['y'] = 0 # Add placeholder if only 1 component


    logging.info(f"Clustering complete. Found {len(clustered_df['cluster'].unique())} clusters.")
    return clustered_df


def plot_clusters(cluster_df: pd.DataFrame):
    """Plots the asset clusters using Seaborn."""
    if cluster_df.empty or 'x' not in cluster_df.columns or 'y' not in cluster_df.columns or 'cluster' not in cluster_df.columns:
        logging.warning("Empty or malformed cluster_df passed to plot_clusters.")
        return None
    if cluster_df['cluster'].isnull().all():
         logging.warning("All cluster labels are NaN. Cannot plot clusters.")
         return None

    # Drop rows with NaN clusters before plotting
    plot_df = cluster_df.dropna(subset=['cluster', 'x', 'y'])
    if plot_df.empty:
         logging.warning("No valid data points left after dropping NaNs for plotting.")
         return None


    plt.style.use('seaborn-v0_8-whitegrid') # Use a seaborn style
    plt.figure(figsize=(10, 7)) # Adjusted size

    # Use a categorical palette suitable for clusters
    num_clusters = int(plot_df['cluster'].max() + 1)
    palette = sns.color_palette("tab10", n_colors=num_clusters)

    scatter = sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='cluster',
        style='type', # Use asset type for different marker styles
        size='sector', # Vary size slightly by sector (optional, can be noisy)
        sizes=(50, 200), # Range of sizes
        palette=palette,
        # s=150, # Base size if not varying by sector
        edgecolor="black",
        alpha=0.8 # Add slight transparency
    )

    # Add labels to points
    for i in range(plot_df.shape[0]):
        plt.text(
            plot_df['x'].iloc[i] + 0.01 * (plot_df['x'].max() - plot_df['x'].min()), # Adjust offset based on range
            plot_df['y'].iloc[i],
            plot_df['ticker'].iloc[i],
            fontsize=8, # Slightly smaller font
            ha='left',
            va='center',
            alpha=0.9
        )

    plt.title('Asset Risk Clusters (Based on Return Correlation)', fontsize=14)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)

    # Improve legend
    handles, labels = scatter.get_legend_handles_labels()
    # Optional: Customize legend title or labels if needed
    plt.legend(title='Cluster / Type / Sector', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    return plt