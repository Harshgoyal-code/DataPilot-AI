"""
eda.py - Exploratory Data Analysis Helper Functions

This file contains utility functions to handle data loading, summary statistics,
and data visualizations. Separating this logic from the main app.py file
makes our codebase cleaner, more modular, and easier to maintain.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def is_dark_mode():
    """
    Detects if the current Streamlit theme is dark mode.
    """
    try:
        base = st.config.get_option("theme.base")
        if base == "dark":
            return True
        elif base == "light":
            return False
            
        bg = st.config.get_option("theme.backgroundColor")
        if bg and bg.startswith("#"):
            rgb = bg.lstrip('#')
            if len(rgb) == 6:
                r, g, b = int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16)
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                return brightness < 128
    except Exception:
        pass
    return False


def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Parameters:
        file_path (str or UploadedFile): The path to the CSV file or a Streamlit file upload object.
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame, or None if loading fails.
    """
    try:
        # Load the CSV file using pandas
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        # Print the error for debugging (visible in console)
        print(f"Error loading CSV file: {e}")
        return None

def get_basic_summary(df):
    """
    Calculates basic metadata and descriptive statistics for the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        dict: A dictionary containing key statistics of the dataset.
    """
    # Initialize a dictionary to store summary statistics
    summary = {}
    
    # Store dimensions (number of rows, number of columns)
    summary["shape"] = df.shape
    
    # Extract total number of rows and columns
    summary["num_rows"] = df.shape[0]
    summary["num_cols"] = df.shape[1]
    
    # Count missing values per column and represent it as a DataFrame
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column Name", "Missing Values Count"]
    summary["missing_values"] = missing_df
    
    # Extract data types for each column
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Column Name", "Data Type"]
    summary["data_types"] = dtypes_df
    
    # Categorize columns into numeric and categorical
    # This is helpful for user selection options in the UI
    summary["numeric_cols"] = list(df.select_dtypes(include=["number"]).columns)
    summary["categorical_cols"] = list(df.select_dtypes(include=["object", "category"]).columns)
    
    return summary

def generate_distribution_plot(df, column):
    """
    Generates a matplotlib figure showing the distribution of a numerical column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the numerical column to plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Create a new matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Make background transparent for dark/light mode compatibility
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Determine colors based on active Streamlit theme mode
    dark_mode = is_dark_mode()
    text_color = st.config.get_option("theme.textColor") or ("white" if dark_mode else "black")
    label_color = text_color
    grid_color = "white" if dark_mode else "gray"
    grid_alpha = 0.2 if dark_mode else 0.5
    
    # Draw a histogram with Kernel Density Estimate (KDE) line
    sns.histplot(data=df, x=column, kde=True, color="#4F46E5", ax=ax)
    
    # Set titles and labels with dynamic theme colors and increased font sizes
    ax.set_title(f"Distribution of {column}", fontsize=18, pad=15, fontweight='bold', color=text_color)
    ax.set_xlabel(column, fontsize=15, labelpad=10, color=label_color)
    ax.set_ylabel("Frequency / Count", fontsize=15, labelpad=10, color=label_color)
    
    # Increase tick label sizes and apply theme color for dark/light mode compatibility
    ax.tick_params(axis='both', labelsize=13, colors=text_color)
    
    # Add subtle high-contrast grid lines that adapt to dark/light backgrounds
    ax.grid(True, linestyle=':', alpha=grid_alpha, color=grid_color)
    
    # Style details: remove top and right borders and apply spine colors
    sns.despine(fig=fig, ax=ax)
    for spine in ax.spines.values():
        spine.set_color(label_color)
        
    # Tight layout ensures everything fits perfectly without overlapping labels
    plt.tight_layout()
    return fig

def generate_scatter_plot(df, x_col, y_col):
    """
    Generates a matplotlib figure displaying a scatter plot between two variables.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column name to represent on the X-axis.
        y_col (str): The column name to represent on the Y-axis.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Create a new matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Make background transparent for dark/light mode compatibility
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Determine colors based on active Streamlit theme mode
    dark_mode = is_dark_mode()
    text_color = st.config.get_option("theme.textColor") or ("white" if dark_mode else "black")
    label_color = text_color
    grid_color = "white" if dark_mode else "gray"
    grid_alpha = 0.2 if dark_mode else 0.5
    
    # Draw the scatter plot with styled color and markers (increased point size for readability)
    sns.scatterplot(data=df, x=x_col, y=y_col, color="#06B6D4", alpha=0.7, edgecolors="w", s=100, ax=ax)
    
    # Set titles and label styles with dynamic theme colors and increased font sizes
    ax.set_title(f"{y_col} vs {x_col}", fontsize=18, pad=15, fontweight='bold', color=text_color)
    ax.set_xlabel(x_col, fontsize=15, labelpad=10, color=label_color)
    ax.set_ylabel(y_col, fontsize=15, labelpad=10, color=label_color)
    
    # Increase tick label sizes and apply theme color for dark/light mode compatibility
    ax.tick_params(axis='both', labelsize=13, colors=text_color)
    
    # Add subtle high-contrast grid lines that adapt to dark/light backgrounds
    ax.grid(True, linestyle=':', alpha=grid_alpha, color=grid_color)
    
    # Modern layout design
    sns.despine(fig=fig, ax=ax)
    for spine in ax.spines.values():
        spine.set_color(label_color)
        
    plt.tight_layout()
    return fig

def generate_correlation_heatmap(df):
    """
    Generates a matplotlib figure containing a correlation heatmap of numeric columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        matplotlib.figure.Figure or None: The heatmap figure, or None if not enough numeric columns exist.
    """
    # Select only numeric columns since correlation works only on numerical data
    numeric_df = df.select_dtypes(include=["number"])
    
    # We need at least two numeric columns to find correlation
    if numeric_df.shape[1] < 2:
        return None
        
    # Calculate the Pearson correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create a new matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Make background transparent for dark/light mode compatibility
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Determine colors based on active Streamlit theme mode
    dark_mode = is_dark_mode()
    text_color = st.config.get_option("theme.textColor") or ("white" if dark_mode else "black")
    
    # Generate the heatmap with a warm-cool color palette (coolwarm)
    # Increased annotation size and made them bold for better readability
    sns.heatmap(
        corr_matrix, 
        annot=True,          # Show correlation values inside cells
        cmap="coolwarm",      # Harmonious red-to-blue color scale
        fmt=".2f",           # Limit decimal points to 2
        linewidths=0.5,      # Add subtle lines separating cells
        ax=ax,
        square=True,         # Make cells square
        cbar_kws={"shrink": 0.8}, # Slightly shrink the colorbar
        annot_kws={"size": 13, "weight": "bold"} # Larger bold font inside cells
    )
    
    # Heatmap title with dynamic theme colors and increased font size
    ax.set_title("Numeric Features Correlation Matrix", fontsize=19, pad=20, fontweight='bold', color=text_color)
    
    # Increase tick label sizes and apply theme color for dark/light mode compatibility
    ax.tick_params(axis='both', labelsize=14, colors=text_color)
    
    # Format colorbar ticks to match the theme color
    try:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
    except Exception:
        pass
        
    plt.tight_layout()
    return fig

