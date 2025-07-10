import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def aggregate_period_counts(df):
    """
    Aggregate record counts by 'period' column.
    Returns a DataFrame with 'period' and 'count'.
    """
    if 'period' not in df.columns:
        return None
    counts = df.groupby('period').size().reset_index(name='count')
    return counts

@st.cache_data
def get_numeric_columns(df):
    """
    Return a list of numeric column names in the DataFrame.
    Cached to avoid repeated dtype checks.
    """
    return df.select_dtypes(include=['number']).columns.tolist()

def render_time_series(df):
    """
    Render a line chart showing the count of records per period.
    """
    counts = aggregate_period_counts(df)
    if counts is None or counts.empty:
        st.warning("No 'period' column found or no data to plot time series.")
        return

    fig, ax = plt.subplots()
    ax.plot(counts['period'], counts['count'], marker='o')
    ax.set_xlabel('Period')
    ax.set_ylabel('Count')
    ax.set_title('Records per Period')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def render_histogram(df):
    """
    Render a histogram of the first numeric column.
    """
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.warning("No numeric columns available to plot histogram.")
        return

    col = numeric_cols[0]
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=20)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {col}')
    st.pyplot(fig)

def render_scatter(df):
    """
    Render a scatter plot of the first two numeric columns.
    """
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to plot scatter.")
        return

    x_col, y_col = numeric_cols[0], numeric_cols[1]
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    st.pyplot(fig)
