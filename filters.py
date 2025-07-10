import streamlit as st

def render_filters(df):
    
    st.sidebar.header("Filters")

    # Period filter (based on filename periods)
    if "period" in df.columns:
        periods = sorted(df["period"].dropna().unique())
        period_choice = st.sidebar.selectbox(
            "Select Period", ["All"] + periods,
            key="period_filter"
        )
        period = None if period_choice == "All" else period_choice
    else:
        period = None

    # Category filter (if exists)
    if "category" in df.columns:
        categories = sorted(df["category"].dropna().unique())
        category_choice = st.sidebar.selectbox(
            "Select Category", ["All"] + categories,
            key="category_filter"
        )
        category = None if category_choice == "All" else category_choice
    else:
        category = None

    return {"period": period, "category": category}
