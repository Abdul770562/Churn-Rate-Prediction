import streamlit as st

def main():
    st.title("📈 Churn Prediction and Analysis App")
    st.markdown("""
    ## Welcome to the Churn Prediction and Analysis Platform!

    This interactive web application is designed to help businesses analyze customer data and predict churn rates.

    ### Key Features:
    - 🔍 **Upload Datasets:** Easily upload your CSV or Excel files for analysis.
    - 📊 **Data Analysis:** Explore interactive visualizations including distribution charts, scatter plots, and heatmaps.
    - 🏷️ **Churn Prediction:** Leverage predictive models to identify customers likely to churn.
    - 🏷️ **Customer Segmentation:** Group customers into clusters for targeted retention strategies.

    ### Why Use This App?
    - ✅ **Data-Driven Insights:** Make informed decisions based on detailed analytics.
    - ⚡ **Interactive Visuals:** Gain deeper understanding through charts and heatmaps.
    - 🎯 **Retention Strategies:** Develop actionable plans to retain valuable customers.

    **Use the sidebar to navigate through different sections of the app.**

    """)

if __name__ == "__main__":
    main()