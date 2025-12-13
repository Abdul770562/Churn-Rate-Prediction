import streamlit as st
import home
import predictions
import upload_dataset
import data_analysis

PAGES = {
    "🏠 Home": home,
    "🔬 Predictions": predictions,
    "🗂️ Upload Dataset": upload_dataset,
    "📊 Data Analysis": data_analysis
}

st.sidebar.markdown("""
<style>
    .sidebar .css-1d391kg {  /* Adjust this class if needed */
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
    }
    .sidebar .css-1d391kg h2 {
        color: #00ffd1;
        font-family: 'Courier New', Courier, monospace;
    }
    .sidebar .css-1d391kg .st-radio {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .sidebar .css-1d391kg .st-radio div {
        font-size: 18px;
        color: white;
    }
    .sidebar .css-1d391kg .st-radio div:hover {
        color: #00ffd1;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title('🚀 Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.main()
