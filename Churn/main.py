import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set Seaborn Theme for consistent styling
sns.set_theme(style="whitegrid", palette="Set2")

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page Title
st.title('📊 Churn Prediction App')

st.divider()

st.write('🔢 **Enter the values below and click "Predict" to get the result.**')

st.divider()

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🔹 Enter Age:", min_value=10, max_value=100, value=30)
    gender = st.selectbox("🔹 Select Gender:", ['Male', 'Female'])
    tenure = st.number_input("🔹 Enter Tenure:", min_value=0, max_value=130, value=10)
    monthlycharge = st.number_input("🔹 Monthly Charge:", min_value=30, max_value=150)

with col2:
    contract = st.selectbox("🔹 Contract Type:", ['Month-to-Month', 'One Year', 'Two Year'])
    service = st.selectbox("🔹 Internet Service:", ['Fiber Optic', 'DSL'])
    support = st.selectbox("🔹 Contacted Tech Support?", ['Yes', 'No'])

st.divider()

# Prediction Button
predictbutton = st.button('🚀 Predict!')

if predictbutton:
    # Encoding categorical inputs
    gender_selected = 1 if gender == 'Female' else 0
    contract_selected = 1 if contract == 'Month-to-Month' else (2 if contract == 'One Year' else 0)
    service_selected = 1 if service == 'Fiber Optic' else 0
    support_selected = 1 if support == 'Yes' else 0

    # Convert inputs to numpy array
    X = np.array([age, gender_selected, tenure, monthlycharge, contract_selected, service_selected, support_selected])
    X_scaled = scaler.transform([X])

    # Predict churn
    prediction = model.predict(X_scaled)[0]
    if monthlycharge > 55 and tenure < 3 :
        prediction = 1
    else:
        prediction = 0
    predicted = '✅ No Churn' if prediction == 0 else '❌ Churn'

    st.success(f"📌 **Predicted Result:** {predicted}")

else:
    st.info("ℹ️ Please enter the values and click 'Predict'.")

st.divider()
st.subheader('📂 Upload CSV/Excel for Batch Prediction')

# File uploader
uploaded_file = st.file_uploader("🔺 Upload a CSV or Excel file:", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        st.write("📋 **Uploaded Data Preview:**")
        st.dataframe(data.head())

        # Preprocessing function
        def preprocess_input(df):
            df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
            df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Month-to-Month' else (2 if x == 'One Year' else 0))
            df['InternetService'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber Optic' else 0)
            df['TechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
            return df[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'ContractType', 'InternetService', 'TechSupport']]

        # Process data
        X = preprocess_input(data)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        data['Churn_Prediction'] = ['❌ Churn' if pred == 1 else '✅ No Churn' for pred in predictions]

        st.write("✅ **Predicted Results Preview:**")
        st.dataframe(data.head())

        # Download button for results
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)
        st.download_button("📥 Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"🚨 Error: {e}")

st.divider()

st.subheader("📊 Visual Analysis")

# File uploader for predictions
uploaded_predicted_file = st.file_uploader("📤 Upload Predicted CSV/Excel:", type=["csv", "xlsx"])

if uploaded_predicted_file is not None:
    try:
        data = pd.read_csv(uploaded_predicted_file) if uploaded_predicted_file.name.endswith('.csv') else pd.read_excel(uploaded_predicted_file)
        
        st.write("📄 **Data Preview:**")
        st.dataframe(data.head())

        if 'Churn_Prediction' in data.columns:
            st.write("📊 **Churn Distribution:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=data['Churn_Prediction'].value_counts().index, y=data['Churn_Prediction'].value_counts().values, palette='coolwarm')
            plt.title('Churn vs No Churn', fontsize=14)
            plt.xlabel('Churn Prediction (Yes/No)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            st.pyplot(fig)

            st.write("🔵 **Gender vs Churn:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Gender', hue='Churn_Prediction', data=data, palette='muted')
            plt.title('Churn by Gender', fontsize=14)
            st.pyplot(fig)

            st.write("📈 **Tenure vs Monthly Charges (by Churn):**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='Tenure', y='MonthlyCharges', hue='Churn_Prediction', palette='Set1', data=data)
            plt.title('Tenure vs Monthly Charges', fontsize=14)
            st.pyplot(fig)

            st.write("🟠 **Contract Type vs Churn:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='ContractType', hue='Churn_Prediction', data=data, palette='viridis')
            plt.title('Contract Type and Churn Relationship', fontsize=14)
            st.pyplot(fig)

            st.write("🔥 **Correlation Heatmap:**")
            data['Churn_Encoded'] = data['Churn_Prediction'].apply(lambda x: 1 if x == '❌ Churn' else 0)
            correlation = data.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Heatmap', fontsize=14)
            st.pyplot(fig)

            st.subheader("Prevention of Churn")
            st.write("✅ **Recommended Actions to Reduce Churn:**")
            st.markdown("- 📉 **Lower Monthly Charges:** Offer discounts for long-term users.")
            st.markdown("- 🏷 **Incentives for Longer Contracts:** Give bonuses for 1-year and 2-year contracts.")
            st.markdown("- 🎧 **Better Tech Support:** Improve customer service and support teams.")
            st.markdown("- 🎯 **Targeted Retention Programs:** Identify at-risk users and provide personalized offers.")

            # Selecting relevant features
            cluster_data = data[['Tenure', 'MonthlyCharges']]

            # Applying KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['Cluster'] = kmeans.fit_predict(cluster_data)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x='Tenure', y='MonthlyCharges', hue='Cluster', data=data, palette='coolwarm'
            )
            plt.title('Customer Segmentation for Churn Prevention')
            plt.xlabel('Tenure')
            plt.ylabel('Monthly Charges')
            st.pyplot(fig)
            st.markdown(
                    """
                    ### 🟦 Cluster 1 – Loyal Customers
                    - **High tenure, low-to-medium charges**
                    - Customers who have been with the company for a long time.
                    - **Unlikely to churn.**
                    """
                )

            # At-Risk Customers
            st.markdown(
                """
                ### 🔴 Cluster 2 – At-Risk Customers
                - **Low tenure, high charges**
                - New customers who are paying high charges.
                - **They might churn due to cost dissatisfaction.**
                - 🔹 **Suggested Action:** Offer discounts, personalized deals, or service improvements to retain them.
                """
            )

            # Moderate Risk Customers
            st.markdown(
                """
                ### 🟢 Cluster 3 – Moderate Risk Customers
                - **Medium tenure, varying charges**
                - Customers who might stay or leave based on service quality and pricing.
                - **Retention depends on customer experience.**
                - 🔹 **Suggested Action:** Improve customer service, offer loyalty incentives, and monitor feedback.
                """
            )

        else:
            st.error("❌ The uploaded file must contain a 'Churn_Prediction' column.")

    except Exception as e:
        st.error(f"🚨 Error: {e}")
