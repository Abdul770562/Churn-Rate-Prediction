import streamlit as st
import pandas as pd
import pickle 

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title("Upload Dataset for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            def preprocess_input(df):
                df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
                df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Month-to-Month' else (2 if x == 'One Year' else 0))
                df['InternetService'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber Optic' else 0)
                df['TechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
                return df[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'ContractType', 'InternetService', 'TechSupport']]

            # ... (Rest of the upload dataset logic from your original script)
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
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()