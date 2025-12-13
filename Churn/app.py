import streamlit as st
import joblib
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Churn Prediction App')

st.divider()

st.write('Please enter the values and hit the prediction button for getting prediction')

st.divider()

#Age', 'Gender', 'Tenure', 'MonthlyCharges', 'ContractType',
 #      'InternetService', 'TechSupport'

age = st.number_input("Enter age",min_value=10, max_value=100, value=30)
gender = st.selectbox("Enter the gender", ['Male','Female'])
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge",min_value=30, max_value=150)
contract = st.selectbox("Enter the Contract Type", ['Month-to-Month','One Year', 'Two Year'])
service = st.selectbox("Enter the Internet Service", ['Fiber Optic','DSL'])
support = st.selectbox("Reached out to Tech Support", ['Yes','No'])

st.divider()

predictbutton = st.button('Predict!')

if predictbutton:
    gender_selected = 1 if gender == 'Female' else 0
    contract_selected = 1 if contract == 'Month-to-Month' else (2 if contract == 'One-Year' else 0)
    service_selected = 1 if service=='Fiber Optic' else 0
    support_selected = 1 if support=='Yes' else 0

    X = [age, gender_selected, tenure, monthlycharge, contract_selected, service_selected, support_selected]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = 'Yes' if prediction==1 else 'No'

    st.write(f"Predicted : {predicted}")

else:
    st.write("Please enter the values and use predict button")



st.divider()
st.write('Upload a CSV or Excel file to predict churn for multiple entries.')
st.divider()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

def preprocess_input(df):
    # Assuming the file has the necessary columns
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Month-to-Month' else (2 if x == 'One Year' else 0))
    df['InternetService'] = df['InternetService'].apply(lambda x: 1 if x == 'Fiber Optic' else 0)
    df['TechSupport'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Selecting feature columns
    features = ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'ContractType', 'InternetService', 'TechSupport']
    return df[features]

if uploaded_file is not None:
    try:
        # Reading the file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded Data:")
        st.dataframe(data.head())

        # Preprocessing input data
        X = preprocess_input(data)

        # Scaling the data
        X_scaled = scaler.transform(X)

        # Making predictions
        predictions = model.predict(X_scaled)

        # Adding predictions to the original data
        data['Churn_Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]

        # Display result
        st.write("### Preview of Results:")
        st.dataframe(data.head())

        # Download the result file
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Please upload a CSV or Excel file to proceed.")


st.divider()

# File uploader for predicted file
uploaded_file = st.file_uploader("Upload your Predicted CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.write("### Data Preview:")
        st.dataframe(data.head())

        # Ensure the 'Churn_Prediction' column exists
        if 'Churn_Prediction' in data.columns:
            # Basic stats
            st.write("### Basic Statistics:")
            st.write(data.describe())
            
            # Churn Distribution
            st.write("### Churn Distribution:")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Churn_Prediction', data=data, palette='Set2')
            plt.title('Churn vs No Churn')
            plt.xlabel('Churn Prediction (Yes/No)')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Gender vs Churn
            st.write("### Gender-wise Churn Distribution:")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Gender', hue='Churn_Prediction', data=data, palette='coolwarm')
            plt.title('Churn by Gender')
            st.pyplot(fig)
            
            # Tenure vs Monthly Charges with Churn
            st.write("### Tenure vs Monthly Charges (Color by Churn):")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x='Tenure', 
                y='MonthlyCharges', 
                hue='Churn_Prediction', 
                palette='Set1', 
                data=data
            )
            plt.title('Tenure vs Monthly Charges with Churn')
            st.pyplot(fig)

            # Contract Type vs Churn
            st.write("### Contract Type vs Churn:")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='ContractType', hue='Churn_Prediction', data=data, palette='viridis')
            plt.title('Contract Type and Churn Relationship')
            st.pyplot(fig)
            
            # Correlation Heatmap
            st.write("### Correlation Heatmap:")
            # Encode Churn for correlation (if not encoded)
            data['Churn_Encoded'] = data['Churn_Prediction'].apply(lambda x: 1 if x == 'Yes' else 0)
            correlation = data.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(fig)

            # Interactive Feature Selection
            st.write("### Interactive Feature Analysis:")
            feature = st.selectbox("Select a Feature to Analyze", data.columns)
            if feature:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x='Churn_Prediction', y=feature, data=data, palette='pastel')
                plt.title(f'{feature} Distribution by Churn')
                st.pyplot(fig)

            # Download updated data
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(data)

            st.download_button(
                label="Download Analysis Data as CSV",
                data=csv,
                file_name='churn_analysis.csv',
                mime='text/csv',
            )

        else:
            st.error("The uploaded file must contain a 'Churn_Prediction' column.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload your predicted CSV or Excel file to start analysis.")
