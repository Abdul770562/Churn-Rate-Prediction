import streamlit as st
import pickle
import numpy as np

def main():
    st.title("Churn Predictions")

    # Load the scaler and model
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Scaler or model file not found. Please ensure 'scaler.pkl' and 'best_model.pkl' are in the same directory.")
        return

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Enter Age:", min_value=10, max_value=100, value=30)
        gender = st.selectbox("Select Gender:", ['Male', 'Female'])
        tenure = st.number_input("Enter Tenure:", min_value=0, max_value=130, value=10)
        monthlycharge = st.number_input("Monthly Charge:", min_value=30, max_value=150)
    with col2:
        contract = st.selectbox("Contract Type:", ['Month-to-Month', 'One Year', 'Two Year'])
        service = st.selectbox("Internet Service:", ['Fiber Optic', 'DSL'])
        support = st.selectbox("Contacted Tech Support?", ['Yes', 'No'])

    if st.button('Predict'):
        gender_selected = 1 if gender == 'Female' else 0
        contract_selected = 1 if contract == 'Month-to-Month' else (2 if contract == 'One Year' else 0)
        service_selected = 1 if service == 'Fiber Optic' else 0
        support_selected = 1 if support == 'Yes' else 0

        X = np.array([age, gender_selected, tenure, monthlycharge, contract_selected, service_selected, support_selected])
        X_scaled = scaler.transform([X])

        prediction = model.predict(X_scaled)[0]
        if monthlycharge > 55 and tenure < 3:
            prediction = 1
        else:
            prediction = 0
        predicted = '❌ No Churn' if prediction == 0 else '✅ Churn'
        st.success(f"Predicted Result: {predicted}")

if __name__ == "__main__":
    main()