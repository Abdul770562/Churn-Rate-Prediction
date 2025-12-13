import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns 
import matplotlib.pyplot as plt
import altair as alt

def main():
    st.title("📊 Data Analysis")

    # File Upload
    uploaded_predicted_file = st.file_uploader("📄 Upload Predicted CSV/Excel:", type=["csv", "xlsx"])

    if uploaded_predicted_file is not None:
        try:
            # Read File
            data = pd.read_csv(uploaded_predicted_file) if uploaded_predicted_file.name.endswith('.csv') else pd.read_excel(uploaded_predicted_file)

            st.write("📄 **Data Preview:**")
            st.dataframe(data.head())

            if 'Churn_Prediction' in data.columns:
                
                # Churn Distribution
                st.write("📊 **Churn Distribution:**")
                st.write("This chart shows the overall distribution of customers who churned vs those who stayed.")
                churn_counts = data['Churn_Prediction'].value_counts()
                st.bar_chart(churn_counts)

                # Gender vs Churn
                st.write("🔵 **Gender vs Churn:**")
                st.write("This plot displays how gender correlates with churn behavior.")
                data['Gender'] = data['Gender'].replace({0: 'Male', 1: 'Female'})
                gender_churn = data.groupby(['Gender', 'Churn_Prediction']).size().unstack()
                st.bar_chart(gender_churn)

                # Tenure vs Monthly Charges (Scatter Plot)
                st.write("📈 **Tenure vs Monthly Charges (by Churn):**")
                st.write("This scatter plot illustrates how monthly charges vary with tenure, categorized by churn status.")
                scatter_plot = alt.Chart(data).mark_circle(size=60).encode(
                    x='Tenure',
                    y='MonthlyCharges',
                    color='Churn_Prediction',
                    tooltip=['Tenure', 'MonthlyCharges', 'Churn_Prediction']
                ).properties(
                    width=600,
                    height=400,
                    title='Tenure vs Monthly Charges (by Churn)'
                ).interactive()
                st.altair_chart(scatter_plot, use_container_width=True)

                # Contract Type vs Churn with Labels
                st.write("🟠 **Contract Type vs Churn:**")
                st.write("This bar chart shows churn distribution across different contract types.")
                contract_mapping = {0: "Month-to-month", 1: "One-year", 2: "Two-year"}
                data['Contract_Label'] = data['ContractType'].map(contract_mapping)
                contract_churn = data.groupby(['Contract_Label', 'Churn_Prediction']).size().unstack()

                # Altair Bar Chart
                contract_chart = alt.Chart(contract_churn.reset_index().melt(id_vars='Contract_Label')).mark_bar().encode(
                    x=alt.X('Contract_Label', title='Contract Type'),
                    y=alt.Y('value', title='Count'),
                    color='Churn_Prediction',
                    tooltip=['Contract_Label', 'Churn_Prediction', 'value']
                ).properties(
                    width=600,
                    height=400,
                    title='Contract Type vs Churn'
                )
                st.altair_chart(contract_chart, use_container_width=True)

                # Correlation Heatmap
                st.write("🔥 **Correlation Heatmap:**")
                st.write("This heatmap displays the correlation between different numerical variables.")
                data['Churn_Encoded'] = data['Churn_Prediction'].apply(lambda x: 1 if x == '❌ Churn' else 0)

                # Dark Theme Correlation Heatmap
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='rocket', fmt='.2f', linewidths=0.5, ax=ax)
                plt.title('Correlation Heatmap', color='white')
                st.pyplot(fig)

                # KMeans Clustering for Customer Segmentation
                st.write("📊 **Customer Segmentation for Churn Prevention:**")
                st.write("This scatter plot segments customers into three clusters for targeted retention strategies.")
                cluster_data = data[['Tenure', 'MonthlyCharges']]
                kmeans = KMeans(n_clusters=3, random_state=42)
                data['Cluster'] = kmeans.fit_predict(cluster_data)

                # Scatter Plot for Clusters
                cluster_scatter = alt.Chart(data).mark_circle(size=80).encode(
                    x='Tenure',
                    y='MonthlyCharges',
                    color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Tenure', 'MonthlyCharges', 'Cluster']
                ).properties(
                    width=700,
                    height=400,
                    title='Customer Segmentation (Clusters)'
                ).interactive()
                st.altair_chart(cluster_scatter, use_container_width=True)

                # Cluster Information
                st.markdown(
                    """
                    ### 🟦 Cluster 1 – Loyal Customers
                    - **High tenure, low-to-medium charges**
                    - **Unlikely to churn.**
                    """
                )

                st.markdown(
                    """
                    ### 🔴 Cluster 2 – At-Risk Customers
                    - **Low tenure, high charges**
                    - **Might churn due to cost dissatisfaction.**
                    - 🔹 **Suggested Action:** Offer discounts or service improvements.
                    """
                )

                st.markdown(
                    """
                    ### 🟢 Cluster 3 – Moderate Risk Customers
                    - **Medium tenure, varying charges**
                    - **Retention depends on customer experience.**
                    - 🔹 **Suggested Action:** Offer loyalty incentives and monitor feedback.
                    """
                )

            else:
                st.error("❌ The uploaded file must contain a 'Churn_Prediction' column.")

        except Exception as e:
            st.error(f"🛑 Error: {e}")

if __name__ == "__main__":
    main()