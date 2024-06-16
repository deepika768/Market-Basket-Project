import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport
from pydantic import BaseModel, BaseSettings, Field

# Load Classification and Clustering Pipeline models
model_classification = joblib.load('C:/Users/sahud/Downloads/Brazilian Ecommerce Classification.bkl')
model_clustering = joblib.load('C:/Users/sahud/Downloads/Brazilian Ecommerce Clustering.bkl')

# Create Sidebar to navigate between EDA, Classification, and Clustering
sidebar = st.sidebar
mode = sidebar.radio('Mode', ['EDA', 'Classification', 'Clustering'])
st.markdown("<h1 style='text-align: center; color: #ff0000;'>Customer Satisfaction Prediction</h1>", unsafe_allow_html=True)

if mode == "EDA":
    def main():
        # Header of Customer Satisfaction Prediction
        html_temp = """
                <div style="background-color:#F5F5F5">
                <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>
                </div>
            """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Check if the CSV file path exists
        try:
            EDA_sample = pd.read_csv(r'C:/Market Basket ML Project/data/EDA.csv', index_col=0)
            st.header('**Input DataFrame**')
            st.write(EDA_sample)
            st.write(f"DataFrame shape: {EDA_sample.shape}")

            # Generate Pandas Profiling Report
            pr = ProfileReport(EDA_sample, explorative=True)
            pr.to_file("report.html")

            st.header('**Pandas Profiling Report**')
            with open("report.html", "r", encoding='utf-8') as f:
                components.html(f.read(), height=800, scrolling=True)

        except FileNotFoundError:
            st.error("File not found. Please check the file path.")
        except pd.errors.EmptyDataError:
            st.error("The file is empty. Please check the file content.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if __name__ == '__main__':
        main()

if mode == "Classification":
    # Define function to predict classification based on assigned features
    def predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value,
                             estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate):
        prediction_classification = model_classification.predict(pd.DataFrame({
            'freight_value': [freight_value],
            'product_description_lenght': [product_description_lenght],
            'product_photos_qty': [product_photos_qty],
            'payment_type': [payment_type],
            'payment_installments': [payment_installments],
            'payment_value': [payment_value],
            'estimated_days': [estimated_days],
            'arrival_days': [arrival_days],
            'arrival_status': [arrival_status],
            'seller_to_carrier_status': [seller_to_carrier_status],
            'estimated_delivery_rate': [estimated_delivery_rate],
            'arrival_delivery_rate': [arrival_delivery_rate],
            'shipping_delivery_rate': [shipping_delivery_rate]
        }))
        return prediction_classification

    def main():
        # Header of Customer Satisfaction Prediction
        html_temp = """
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>
                    </div>
                """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Assign all features with desired data input method
        sidebar.title('Numerical Features')
        product_description_lenght = sidebar.slider('product_description_lenght', 4, 3990, 100)
        product_photos_qty = sidebar.slider('product_photos_qty', 1, 20, 1)
        payment_installments = sidebar.slider('payment_installments', 1, 24, 1)
        estimated_days = sidebar.slider('estimated_days', 3, 60, 1)
        arrival_days = sidebar.slider('arrival_days', 0, 60, 1)
        payment_type = st.selectbox('payment_type', ['credit_card', 'boleto', 'voucher', 'debit_card'])
        arrival_status = st.selectbox('arrival_status', ['OnTime/Early', 'Late'])
        seller_to_carrier_status = st.selectbox('seller_to_carrier_status', ['OnTime/Early', 'Late'])
        estimated_delivery_rate = st.selectbox('estimated_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        arrival_delivery_rate = st.selectbox('arrival_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        shipping_delivery_rate = st.selectbox('shipping_delivery_rate Date', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        payment_value = st.text_input('payment_value', '')
        freight_value = st.text_input('freight_value', '')
        result = ''

        # Predict Customer Satisfaction
        if st.button('Predict_Satisfaction'):
            result = predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value,
                                          estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate)

        if result == 0:
            result = 'Not Satisfied'
            st.success(f'The Customer is {result}')
        else:
            result = 'Satisfied'
            st.success(f'The Customer is {result}')

    if __name__ == '__main__':
        main()

if mode == "Clustering":
    def predict_clustering(freight_value, price, payment_value, payment_installments, payment_sequential):
        prediction_clustering = model_clustering.predict(pd.DataFrame({
            'freight_value': [freight_value],
            'price': [price],
            'payment_installments': [payment_installments],
            'payment_value': [payment_value],
            'payment_sequential': [payment_sequential]
        }))
        return prediction_clustering

    def main():
        # Header of Customer Segmentation
        html_temp = """
                <div style="background-color:#F5F5F5">
                <h1 style="color:#31333F;text-align:center;"> Customer Segmentation </h1>
                </div>
            """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Specify CSV file path (change this path as needed)
        csv_path = r'C:/Market Basket ML Project/data/Clustering Sample.csv'

        # Read dataset
        try:
            sample = pd.read_csv(csv_path)
            st.header('Uploaded Data')
            st.write(sample)
            st.write(f"DataFrame shape: {sample.shape}")

            # Define sidebar for clustering algorithm
            selected_algorithm = st.sidebar.selectbox('Select Clustering Algorithm', ['K-Means', 'Agglomerative'])

            # Define sidebar for number of clusters
            selected_clusters = st.sidebar.slider('Select number of clusters', 2, 10, 2)

            # Define sidebar for PCA
            use_pca = st.sidebar.radio('Use PCA', ('Yes', 'No'))

            # Perform clustering
            if st.button('Predict Clusters'):
                # Preprocessing
                scaler = StandardScaler()
                sample_scaled = scaler.fit_transform(sample)

                # Clustering
                if selected_algorithm == 'K-Means':
                    model = KMeans(n_clusters=selected_clusters)
                else:
                    model = AgglomerativeClustering(n_clusters=selected_clusters)

                clusters = model.fit_predict(sample_scaled)

                # Add cluster labels to dataframe
                sample['Cluster'] = clusters

                # Plot results (optional)
                if use_pca == 'Yes':
                    pca = PCA(n_components=2)
                    sample_pca = pca.fit_transform(sample_scaled)
                    st.header('PCA Scatter Plot')
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=sample_pca[:, 0], y=sample_pca[:, 1], hue=clusters, palette='viridis')
                    plt.title('PCA Scatter Plot')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    st.pyplot(plt)

                # Display clustered data
                st.success('Customer clusters predicted successfully!')
                st.header('Clustered Data')
                st.write(sample)
        
        except FileNotFoundError:
            st.error(f"File '{csv_path}' not found. Please check the file path and try again.")
        except pd.errors.EmptyDataError:
            st.error("The file is empty. Please check the file content.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if __name__ == '__main__':
        main()
