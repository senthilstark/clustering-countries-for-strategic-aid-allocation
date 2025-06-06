from sklearn.svm import SVC
from sklearn.utils.validation import has_fit_parameter
import streamlit as st
import pickle
import pandas as pd
import warnings
import sklearn

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

st.header('Country Clustering Prediction')

with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('hierarchical.pkl', 'rb') as f:
    hierarchical = pickle.load(f)

existing_data = pd.read_csv('Country-data.csv')  # Replace with your actual dataset path

existing_data_numeric = existing_data.select_dtypes(include=[float, int])

country_names = existing_data['country'].tolist()

selected_country = st.selectbox('Select a country to auto-fill fields:', country_names)

# Autofill the input fields based on the selected country
if selected_country:
    country_data = existing_data[existing_data['country'] == selected_country].iloc[0]
    child_mort = st.number_input('Child Mortality', value=country_data['child_mort'])
    exports = st.number_input('Exports', value=country_data['exports'])
    health = st.number_input('Health Spending', value=country_data['health'])
    imports = st.number_input('Imports', value=country_data['imports'])
    income = st.number_input('Income', value=country_data['income'])
    inflation = st.number_input('Inflation', value=country_data['inflation'])
    life_expec = st.number_input('Life Expectancy', value=country_data['life_expec'])
    total_fer = st.number_input('Total Fertility', value=country_data['total_fer'])
    gdpp = st.number_input('GDP per capita', value=country_data['gdpp'])
else:
    child_mort = st.number_input('Child Mortality')
    exports = st.number_input('Exports')
    health = st.number_input('Health Spending')
    imports = st.number_input('Imports')
    income = st.number_input('Income')
    inflation = st.number_input('Inflation')
    life_expec = st.number_input('Life Expectancy')
    total_fer = st.number_input('Total Fertility')
    gdpp = st.number_input('GDP per capita')

if st.button('Predict and Visualize Clusters'):
    input_data = {
        'child_mort': [child_mort],
        'exports': [exports],
        'health': [health],
        'imports': [imports],
        'income': [income],
        'inflation': [inflation],
        'life_expec': [life_expec],
        'total_fer': [total_fer],
        'gdpp': [gdpp]
    }

    input_df = pd.DataFrame(input_data)
    input_df1 = pd.DataFrame(input_data)
    # Combine input data with the existing dataset
    combined_df = pd.concat([existing_data_numeric, input_df], ignore_index=True)
    combined_df1 = pd.concat([existing_data_numeric,input_df1 ], ignore_index=True)
    # Predict the cluster using hierarchical clustering (AgglomerativeClustering)
    hierarchical_clusters = hierarchical.fit_predict(combined_df)
    hierarchical_predicted_cluster = hierarchical_clusters[-1]  # The cluster of the new input

    st.write(f'Predicted Hierarchical Cluster: {hierarchical_predicted_cluster}')


    kmeans_clusters = kmeans.fit_predict(combined_df1)
    kmeans_predicted_cluster = kmeans_clusters[-1]  # The cluster of the new input

    st.write(f'Predicted KMeans Cluster: {kmeans_predicted_cluster}')

