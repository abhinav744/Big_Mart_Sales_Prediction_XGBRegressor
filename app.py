import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Train.csv")
    return data

big_mart_data = load_data()

st.title("🏬 Big Mart Sales Prediction App")
st.write("This app predicts the **Item Outlet Sales** using XGBoost.")

# Handle missing values
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# Clean Item_Fat_Content
big_mart_data.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}}, inplace=True)

# Label Encoding
encoder = LabelEncoder()
cols = ['Item_Identifier','Item_Fat_Content','Item_Type',
        'Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
for col in cols:
    big_mart_data[col] = encoder.fit_transform(big_mart_data[col])

# Splitting data
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Sidebar for user input
st.sidebar.header("Enter Product & Outlet Details")

def user_input_features():
    Item_Weight = st.sidebar.slider("Item Weight", float(X['Item_Weight'].min()), float(X['Item_Weight'].max()), 10.0)
    Item_Fat_Content = st.sidebar.selectbox("Item Fat Content", sorted(big_mart_data['Item_Fat_Content'].unique()))
    Item_Visibility = st.sidebar.slider("Item Visibility", float(X['Item_Visibility'].min()), float(X['Item_Visibility'].max()), 0.05)
    Item_Type = st.sidebar.selectbox("Item Type", sorted(big_mart_data['Item_Type'].unique()))
    Item_MRP = st.sidebar.slider("Item MRP", float(X['Item_MRP'].min()), float(X['Item_MRP'].max()), 150.0)
    Outlet_Identifier = st.sidebar.selectbox("Outlet Identifier", sorted(big_mart_data['Outlet_Identifier'].unique()))
    Outlet_Establishment_Year = st.sidebar.selectbox("Establishment Year", sorted(big_mart_data['Outlet_Establishment_Year'].unique()))
    Outlet_Size = st.sidebar.selectbox("Outlet Size", sorted(big_mart_data['Outlet_Size'].unique()))
    Outlet_Location_Type = st.sidebar.selectbox("Outlet Location Type", sorted(big_mart_data['Outlet_Location_Type'].unique()))
    Outlet_Type = st.sidebar.selectbox("Outlet Type", sorted(big_mart_data['Outlet_Type'].unique()))

    data = {
        'Item_Identifier': Item_Type,  # placeholder
        'Item_Weight': Item_Weight,
        'Item_Fat_Content': Item_Fat_Content,
        'Item_Visibility': Item_Visibility,
        'Item_Type': Item_Type,
        'Item_MRP': Item_MRP,
        'Outlet_Identifier': Outlet_Identifier,
        'Outlet_Establishment_Year': Outlet_Establishment_Year,
        'Outlet_Size': Outlet_Size,
        'Outlet_Location_Type': Outlet_Location_Type,
        'Outlet_Type': Outlet_Type
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction
if st.button("🔮 Predict Sales"):
    prediction = regressor.predict(input_df)
    st.subheader("💰 Predicted Item Outlet Sales")
    st.success(f"₹ {prediction[0]:,.2f}")
