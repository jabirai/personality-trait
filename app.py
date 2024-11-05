import streamlit as st
import plotly.graph_objs as go
import joblib
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_excel('combined_hvp_numeric.xlsx')
Traits = dataset.drop(columns=['Message','ID'])
y = Traits.values
scaler = MinMaxScaler()
y = scaler.fit_transform(y)
pca = PCA(n_components=20)
pca.fit(y)

vectorizer = TfidfVectorizer(max_features=512)
tuned_params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100
}
xgb_model = MultiOutputRegressor(XGBRegressor(**tuned_params))

def scale_back(scaler:MinMaxScaler,vector):
    return scaler.inverse_transform(vector.reshape(1,-1))

def predict_traits(input_text):
    y_pred_reduced = pipeline.predict([input_text])
    y_pred_full = pca.inverse_transform(y_pred_reduced)[0]
    y_pred_full = np.array([float(score) for score in y_pred_full])
    trait_names = list(Traits.columns)
    trait_dict = dict(zip(trait_names, [round(float(trait),4) for trait in scale_back(scaler,y_pred_full)[0]] ))
    return trait_dict,y_pred_full

if __name__ == "__main__":
    
    pipeline = joblib.load('my_pipeline.pkl')
    st.title("HVP | Personality Trait Prediction")
    user_input = st.text_input("Enter a text here:")
    if user_input:

        y,y_out = predict_traits(user_input)

        st.write("User input: \n\n",f"{user_input}")
        
        fig = go.Figure(data=go.Scatter(x=list(Traits.columns), y=scale_back(scaler,y_out)[0], mode='lines+markers', name='Line Plot'))

        st.plotly_chart(fig)

        st.write("HVP Scores\n:",y)
        
        
