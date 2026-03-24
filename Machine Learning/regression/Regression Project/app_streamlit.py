import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 USA House Price Prediction")
st.markdown("Select a model and enter house features to get a price prediction.")

@st.cache_resource
def load_models():
    model_names = [
        'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
        'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM',
        'XGBoost', 'KNN'
    ]
    models = {}
    for name in model_names:
        try:
            models[name] = pickle.load(open(f'{name}.pkl', 'rb'))
        except FileNotFoundError:
            st.warning(f"Model file {name}.pkl not found. Please run model.py first.")
    return models

@st.cache_data
def load_results():
    try:
        return pd.read_csv(r'c:\Users\Amrutha Thalla\FSDS\DataScience_AI\Machine Learning\Regression Project\model_evaluation_results.csv')
    except FileNotFoundError:
        st.warning("Evaluation results not found. Please run model.py first.")
        return None

models = load_models()
results_df = load_results()



st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.header("Input Features")
income = st.sidebar.number_input("Avg. Area Income", value=50000.0, step=1000.0)
house_age = st.sidebar.number_input("Avg. Area House Age", value=5.0, step=0.1)
rooms = st.sidebar.number_input("Avg. Area Number of Rooms", value=6.0, step=0.1)
bedrooms = st.sidebar.number_input("Avg. Area Number of Bedrooms", value=3.0, step=0.1)
population = st.sidebar.number_input("Area Population", value=20000.0, step=1000.0)

predict_btn = st.sidebar.button("Predict Price")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    if predict_btn:
        input_data = pd.DataFrame([[income, house_age, rooms, bedrooms, population]],
                                   columns=['Avg. Area Income', 'Avg. Area House Age',
                                            'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                                            'Area Population'])
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: **${prediction:,.2f}**")
    else:
        st.info("Enter values in the sidebar and click Predict.")

with col2:
    st.subheader("Model Evaluation Results")
    if results_df is not None:
        st.dataframe(results_df)
    else:
        st.write("No results available.")

with st.expander("About the selected model"):
    st.write(f"**{model_name}**")