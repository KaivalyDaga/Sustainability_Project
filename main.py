import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

# Preprocessing and training for Sustainability Metrics
def preprocess_data(data):
    data = pd.get_dummies(data, columns=['Kharif Crop', 'Rabi Crop', 'State', 'Sustainable Method (1)', 'Sustainable Method (2)'])
    X = data.drop(['Carbon Sequestration', 'Cost', 'Yield Increase (%)', 'Emission Reduction (%)'], axis=1)
    y = data[['Carbon Sequestration', 'Cost', 'Yield Increase (%)', 'Emission Reduction (%)']]
    return X, y, X.columns

def train_sustainability_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_output_model = MultiOutputRegressor(model)
    multi_output_model.fit(X, y)
    return multi_output_model

# Preparing Sustainable Methods input data
def prepare_input_sus_method(data, feature_names):
    # Unique values seen during training for Kharif Crops, Rabi Crops, and States
    unique_kharif_crops = ['Rice', 'Maize', 'Sorghum', 'Cotton', 'Sugarcane', 'Soybean', 'Pearl Millet', 'Groundnut', 'Chickpea']  # Add other Kharif Crops seen during training
    unique_rabi_crops = ['Wheat', 'Mustard','Chickpea', 'Barley', 'Gram', 'Lentils']  # Add other Rabi Crops seen during training
    unique_states = ['Punjab', 'Haryana', 'Maharashtra', 'Gujarat', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan', 'Andhra Pradesh', 'Tamil Nadu', 'Karnataka']  # Add other States seen during training

    # Initialize lists to hold the one-hot encoded values
    kharif_values = [1 if crop in data['Kharif Crop'] else 0 for crop in unique_kharif_crops]
    rabi_values = [1 if crop in data['Rabi Crop'] else 0 for crop in unique_rabi_crops]
    state_values = [1 if state in data['State'] else 0 for state in unique_states]

    # Create a dictionary for input data
    new_data = {'Kharif Crop_' + crop: val for crop, val in zip(unique_kharif_crops, kharif_values)}
    new_data.update({'Rabi Crop_' + crop: val for crop, val in zip(unique_rabi_crops, rabi_values)})
    new_data.update({'State_' + state: val for state, val in zip(unique_states, state_values)})
    new_data['Acres'] = data['Acres']

    # Reorder the columns according to the observed feature names
    new_data = {col: new_data[col] for col in feature_names}

    # Create a DataFrame using the prepared data
    new_data = pd.DataFrame(new_data)
    return new_data
# Training Sustainable Methods models (model_sus_method_1 and model_sus_method_2)

# Load your datasets
data_sustainability = pd.read_csv('demo_data.csv')  # Replace with sustainability dataset
X_sustainability, y_sustainability, feature_names = preprocess_data(data_sustainability)
sustainability_model = train_sustainability_model(X_sustainability, y_sustainability)


data_sus_method = pd.read_csv('sus_method.csv')  # Replace with sus method dataset
data_sus_method = data_sus_method.fillna('NA')

X_sus_method = data_sus_method[['Kharif Crop', 'Rabi Crop', 'State', 'Acres']]
y_sus_method = data_sus_method[['Sustainable Method (1)', 'Sustainable Method (2)']]
X_sus_method = pd.get_dummies(X_sus_method)

model_sus_method_1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_sus_method_1.fit(X_sus_method, y_sus_method['Sustainable Method (1)'])
model_sus_method_2 = RandomForestClassifier(n_estimators=100, random_state=42)
model_sus_method_2.fit(X_sus_method, y_sus_method['Sustainable Method (2)'])
#predictions_sus_method_2 = model_sus_method_2.predict(X)
feature_names_1 = model_sus_method_1.feature_names_in_
feature_names_2 = model_sus_method_2.feature_names_in_

st.title('Sustainable Farming Prediction')

# Sidebar input
st.sidebar.title('Input Parameters')
kharif_crop = st.sidebar.selectbox('Kharif Crop', data_sustainability['Kharif Crop'].unique())
rabi_crop = st.sidebar.selectbox('Rabi Crop', data_sustainability['Rabi Crop'].unique())
state = st.sidebar.selectbox('State', data_sustainability['State'].unique())
acres = st.sidebar.slider('Acres', 1, 25, 10)

user_input = pd.DataFrame({'Kharif Crop': [kharif_crop], 'Rabi Crop': [rabi_crop], 'State': [state], 'Acres': [acres]})
user_input_encoded = pd.get_dummies(user_input, columns=['Kharif Crop', 'Rabi Crop', 'State'])
user_input_encoded = user_input_encoded.reindex(columns=feature_names, fill_value=0)

# Prediction for Sustainability Metrics
predicted_metrics = sustainability_model.predict(user_input_encoded)
trimmed_predicted_metrics = [round(metric, 2) if isinstance(metric, float) else metric for metric in predicted_metrics]

st.write("## Predictions for Sustainability Metrics")
st.write(f"### Carbon Sequestration: {trimmed_predicted_metrics[0][0]:.2f}")
st.write(f"### Cost: {trimmed_predicted_metrics[0][1]:.2f}")
st.write(f"### Yield Improvement: {trimmed_predicted_metrics[0][2]:.2f}%")
st.write(f"### Emission Reduction: {trimmed_predicted_metrics[0][3]:.2f}%")

if st.sidebar.button('Predict Sustainable Methods'):
    new_data_sus_method_1 = prepare_input_sus_method(user_input, feature_names_1)
    new_data_sus_method_2 = prepare_input_sus_method(user_input, feature_names_2)
    predicted_sus_method_1 = list(model_sus_method_1.predict(new_data_sus_method_1))[0]
    predicted_sus_method_2 = list(model_sus_method_2.predict(new_data_sus_method_2))[0]

    st.write("## Predicted Sustainable Methods")
    st.write(f"### Sustainable Method 1: {predicted_sus_method_1}")
    st.write(f"### Sustainable Method 2: {predicted_sus_method_2}")
