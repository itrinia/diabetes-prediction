import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

file_path = 'diabetes.csv'
df = load_data(file_path)

# Remove duplicates
df = df.drop_duplicates()

# Define function to remove outliers
def remove_outliers(dataframe, threshold=1.5):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1
    return dataframe[~((dataframe < (Q1 - threshold * IQR)) | (dataframe > (Q3 + threshold * IQR))).any(axis=1)]

# Remove outliers
df = remove_outliers(df)

# Preprocessing data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential()
model.add(Dense(8, input_dim=X_train_scaled.shape[1], activation='relu', name='dense_1'))
model.add(Dropout(0.2, name='dropout_1'))
model.add(Dense(32, activation='relu', name='dense_2'))
model.add(Dropout(0.2, name='dropout_2'))
model.add(Dense(64, activation='relu', name='dense_3'))
model.add(Dropout(0.2, name='dropout_3'))
model.add(Dense(128, activation='relu', name='dense_4'))
model.add(Dropout(0.2, name='dropout_4'))
model.add(Dense(32, activation='relu', name='dense_5'))
model.add(Dropout(0.2, name='dropout_5'))
model.add(Dense(16, activation='relu', name='dense_6'))
model.add(Dropout(0.2, name='dropout_6'))
model.add(Dense(1, activation='sigmoid', name='dense_output'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=1024, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

# Make predictions
predictions = model.predict(X_test_scaled)
binary_predictions = (predictions > 0.5).astype(int)

# Display metrics
st.write("Test Accuracy:", test_accuracy)

# Prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    user_data_scaled = scaler.transform(user_data)
    user_prediction = model.predict(user_data_scaled)
    return user_prediction

# User input
st.subheader("Diabetes Prediction: Enter User Data")
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

if st.button("Predict"):
    prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    st.write("Diabetes Prediction Probability:", prediction[0][0])
