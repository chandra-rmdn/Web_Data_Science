import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

diabets_dataset =pd.read_csv('diabetes.csv')

# Memisahkan data dan label
x = diabets_dataset.drop('Outcome', axis=1)

# 3. Standarisasi Data
scaler = StandardScaler()
scaler.fit(x)
standarized_data = scaler.transform(x)

x = standarized_data
y = diabets_dataset['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Membuat aplikasi Flask
app = Flask(__name__)

@app.route('/Diabets_Prediction', methods=['GET', 'POST'])
def predict():
    # Mengambil data dari form
    if request.method == 'POST':
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        input_data_as_dataframe = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        std_data = scaler.transform(input_data_as_dataframe)

        prediction = classifier.predict(std_data)
        
        if (prediction[0] == 0):
            result = 'Pasien tidak memiliki diabetes'
        else:
            result = 'Pasien memiliki diabetes'

        return render_template('index.html', result=result, prediction=prediction)
        
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)