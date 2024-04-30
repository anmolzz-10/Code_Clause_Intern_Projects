# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained SVM model
svm_model = joblib.load('model.pkl')

# Define mappings for categorical features
cp_mapping = {'typical': 0, 'atypical': 1, 'non-anginal': 2, 'asymptomatic': 3}
restecg_mapping = {'normal': 0, 'stt_wave_abnormality': 1, 'left_ventricular_hypertrophy': 2}
slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
thal_mapping = {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2, 'critical': 3}
gender_mapping = {'male': 0, 'female': 1}

@app.route('/')
def index():
    """Render the index.html template."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make predictions."""
    if request.method == 'POST':
        # Get user input from the form
        age = int(request.form['age'])
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = request.form['restecg']
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = request.form['slope']
        ca = int(request.form['ca'])
        thal = request.form['thal']
        
        # Map categorical values to numerical representations
        cp = cp_mapping.get(cp)
        restecg = restecg_mapping.get(restecg)
        slope = slope_mapping.get(slope)
        thal = thal_mapping.get(thal)
        sex = gender_mapping.get(sex)
        
        # Make predictions
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = svm_model.predict(input_data)[0]

        # Render the prediction result template with the predicted class
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
