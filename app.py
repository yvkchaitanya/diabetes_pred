from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = joblib.load('diabetes_model.joblib')

# Home route - displays input form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route - processes form data and displays result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input with default fallback
        form_data = {
            'Pregnancies': float(request.form.get('Pregnancies', 0)),
            'Glucose': float(request.form.get('Glucose', 0)),
            'BloodPressure': float(request.form.get('BloodPressure', 0)),
            'SkinThickness': float(request.form.get('SkinThickness', 0)),
            'Insulin': float(request.form.get('Insulin', 0)),
            'BMI': float(request.form.get('BMI', 0)),
            'DiabetesPedigreeFunction': float(request.form.get('DiabetesPedigreeFunction', 0)),
            'Age': float(request.form.get('Age', 0))
        }

        # Create feature array in the expected order
        features = np.array([[form_data[key] for key in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100  # Probability of being diabetic

        return render_template('result.html',
                               prediction=int(prediction),
                               prediction_text="Likely Diabetic" if prediction == 1 else "Not Diabetic",
                               prediction_prob=round(probability, 2),
                               form_data=form_data)

    except Exception as e:
        return render_template('result.html',
                               error=str(e),
                               prediction=None,
                               prediction_text="Error in prediction",
                               prediction_prob=None)

if __name__ == '__main__':
    app.run(debug=True)
