from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
pipeline = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        input_data = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': request.form['smoker'],
            'region': request.form['region']
        }

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Debug: Print the input DataFrame
        print("Input DataFrame:")
        print(input_df)

        # Predict charges for the input data
        predicted_charge = pipeline.predict(input_df)[0]

        return render_template('result.html', predicted_charge=predicted_charge)

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)







