
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__,static_folder='templates/static')
CORS(app)

model = joblib.load("best_adaboost_model.pkl",mmap_mode='r')
scaler = joblib.load("scaler.pkl",mmap_mode='r')
feature = pd.read_csv('feature_columns.csv')
feature_columns = feature.values.flatten().tolist()

def preprocess_and_predict(user_input, scaler, model, feature_columns):
    user_df = pd.DataFrame([user_input], columns=feature_columns)
    user_df = pd.get_dummies(user_df, drop_first=True)
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[feature_columns]
    user_scaled = scaler.transform(user_df)
    prediction = model.predict_proba(user_scaled)[:, 1]
    return prediction[0]

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Extract user input from the request
    age = int(request.form.get('age'))
    hypertension = int(request.form.get('hypertension'))
    heart_disease = int(request.form.get('heart_disease'))
    ever_married = request.form.get('ever_married')
    avg_glucose_level = float(request.form.get('avg_glucose_level'))
    bmi = float(request.form.get('bmi'))

    user_input = [age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi]
    print(user_input)
    # Predict the probability of stroke
    stroke_probability = preprocess_and_predict(user_input, scaler, model, feature_columns)
    stroke_probability *= 100
    
    # Return the result
    return render_template('result.html', probability=f'{stroke_probability:.2f}%')
    # return render_template('result.html', probability='0.5')
    
if __name__ == '__main__':
    app.run(debug=True)
