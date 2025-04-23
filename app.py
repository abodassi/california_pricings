from flask import Flask, render_template, request, session
import joblib
import numpy as np
import datetime
import os
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.secret_key = "california_house_predictor_secret_key"
Bootstrap(app)

# Load trained model
model = joblib.load("xgb_model.pkl")

# Initialize session history if not exists
@app.before_request
def before_request():
    if 'history' not in session:
        session['history'] = []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    if request.method == 'POST':
        # Get form data
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = int(request.form['housing_median_age'])
        total_rooms = int(request.form['total_rooms'])
        total_bedrooms = int(request.form['total_bedrooms'])
        population = int(request.form['population'])
        households = int(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean_proximity_inland = int(request.form['ocean_proximity_inland'])
        ocean_proximity_near_bay = 1 if 'ocean_proximity_near_bay' in request.form else 0
        ocean_proximity_near_ocean = 1 if 'ocean_proximity_near_ocean' in request.form else 0
        rooms_per_household = float(request.form['rooms_per_household'])
        
        # Prepare input for prediction
        user_data = np.array([[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            ocean_proximity_inland, ocean_proximity_near_bay,
            ocean_proximity_near_ocean, rooms_per_household
        ]])
        
        # Make prediction
        prediction = model.predict(user_data)
        result = round(prediction[0], 2)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update history
        history = session.get('history', [])
        history.append({
            "time": timestamp,
            "price": result,
            "input": user_data.tolist()[0]
        })
        session['history'] = history[-5:]  # Keep only last 5 predictions
        
        prediction_result = f"${result:,.2f}"
    
    return render_template('index.html', prediction_result=prediction_result, history=session.get('history', []))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True)