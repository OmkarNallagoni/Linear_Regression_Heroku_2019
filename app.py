from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('house_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    MedInc = float(request.form['MedInc'])
    HouseAge = float(request.form['HouseAge'])
    Population = float(request.form['Population'])
    AveOccup = float(request.form['AveOccup'])

    # Create a numpy array with the form data
    data = np.array([[MedInc, HouseAge, Population, AveOccup]])

    # Make prediction using the loaded model
    prediction = model.predict(data)[0]

    # Render the template with the prediction
    return render_template('index1.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=8000)
