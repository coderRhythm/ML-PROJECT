from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['SPX']), float(request.form['USO']), float(request.form['SLV']), float(request.form['EUR/USD'])]
    input_features = np.array(features).reshape(1, -1)
    prediction = regressor.predict(input_features)
    predicted_gold_price = round(prediction[0], 2)
    return redirect(url_for('show_prediction', predicted_gold_price=predicted_gold_price))

@app.route('/prediction/<float:predicted_gold_price>')
def show_prediction(predicted_gold_price):
    return render_template('prediction.html', predicted_gold_price=predicted_gold_price)

if __name__ == '__main__':
    app.run(debug=True)