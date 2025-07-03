import numpy as np
from flask import Flask, request, render_template
import joblib

# Load the trained model
model = joblib.load('stock_sentiment_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        prediction = model.predict([news])[0]
        
        if prediction == 0:
            result = "The stock price will likely go DOWN or stay the SAME."
        else:
            result = "The stock price will likely go UP!"
        
        return render_template('index.html', prediction=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)
