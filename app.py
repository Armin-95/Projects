import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

# load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# simple health check
@app.route('/health')
def health():
    return 'OK', 200

# home route to submit stock symbol
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# analysis route to process symbol
@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form.get('symbol', '').strip().upper()
    return render_template('analysis.html', symbol=symbol)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')