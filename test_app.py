from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "SnapML is running!"

if __name__ == '__main__':
    print("Starting test Flask application on port 5050...")
    app.run(host='0.0.0.0', port=5050) 