"""Simple SnapML App"""
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return f"""
    <h1>SnapML Simple Test</h1>
    <p>Current directory: {os.getcwd()}</p>
    <p>This is a simple test page for SnapML</p>
    """

if __name__ == '__main__':
    print("Starting simple SnapML app on port 8080...")
    app.run(host='0.0.0.0', port=8080) 