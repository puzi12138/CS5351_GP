"""App

This script connects and interacts with Flask and Dash components.

This script requires that `flask` and `dash` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * dataframe_return - returns list of dataframes from dataframe_visualizer
    * root - returns the index page
"""
from flask import Flask, render_template, request, send_from_directory
import dash_bootstrap_components as dbc
import dash
from dash import html
from libs.dataframe_visualizer import dataframe_visualizer
import os


# Get absolute path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)
            
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Templates directory: {TEMPLATE_DIR}")
print(f"Templates directory exists: {os.path.exists(TEMPLATE_DIR)}")
print(f"Template files: {os.listdir(TEMPLATE_DIR) if os.path.exists(TEMPLATE_DIR) else 'Directory not found'}")


DASH_APP = dash.Dash(
    routes_pathname_prefix='/visualizer/',
    server=app,
    external_scripts=[
        'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.js',
        'custom-script.js'
    ],
    external_stylesheets=[
        'https://fonts.googleapis.com/css?family=Lato',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
        'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/theme/monokai.min.css',
        'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.css',
        'styles.css',
        dbc.themes.BOOTSTRAP
    ],
    name='CSV Visualizer',
    title='CSV Visualizer'
)


DASH_APP.config.suppress_callback_exceptions = True

DASH_APP.validation_layout = html.Div()

DASH_APP.layout = html.Div()

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route('/DataVisualizer', methods=['POST', 'GET'])
def dataframe_return():
    """Returns list of dataframes from dataframe_visualizer

    Returns:
        string: list of dataframes
    """
    # pylint: disable=W0603
    global DASH_APP
    list_dataframe, DASH_APP = dataframe_visualizer(request.form, DASH_APP)
    return str(list_dataframe)


@app.route('/', methods=['POST', 'GET'])
def root():
    """Renders index.html

    Returns:
        _render: rendered html
    """
    print("Root route accessed!")
    return render_template('index.html')


if __name__ == '__main__':
    print("Starting Flask application on port 5004...")
    app.run(host='0.0.0.0', port=5004, debug=True) 