"""App

This script allows connects and interacts with Flask and Dash components.

This script requires that `flask` and 'dash' be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * dataframe_return - list of Dataframes from dataframe_visualizer
    * root - returns the index page
    * run_code - executes Python code and returns the result
"""
from flask import Flask, render_template, request, redirect, url_for, jsonify
import dash_bootstrap_components as dbc
import dash
from dash import html
from libs.dataframe_visualizer import dataframe_visualizer
import io
import sys
import traceback
from contextlib import redirect_stdout


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True


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

@app.route('/DataVisualizer', methods=['POST', 'GET'])
def dataframe_return():
    """returns list of Dataframes from dataframe_visualizer

    Returns:
        string: list of datafrmaes
    """
    # pylint: disable=W0603
    global DASH_APP
    list_dataframe, DASH_APP = dataframe_visualizer(request.form, DASH_APP)
    return str(list_dataframe)


@app.route('/', methods=['POST', 'GET'])
def root():
    """renders undex.html

    Returns:
        _render: rendered html
    """
    return render_template('index.html')


@app.route('/run_code', methods=['POST'])
def run_code():
    """Execute Python code submitted from the front-end
    
    Returns:
        json: execution results
    """
    if request.method == 'POST':
        code = request.form.get('code', '')
        
        # 捕获标准输出
        output_buffer = io.StringIO()
        error_message = None
        result = None
        
        try:
            # 重定向标准输出以捕获打印语句
            with redirect_stdout(output_buffer):
                # 创建本地变量空间以存储代码执行结果
                local_vars = {}
                # 执行代码
                exec(code, globals(), local_vars)
                # 保存最后一个表达式的值（如果有）
                if '_' in local_vars:
                    result = local_vars['_']
        except Exception as e:
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        
        # 获取标准输出
        output = output_buffer.getvalue()
        
        # 返回JSON格式的响应
        return jsonify({
            'code': code,
            'output': output,
            'error': error_message,
            'result': result
        })
    
    return redirect(url_for('root'))


@app.route('/run_sklearn_example', methods=['GET'])
def run_sklearn_example():
    """Run a scikit-learn based example that doesn't require libomp
    
    Returns:
        json: execution results with example code and output
    """
    code = """
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris_data = sns.load_dataset("iris")
print("Iris dataset loaded successfully")
print(f"Dataset shape: {iris_data.shape}")

# Split the data
X = iris_data.drop(columns=['species'])
y = iris_data['species']
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.1, random_state=42)

# Clean data - drop NA values
def dropNa(train_X, test_X, train_Y, test_Y):
    train_X = train_X.dropna()
    train_Y = train_Y.loc[train_X.index.values.tolist()]
    test_X = test_X.dropna()
    test_Y = test_Y.loc[test_X.index.values.tolist()]
    return train_X, test_X, train_Y, test_Y

train_X, test_X, train_Y, test_Y = dropNa(train_X, test_X, train_Y, test_Y)

# Train a random forest model (similar to PyCaret's create_model('rf'))
RandomForest_ML = RandomForestClassifier(n_estimators=100, random_state=42)
RandomForest_ML.fit(train_X, train_Y)

# Make predictions (similar to PyCaret's predict_model())
predictions = RandomForest_ML.predict(test_X)
proba = RandomForest_ML.predict_proba(test_X)

# Create output dataframe similar to PyCaret's predict_model output
output = test_X.copy()
output['prediction_label'] = predictions
output['prediction_score'] = np.max(proba, axis=1)
output['target'] = test_Y.values

# Display results
print("\\nModel trained successfully")
print(f"Accuracy: {accuracy_score(test_Y, predictions):.4f}")
print("\\nClassification Report:")
print(classification_report(test_Y, predictions))

print("\\nSample predictions (first 5 rows):")
print(output.head())
"""
    
    # 捕获标准输出
    output_buffer = io.StringIO()
    error_message = None
    result = None
    
    try:
        # 重定向标准输出以捕获打印语句
        with redirect_stdout(output_buffer):
            # 创建本地变量空间以存储代码执行结果
            local_vars = {}
            # 执行代码
            exec(code, globals(), local_vars)
            # 保存最后一个表达式的值（如果有）
            if '_' in local_vars:
                result = local_vars['_']
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
    
    # 获取标准输出
    output = output_buffer.getvalue()
    
    # 返回JSON格式的响应
    return jsonify({
        'code': code,
        'output': output,
        'error': error_message,
        'result': result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
