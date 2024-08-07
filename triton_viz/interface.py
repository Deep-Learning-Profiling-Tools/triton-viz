import threading
import webbrowser
import json
from flask import Flask, render_template, jsonify
from .analysis import analyze_records
from .draw import collect_grid,get_visualization_data
from .tooltip import get_tooltip_data
import pandas as pd
import os


app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    analysis_data = analyze_records()

    df = pd.DataFrame(analysis_data, columns=["Metric", "Value"])
    analysis_with_tooltip = get_tooltip_data(df)

    return jsonify({"ops":get_visualization_data()})

def run_flask():
    app.run(port=5000,host='127.0.0.1')

def launch(share=True):
    print("Launching Triton viz tool")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    return flask_thread  

def stop_server(flask_thread):
    # Implement a way to stop the Flask server
    pass