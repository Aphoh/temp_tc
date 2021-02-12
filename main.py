from typing import 
from flask import Flask

app = Flask(__name__)

# front-facing endpoints


@app.route("/load_and_predict", methods=["GET"])
def load_and_predict(date):
    """
    """
    pass


@app.route("/load_previous_day_energy", methods=["GET"])
def load_previous_day_energy(person, date):
    pass


@app.route("/log_metrics", methods=["GET"])
def log_metrics():
    """
    Uses a pre-loaded model (or one saved to memory/disk) to output relevant 
    information for each of the participants (rewards, RL state information)
    into a CSV and output it for the user. 
    """
    pass


# back-end endpoints


@app.route("/train", methods=["GET", "POST"])
def train(data):
    """
    Data will be passed in as a CSV. TODO: Figure out how to use a POST endpoint 
    to load data as a CSV.
    """
    pass
