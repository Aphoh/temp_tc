from typing import List
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask import (
    Flask,
    request,
)

app = Flask(__name__)
# app.config.from_object(os.environ["APP_SETTINGS"])
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
db = SQLAlchemy(app)

from models import Game, Participant, EnergyUsage, Points

# front-facing endpoints


@app.route("/participants", methods=["POST"])
def add_participants():

    req = request.get_json()

    game_id = req.get("gameId")
    participants = req.get("participants")

    stored_game = Game.query.get(game_id)
    if stored_game == None:
        stored_game = Game(game_id, "social_game")
        db.session.add(stored_game)
    for participant_name in participants:
        new_participant = Participant(game_id, participant_name)
        db.session.add(new_participant)
    db.session.commit()

    return "Done!"

@app.route("/docker_test")
def hello():
    return "Hello world!"


# @app.route("/load_and_predict", methods=["GET"])
# def load_and_predict(date):
#     """
#     Outputs: A dictionary of structure:

#           {
#             "participant": "ramit",
#             "pricing": [
#                 {
#                     "time": "11",
#                     "price": "4"
#                 },
#                 {
#                     "time": "11",
#                     "price": "5"
#                 },
#                 {
#                     "time": "13",
#                     "price": "2"
#                 }
#             ]
#         },


#     """
#     pass


# # calculate game winners

# # get points earned and base points


# @app.route("/load_previous_day_energy", methods=["GET"])
# def load_previous_day_energy(person, date):

#     """ @Manan does this need to be external facing?
#     """

#     pass


# @app.route("/log_metrics", methods=["GET"])
# def log_metrics():
#     """
#     Uses a pre-loaded model (or one saved to memory/disk) to output relevant
#     information for each of the participants (rewards, RL state information)
#     into a CSV and output it for the user.
#     """
#     pass


# # back-end endpoints


# @app.route("/train", methods=["GET", "POST"])
# def train(data):
#     """
#     Data will be passed in as a CSV. TODO: Figure out how to use a POST endpoint
#     to load data as a CSV.

#     (Internal facing)

#     """
#     pass
