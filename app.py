from config import END_OF_INITIAL_PERIOD, IS_MULTIAGENT
from datetime import datetime
from flask import request

from init import create_app, 
from database import get_all_instances_from_key, get_instance_from_key, add_instance
from models import (
    BasePoints, db,
    Game,
    Participant,
    EnergyUsage,
    Points,
    ModelParams,
    Acknowledgments,
)

app = create_app()

###
# HELPER FUNCTIONS
###

def load_latest_model_params():
    raise NotImplementedError

def get_prices_from_model(params, is_multiagent):
    raise NotImplementedError

def is_initial_phase():
    return datetime.strptime(END_OF_INITIAL_PERIOD, "%d/%m/%Y") > datetime.now()

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

    return {"accepted": "true"}, 202


@app.route("/energy/pricing", methods=["GET"])
def get_energy_pricing(game_id):
    """
    """

    if is_initial_phase():
        current_participants = get_all_instances_from_key(
            Participant, "game_id", game_id
        )
        prices_per_person = [{
            "participant": participant.id,
            # "pricing": [{db.session.query(BasePoints)]
        }]
    else:
        params = load_latest_model_params()
        prices_per_person = get_prices_from_model(
            params=params, is_multiagent=IS_MULTIAGENT
        )

    req = request.get_json()

    for participant_list in req:
        pass

        pass
    pass


@app.route("/energy/consumption", methods=["POST"])
def submit_energy_consumption():

    req = request.get_json()

    date_string = req.get("date")
    timestamp = datetime.strptime(date_string, "%d/%m/%Y")

    for participant_info in req.get("values"):
        participant_name = participant_info.get("participant")
        participant_object = get_instance_from_key(
            Participant, "name", participant_name
        )
        participant_id = participant_object.id

        for energy_input in participant_info.get("energy"):
            timestamp_string = energy_input.get("datetime")
            timestamp = datetime.strptime(timestamp_string, "%d/%m/%Y %H:%M")
            value = float(energy_input.get("value"))
            unit = energy_input.get("unit")
            add_instance(
                EnergyUsage,
                participant_id=participant_id,
                timestamp=timestamp,
                value=value,
                unit=unit,
            )

        return {
            "accepted": "true"
        }, 202


@app.route("/energy/points", methods=["GET"])
def get_points_with_base_points():
    if is_initial_phase():
        pass
    else:
        pass
    pass


@app.route("/game/winners", methods=["POST"])
def calculate_game_winners():

    req = request.get_json()

    sorted_points_mapping = sorted(
        req.get("values"), key=lambda mapping: int(mapping.get("rank"))
    )

    return_response = {
        "values": [
            {"participant": mapping.get("participant"), "rank": str(idx + 1)}
            for idx, mapping in enumerate(sorted_points_mapping)
        ]
    }

    return return_response