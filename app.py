from config import END_OF_DAY_HOUR, END_OF_INITIAL_PERIOD, IS_MULTIAGENT, START_OF_DAY_HOUR
from datetime import datetime
import typing
from flask import request

from init import create_app, get_base_price_signal_for_day, load_square_data_df, 
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

base_signal_db = load_square_data_df()

###
# HELPER FUNCTIONS
###

def load_latest_model_params():
    raise NotImplementedError

def get_prices_from_model(params, is_multiagent):
    raise NotImplementedError

def is_initial_phase():
    return datetime.strptime(END_OF_INITIAL_PERIOD, "%d/%m/%Y") > datetime.now()

def update_model():
    """
    Use loaded energy consumption from the latest day
    to update model parameters.
    """
    raise NotImplementedError

def todays_energy_pricing():
    if is_initial_phase():
        current_participants = Participant.query.all()
        prices_per_person = [{
            "participant": participant.id,
            "pricing": [{"time": str(time), "price": get_base_price_signal_for_day(base_signal_db)} for
                        time in range(START_OF_DAY_HOUR, END_OF_DAY_HOUR + 1)]
        } for participant in current_participants]
    else:
        params = load_latest_model_params()
        prices_per_person = get_prices_from_model(
            params=params, is_multiagent=IS_MULTIAGENT
        )
    
    return prices_per_person

def get_base_points_for_all_participants():
    raise NotImplementedError

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
def get_energy_pricing():
    """
    """

    prices_per_person = todays_energy_pricing()
        
    return prices_per_person, 200


@app.route("/energy/consumption", methods=["POST"])
def submit_energy_consumption():

    req = request.get_json()

    date_string = req.get("date")
    timestamp = datetime.strptime(date_string, "%d/%m/%Y")
    
    acknowledgment = add_instance(Acknowledgments, timestamp)

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
                ack_id=acknowledgment.id
            )

        return {
            "acknowledged": "true",
            "acknowledgmentId": str(acknowledgment.id)
        }, 202


@app.route("/energy/points", methods=["GET"])
def get_points_with_base_points(acknowledgmentId):
    if is_initial_phase():
        update_model()
    
    latest_energy_vector = get_all_instances_from_key(EnergyUsage, "ack_id", acknowledgmentId)
    
    prices_per_person_vector = todays_energy_pricing()
    
    base_points_vector = get_base_points_for_all_participants()
    
    assert len(latest_energy_vector) == len(prices_per_person_vector) == len(base_points_vector), f"Price/energy/base points vectors are mismatched: energy: {len(latest_energy_vector)}, prices: {len(prices_per_person_vector)}, base_points: {len(base_points_vector)} "
    
    base_points_and_earned_per_person = calculate_earned_points(latest_energy_vector, prices_per_person_vector, base_points_vector)
    
    return {"acknowledgmentId": acknowledgmentId,
            "status": "completed",
            "values": base_points_and_earned_per_person}, 200


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