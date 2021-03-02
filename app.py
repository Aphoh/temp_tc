from datetime import datetime
from typing import List
from flask import Flask, request, Response
from flask_sqlalchemy import SQLAlchemy

from init import create_app
from database import get_instance_from_key, add_instance
from models import (
    db,
    Game,
    Participant,
    EnergyUsage,
    Points,
    ModelParams,
    Acknowledgments,
)

app = create_app()


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

    return Response("{'accepted':'true'}", status=202, mimetype="application/json")


@app.route("/energy/pricing", methods=["GET"])
def get_energy_pricing():
    """
    1. monday morning -- get_energy_pricing for monday day (requires state space information)
        i. needs friday's energy consumption, and grid information (pre-stored)
    2. monday night -- submit_energy_consumption for monday day
        i. energy consumption stored in database
    3. monday night -- get_points_and_base_points
        i. Update model (can be done in parallel with calculating earned points for monday day).
            a. Use loaded energy consumption to update model parameters
            b. Store model parameters in database, along with ack_id
        ii. Calculate earned points for monday day
            a. Depends on prices for monday day (already calculated monday morning, monday's energy, base_points)
    4. tuesday morning - get energy_pricing for tuesday day. 
        i. Load latest model parameters, use them to generate newest price signal for current day
        ii. Store price signal in database
        iii. Return price signal
    """

    req = request.get_json()

    for participant_list in req:

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

        acknowledgment_instance = add_instance(
            Acknowledgments, timestamp=datetime.now()
        )

        return {
            "acknowledged": "true",
            "acknowledgmentId": str(acknowledgment_instance.id),
        }


@app.route("/energy/points", methods=["GET"])
def get_points_with_base_points():
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
