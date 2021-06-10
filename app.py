from datetime import datetime
import flask
import logging
import os
import pytz
import typing
import traceback
from time import localtime, strftime

from flask import jsonify, request
from sqlalchemy.sql import text

from config import (
    BASE_POINTS_VALUE,
    DATE_FORMAT,
    END_OF_DAY_HOUR,
    IS_MULTIAGENT,
    START_OF_DAY_HOUR,
    TIMESTAMP_FORMAT,
    LOCAL_TIMEZONE,
)
from database import (
    add_instance,
    get_all_instances_from_key,
    get_instance_from_key,
    load_square_data_df,
    get_base_price_signal_for_day,
    load_square_data_df,
    get_all,
)
from init import create_app
from models import (
    Acknowledgments,
    BasePoints,
    EnergyUsage,
    Game,
    ModelParams,
    Participant,
    Points,
    db,
)

app = create_app()

base_signal_db = load_square_data_df()

###
# LOGGING FUNCTIONS
###


@app.before_first_request
def before_first_request():
    log_level = logging.DEBUG

    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)

    root = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(root, "app.log")
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    app.logger.addHandler(handler)

    app.logger.setLevel(log_level)


@app.before_request
def before():
    ts = strftime("[%Y-%b-%d %H:%M]")
    app.logger.debug(
        "\n\nRequest: %s %s %s %s %s",
        ts,
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
    )
    app.logger.debug("Request body: " + request.get_data(as_text=True))


# Useful debugging interceptor to log all endpoint responses
@app.after_request
def after(response):
    if response.status_code != 500:
        ts = strftime("[%Y-%b-%d %H:%M]", localtime())
        app.logger.debug(
            "Response: %s %s %s %s %s %s",
            ts,
            request.remote_addr,
            request.method,
            request.scheme,
            request.full_path,
            response.status,
        )
        app.logger.debug("Response data: " + response.get_data(as_text=True))
    return response


# Default handler for uncaught exceptions in the app
@app.errorhandler(500)
def internal_error(exception):
    """ Logging after every Exception. """
    ts = strftime("[%Y-%b-%d %H:%M]", localtime())
    tb = traceback.format_exc()
    app.logger.error(
        "%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s",
        ts,
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
        tb,
    )
    return "Internal Server Error", 500


# Default handler for all bad requests sent to the app
@app.errorhandler(400)
def handle_bad_request(e):
    app.logger.info("Bad request", e)
    return flask.make_response("bad request", 400)


###
# HELPER FUNCTIONS
###


def load_latest_model_params():
    raise NotImplementedError


def get_prices_from_model(params, is_multiagent):
    raise NotImplementedError


def is_initial_phase():
    return True


def calculate_earned_points(energy_vector, prices_vector, base_points_vector):
    if is_initial_phase():
        return base_points_vector

    else:
        raise NotImplementedError


def final_output_get_points(
    gameId,
    acknowledgmentId,
    base_points_per_participant,
    earned_points_per_participant,
):
    return {
        "gameId": gameId,
        "acknowledgementId": acknowledgmentId,
        "message": "completed",
        "users": [
            {
                "user": username,
                "basepoints": base_points_per_participant[username],
                "earnedpoints": earned_points_per_participant[username],
            }
            for username in base_points_per_participant.keys()
        ],
    }


def update_model():
    """
    Use loaded energy consumption from the latest day
    to update model parameters.
    """
    raise NotImplementedError


def todays_energy_pricing(gameId):
    if is_initial_phase():
        current_participants = get_all_instances_from_key(
            Participant, "game_id", text(gameId)
        )
        prices_per_person = [
            {
                "username": participant.username,
                "pricing": [
                    {
                        "houroftheday": str(time),
                        "rate": get_base_price_signal_for_day(
                            base_signal_db,
                            datetime.now(pytz.timezone(LOCAL_TIMEZONE)).date(),
                        )[time - START_OF_DAY_HOUR],
                    }
                    for time in range(START_OF_DAY_HOUR, END_OF_DAY_HOUR + 1)
                ],
            }
            for participant in current_participants
        ]
    else:
        params = load_latest_model_params()
        prices_per_person = get_prices_from_model(
            params=params, is_multiagent=IS_MULTIAGENT
        )

    return {"data": prices_per_person}


def get_base_points_for_game(game_id):
    participants = get_all_instances_from_key(Participant, "game_id", text(game_id))
    return {participant.username: BASE_POINTS_VALUE for participant in participants}


@app.route("/participants", methods=["PUT"])
def add_participants():

    req = request.get_json()

    game_id = req.get("gameId")
    name = req.get("name")
    active_status = req.get("active") == "true"
    start_date = datetime.strptime(req.get("startDate"), DATE_FORMAT).date()
    end_date = datetime.strptime(req.get("endDate"), DATE_FORMAT).date()
    num_winners = req.get("numWinners")
    participants = req.get("users")
    current_participant_names = {
        participant.username for participant in get_all(Participant)
    }

    stored_game = Game.query.get(game_id)
    if stored_game == None:
        stored_game = Game(
            game_id, name, start_date, end_date, num_winners, active_status
        )
        db.session.add(stored_game)
    for participant_name_dict in participants:
        participant_name = participant_name_dict.get("username")
        if participant_name not in current_participant_names:
            new_participant = Participant(
                game_id, participant_name_dict.get("username")
            )
            db.session.add(new_participant)
    db.session.commit()

    return {"accepted": "true"}, 202


@app.route("/energy/pricing", methods=["GET"])
def get_energy_pricing():
    """
    """
    req = request.get_json()

    gameId = req.get("gameId")
    prices_per_person = todays_energy_pricing(gameId)

    return prices_per_person, 200


@app.route("/energy/consumption", methods=["POST"])
def submit_energy_consumption():

    req = request.get_json()

    date_string = req.get("date")
    timestamp = datetime.strptime(date_string, TIMESTAMP_FORMAT)
    game_id = req.get("gameId")

    acknowledgment = add_instance(
        Acknowledgments,
        curr_timestamp=datetime.now(pytz.timezone(LOCAL_TIMEZONE)),
        data_datetime=timestamp,
        game_id=game_id,
    )

    for participant_info in req.get("users"):
        participant_name = participant_info.get("username")
        # participant_object = get_instance_from_key(
        #     Participant, "username", text(participant_name)
        # )
        # participant_id = participant_object.id

        for energy_input in participant_info.get("energy"):
            timestamp_string = energy_input.get("datetime")
            timestamp = datetime.strptime(timestamp_string, TIMESTAMP_FORMAT)
            value = float(energy_input.get("usage"))
            unit = energy_input.get("unit")
            add_instance(
                EnergyUsage,
                participant_username=participant_name,
                timestamp=timestamp,
                value=value,
                type="energy",
                unit=unit,
                ack_id=str(acknowledgment.id),
            )

        if "preference" in participant_info:
            for energy_type, vals_list in participant_info["preference"].items():
                for vals in vals_list:
                    timestamp_string = vals.get("datetime")
                    timestamp = datetime.strptime(timestamp_string, TIMESTAMP_FORMAT)
                    value = float(vals.get("value"))
                    add_instance(
                        EnergyUsage,
                        participant_username=participant_name,
                        timestamp=timestamp,
                        value=value,
                        type=energy_type,
                        ack_id=acknowledgment.id,
                    )

    return (
        {
            "gameId": game_id,
            "acknowledged": "true",
            "acknowledgmentId": str(acknowledgment.id),
        },
        202,
    )


@app.route("/energy/points", methods=["GET"])
def get_points_with_base_points():
    if not is_initial_phase():
        update_model()

    req = request.get_json()

    game_id = req.get("gameId")
    acknowledgment_id = req.get("acknowledgementId")

    latest_energy_vector = get_all_instances_from_key(
        EnergyUsage, "ack_id", text(acknowledgment_id)
    ).filter_by(type="energy")

    prices_per_person_vector = todays_energy_pricing(game_id)

    base_points_per_participant = get_base_points_for_game(game_id)

    # assert (
    #     len(latest_energy_vector)
    #     == len(prices_per_person_vector)
    #     == len(base_points_per_participant)
    # ), f"Price/energy/base points vectors are mismatched: energy: {len(latest_energy_vector)}, prices: {len(prices_per_person_vector)}, base_points: {len(base_points_vector)} "

    earned_points_per_participant = calculate_earned_points(
        latest_energy_vector, prices_per_person_vector, base_points_per_participant
    )

    combined_base_and_earned_points = final_output_get_points(
        game_id,
        acknowledgment_id,
        base_points_per_participant,
        earned_points_per_participant,
    )

    return (
        {
            "acknowledgmentId": acknowledgment_id,
            "status": "completed",
            "values": combined_base_and_earned_points,
        },
        200,
    )


@app.route("/game/winners", methods=["POST"])
def calculate_game_winners():

    req = request.get_json()

    sorted_points_mapping = sorted(
        req.get("users"),
        key=lambda mapping: int(mapping.get("earnedpoints")),
        reverse=True,
    )

    return_response = {
        "values": [
            {"username": mapping.get("username"), "rank": str(idx + 1)}
            for idx, mapping in enumerate(sorted_points_mapping)
        ]
    }

    return return_response
