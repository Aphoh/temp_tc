from datetime import datetime, date
from flask import Flask
import flask_sqlalchemy
import logging
import numpy as np
import pandas as pd

from models import db
import config


def create_app():
    flask_app = Flask(__name__)
    logging.basicConfig(
        filename="events.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
    )
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_CONNECTION_URI
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    flask_app.app_context().push()
    db.init_app(flask_app)
    db.create_all()
    return flask_app


def load_square_data_df():
    return pd.read_csv(config.SQUARE_WAVE_DATA_PATH)


def get_base_price_signal_for_day(base_price_df: pd.DataFrame, date: date):
    column_index = (
        np.busday_count(
            date, datetime.strptime(config.START_OF_EXPERIMENT, "%d/%m/%Y").date
        ).days()
        % 10
    )
    prices = base_price_df.iloc[column_index]

    return prices
