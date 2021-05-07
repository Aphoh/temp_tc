"""
Interfacing with database and static data.
"""

from datetime import date, datetime
import numpy as np
import pandas as pd

import config
from models import db


def get_all(model):
    data = model.query.all()
    return data


def get_instance_from_key(model, key_name, key_value):
    return model.query.filter(getattr(model, key_name), key_value)[0]


def get_all_instances_from_key(model, key_name, key_value):
    return model.query.filter(getattr(model, key_name), key_value)


def add_instance(model, **kwargs):
    instance = model(**kwargs)
    db.session.add(instance)
    commit_changes()
    return instance


def delete_instance(model, id):
    model.query.filter_by(id=id).delete()
    commit_changes()


def edit_instance(model, id, **kwargs):
    instance = model.query.filter_by(id=id).all()[0]
    for attr, new_value in kwargs:
        setattr(instance, attr, new_value)
    commit_changes()


def commit_changes():
    db.session.commit()


def load_square_data_df():
    return pd.read_csv(config.SQUARE_WAVE_DATA_PATH)


def get_base_price_signal_for_day(base_price_df: pd.DataFrame, date: date):
    column_index = (
        np.busday_count(
            date,
            datetime.strptime(config.START_OF_EXPERIMENT, config.DATE_FORMAT).date(),
        )
        % 10
    )
    prices = base_price_df.iloc[:, column_index + 1]

    return list(prices)