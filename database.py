"""
Interfacing with database and static data.
"""

import pandas as pd


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
