from flask import Flask
import flask_sqlalchemy
from flask_migrate import Migrate
import logging


import config

from models import db


def create_app():
    flask_app = Flask(__name__)

    log = logging.getLogger("werkzeug")
    log.disabled = True

    flask_app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_CONNECTION_URI
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    flask_app.app_context().push()

    db.init_app(flask_app)
    migrate = Migrate(flask_app, db, render_as_batch=True)

    db.create_all()
    db.session.commit()
    return flask_app
