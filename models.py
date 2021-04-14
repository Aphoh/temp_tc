import flask
import flask_sqlalchemy

db = flask_sqlalchemy.SQLAlchemy()


class Game(db.Model):
    __tablename__ = "games"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    num_winners = db.Column(db.Integer)
    active_status = db.Column(db.Boolean)

    def __init__(self, id, name, start_date, end_date, num_winners, active_status):
        self.id = id
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.num_winners = num_winners
        self.active_status = active_status

    def __repr__(self):
        return "<id {}>".format(self.id)


class Participant(db.Model):
    __tablename__ = "participants"

    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey("games.id"))
    username = db.Column(db.String)

    def __init__(self, game_id, username):
        self.game_id = game_id
        self.username = username


class EnergyUsage(db.Model):
    __tablename__ = "energy_usage"

    id = db.Column(db.Integer, primary_key=True)
    participant_username = db.Column(db.String, db.ForeignKey("participants.username"))
    timestamp = db.Column(db.DateTime)
    type = db.Column(db.String)
    value = db.Column(db.Float)
    unit = db.Column(db.String)
    ack_id = db.Column(db.Integer, db.ForeignKey("acknowledgments.id"))

    def __init__(
        self, participant_username, timestamp, type, value, unit=None, ack_id=None
    ):
        self.participant_username = participant_username
        self.timestamp = timestamp
        self.value = value
        self.type = type
        self.unit = unit
        self.ack_id = ack_id


class Points(db.Model):
    __tablename__ = "points"

    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer, db.ForeignKey("participants.id"))
    game_id = db.Column(db.Integer, db.ForeignKey("games.id"))
    hour = db.Column(db.Integer)
    point_value = db.Column(db.Float)

    def __init__(self, participant_id, game_id, hour, point_value):
        self.participant_id = participant_id
        self.game_id = game_id
        self.hour = hour
        self.point_value = point_value


class BasePoints(db.Model):
    __tablename__ = "base_points"

    hour = db.Column(db.Integer, primary_key=True)
    point_value = db.Column(db.Float)

    def __init__(self, hour, point_value):
        self.hour, self.point_value = hour, point_value


class ModelParams(db.Model):
    __tablename__ = "model_params"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, unique=True)
    params = db.Column(db.Binary)
    ack_id = db.Column(db.Integer, db.ForeignKey("acknowledgments.id"))

    def __init__(self, timestamp, params, ack_id=None):
        self.timestamp, self.params, self.ack_id = timestamp, params, ack_id


class Acknowledgments(db.Model):
    __tablename__ = "acknowledgments"

    id = db.Column(db.Integer, primary_key=True)
    curr_timestamp = db.Column(db.DateTime, unique=True)
    game_id = db.Column(db.Integer, db.ForeignKey("games.id"))
    data_datetime = db.Column(db.DateTime)

    def __init__(self, curr_timestamp, data_datetime, game_id):
        self.curr_timestamp = curr_timestamp
        self.data_datetime = data_datetime
        self.game_id = game_id
