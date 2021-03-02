import flask
import flask_sqlalchemy

db = flask_sqlalchemy.SQLAlchemy()


class Game(db.Model):
    __tablename__ = "games"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __repr__(self):
        return "<id {}>".format(self.id)


class Participant(db.Model):
    __tablename__ = "participants"

    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey("games.id"))
    phone_num = db.Column(db.String)
    name = db.Column(db.String)

    def __init__(self, game_id, name, phone_num=None):
        self.game_id = game_id
        self.name = name
        self.phone_num = None


class EnergyUsage(db.Model):
    __tablename__ = "energy_usage"

    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer, db.ForeignKey("participants.id"))
    timestamp = db.Column(db.DateTime, unique=True)
    value = db.Column(db.Float)
    unit = db.Column(db.String)

    def __init__(self, participant_id, timestamp, value, unit):
        self.participant_id = participant_id
        self.timestamp = timestamp
        self.value = value
        self.unit = unit


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


class ModelParams(db.Model):
    __tablename__ = "model_params"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, unique=True)
    params = db.Column(db.Binary)
    ack_id = db.Column(db.Integer, db.ForeignKey("acknowledgments.id"))


class Acknowledgments(db.Model):
    __tablename__ = "acknowledgments"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, unique=True)

    def __init__(self, timestamp):
        self.timestamp = timestamp

