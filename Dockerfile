# This necc when running our SG: FROM "nvcr.io/nvidia/tensorflow:20.11-tf1-py3"

FROM python:3.6

COPY ./transactive_control/gym-socialgame/ ./gym-socialgame/
RUN pip install -e ./gym-socialgame/ 

COPY ./transactive_control/rl_algos/ ./rl_algos/
RUN pip install -e ./rl_algos/stableBaselines/

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY models.py /app/
COPY app.py /app/
WORKDIR /app

RUN export FLASK_APP=app.py
EXPOSE 5000

CMD ["/usr/local/bin/flask", "run", "--host", "0.0.0.0"]
