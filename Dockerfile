FROM "nvcr.io/nvidia/pytorch:20.11-py3"

# FROM python:3.6

# You should really specify these unless running the run.sh script
# ARG UNAME=tc
# ARG UID=$(id -u)
# ARG GID=$(id -g)

RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# COPY ./transactive_control/gym-socialgame/ ./gym-socialgame/
# RUN pip install -e ./gym-socialgame/ 

# COPY ./transactive_control/rl_algos/ ./rl_algos/
# RUN pip install -e ./rl_algos/stableBaselines/

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY models.py /app/
COPY app.py /app/  
COPY init.py /app/
COPY config.py /app/  
COPY database.py /app/


ENV FLASK_APP app.py #all users should see this
EXPOSE 5000

CMD ["/usr/local/bin/flask", "run", "--host", "0.0.0.0"]
