FROM "nvcr.io/nvidia/pytorch:20.11-py3"

# You should really specify these unless running the run.sh script
ARG UNAME=tc
ARG UID=1000
ARG GID=1000

RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN useradd --create-home --shell /bin/bash -u $UID $UNAME
WORKDIR /home/$UNAME

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY ./gym-socialgame/ ./gym-socialgame/
RUN pip install -e ./gym-socialgame/ 

ENV TUNE_RESULT_DIR=/home/$UNAME/logs

USER $UNAME

COPY models.py /app/
COPY app.py /app/  
COPY init.py /app/
COPY config.py /app/  
COPY database.py /app/
WORKDIR /app

RUN export FLASK_APP=app.py
EXPOSE 5000

CMD ["/usr/local/bin/flask", "run", "--host", "0.0.0.0"]
