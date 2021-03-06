FROM "nvcr.io/nvidia/tensorflow:20.11-tf1-py3"

# You should really specify these unless running the run.sh script
ARG UNAME=tc
ARG UID=1000
ARG GID=1000

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN useradd --create-home --shell /bin/bash -u $UID $UNAME
WORKDIR /home/$UNAME

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY ./gym-socialgame/ ./gym-socialgame/
RUN pip install -e ./gym-socialgame/ 

COPY ./rl_algos/ ./rl_algos/
RUN pip install -e ./rl_algos/stableBaselines/

USER $UNAME
