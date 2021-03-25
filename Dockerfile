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

USER $UNAME
