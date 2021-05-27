#!/bin/sh
python3 -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
python -m pip install -e ./gym-socialgame/
python -m pip install -e ./gym-microgrid/
