#!/bin/sh
mkdir /usr/bin/nvidia-smi
mkdir /usr/share/zoneinfo/Etc/UTC
pip install --user --upgrade pip
pip install -r ./requirements.txt
pip install -e ./gym-socialgame/
pip install -e ./gym-microgrid/
