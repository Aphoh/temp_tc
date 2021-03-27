Bootstrap: docker
From: nvcr.io/nvidia/pytorch:20.11-py3

%files
  requirements.txt /tc/requirements.txt 
  ./gym-socialgame/ /tc/gym-socialgame/

%post
  python -m pip install --upgrade pip
  sudo apt-get update && apt-get install -y libgl1-mesa-glx
  pip install -r /tc/requirements.txt
  pip install -e /tc/gym-socialgame/ 

