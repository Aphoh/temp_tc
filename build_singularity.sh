DOCKER_BUILDKIT=1 docker build . -t local/tc-temp
sudo singularity build tc-temp.sif docker-daemon://local/tc-temp:latest
