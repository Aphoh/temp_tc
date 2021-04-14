DOCKER_BUILDKIT=1 docker build . -t social_game_api_docker

docker run -p 5000:5000 social_game_api_docker