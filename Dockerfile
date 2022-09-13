# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY lung_ai/requirements.txt requirements.txt
COPY flask_app/requirements.txt web_requirements.txt


RUN pip3 install -r requirements.txt
RUN pip3 install -r web_requirements.txt

RUN apt-get update && \
    apt-get -y install libsndfile1


COPY . .

RUN pip3 install -e lung_ai/


CMD ["waitress-serve", "--port", "8080",  "--call", "flask_app.app:create_app"]
