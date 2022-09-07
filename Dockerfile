# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY lung_ai/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install flask

RUN apt-get update && \
    apt-get -y install libsndfile1


COPY . .

RUN pip3 install -e lung_ai/


CMD ["python3", "-m", "flask", "--app", "flask_app/app.py","run", "--host=0.0.0.0"]
