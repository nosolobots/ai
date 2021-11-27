FROM ubuntu:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip python3-dev build-essential python3-flask

COPY requirements.txt  /app/

WORKDIR /app

RUN pip install -r requirements.txt

COPY .flaskenv /app/
COPY app/ app/

CMD ["flask","run"]




