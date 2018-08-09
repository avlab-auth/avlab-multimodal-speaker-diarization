FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install vim python3-pip python3 -y
RUN apt-get install youtube-dl sox ffmpeg -y
RUN apt-get install python3-tk -y
RUN apt-get install cmake -y

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src /src

WORKDIR src
