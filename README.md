# Getting Started

There are two ways to run the code:

- On linux or macos by installing `ffmpeg`, `youtube-dl` and `requirements.txt` in a python 3 virtual environment.
- Using the provided docker container which comes with all the required dependencies pre-installed.

## Run using virtualenv

By running the following steps 
```
virtualenv venv
. venv/bin/activate
python run.py
```
we are going to get the cli help message
```
usage: run.py [-h] --speakers SPEAKERS [--ground-truth GROUND_TRUTH]
              [--analysis-type {unimodal,multimodal}] [--no-cache]
              [--output-dir OUTPUT_DIR]
              (--audio-file AUDIO_FILE | --video-file VIDEO_FILE | --youtube-video-url YOUTUBE_VIDEO_URL)
```

## Run using docker

To run with docker we are going to use the pre-baked image of the algorithm available in docker hub. 
By running the following under the root of this repo 
```
docker run -v $(pwd):/volume -it nicktgr15/avlab-multimodal-speaker-diarization bash
```
we are going to get a command line prompt like the following
```
root@a1ae2484325a:/src#
```
Running `python3 run.py` will print the help message. 

To run the code against the test fixtures we can do the following 

```
python3 run.py --speakers 4 --video-file /volume/tests/fixtures/trimmed-video.mp4 --output-dir /volume --ground-truth /volume/tests/fixtures/ground_truth.txt --analysis-type multimodal

```
the `/volume` dir is a docker volume that is used to exchange data between the container and the host machine.
