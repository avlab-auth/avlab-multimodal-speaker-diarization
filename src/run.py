import argparse
import json
import shutil

import subprocess
from tempfile import gettempdir
import os
import librosa
import uuid
import hashlib
import re
from sac.util import Util

from steps.audio_based_segmentation import generate_audio_based_segmentation
from steps.face_based_segmentation import extract_images_from_video, generate_face_based_segmentation
from steps.fusion import calculate_fusion
from sac.util import Util, AudacityLabel
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


def calculate_der(reference_filename, hypothesis_filename):
    lbls = Util.read_audacity_labels(reference_filename)
    reference = Annotation()
    for lbl in lbls:
        reference[Segment(lbl.start_seconds, lbl.end_seconds)] = lbl.label

    predicted_lbls = Util.read_audacity_labels(hypothesis_filename)
    hypothesis = Annotation()
    for lbl in predicted_lbls:
        if lbl.label != 'non_speech':
            hypothesis[Segment(lbl.start_seconds, lbl.end_seconds)] = lbl.label

    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)
    return der


def get_youtube_data(youtube_url, no_cache):

    m = re.search('.*?v=(.{11}).*', youtube_url)
    output = m.group(1)

    h = get_hash(youtube_url)
    tmp_dir = os.path.join(gettempdir(), h)

    if no_cache is True:
        shutil.rmtree(tmp_dir)
        tmp_dir = os.path.join(gettempdir(), h)

    video_abs_path = os.path.join(tmp_dir, "video.mp4")
    audio_abs_path = os.path.join(tmp_dir, "audio.wav")

    if not os.path.isdir(tmp_dir):
        subprocess.check_output(['youtube-dl', '-f', '18', '-o', video_abs_path,
                                 args.youtube_video_url, "--print-json"])

        subprocess.check_call(['ffmpeg', '-y', '-i', video_abs_path, '-ar', '16000', '-ac', '1',
                               audio_abs_path])

    return video_abs_path, audio_abs_path, output, tmp_dir


def get_audio_data(audio_file_path, no_cache):
    output = os.path.splitext(os.path.basename(audio_file_path))[0]

    h = get_hash(audio_file_path)
    tmp_dir = os.path.join(gettempdir(), h)

    if no_cache is True:
        shutil.rmtree(tmp_dir)
        tmp_dir = os.path.join(gettempdir(), h)

    audio_abs_path = os.path.join(tmp_dir, "audio.wav")

    if not os.path.isdir(tmp_dir):
        subprocess.check_call(['ffmpeg', '-y', '-i', audio_file_path, '-ar', '16000', '-ac', '1',
                               audio_abs_path])

    return audio_abs_path, output, tmp_dir


def get_video_data(video_file_path):
    # TODO
    pass


def get_hash(string_to_hash):
    hash_object = hashlib.md5(string_to_hash.encode())
    return hash_object.hexdigest()


def calculate_metrics(output):
    hypothesis_filename = os.path.abspath('%s.fusion.txt' % output)
    print("fusion der: %s" % calculate_der(
        os.path.abspath(args.ground_truth), hypothesis_filename
    ))
    # audio
    hypothesis_filename = os.path.abspath("%s.audio.txt" % output)
    print("audio der: %s" % calculate_der(
        os.path.abspath(args.ground_truth), hypothesis_filename
    ))


def unimodal_analysis(audio_path, output):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration > 600:
        raise Exception("Video duration greater than 10 min (600 sec)")

    generate_audio_based_segmentation(
        audio_path, 15, 20, 128, 128, 0.2,
        os.path.abspath('models/weights.h5'),
        os.path.abspath('models/scaler.pickle'),
        1024, 3, 1024, output,
        os.path.abspath('.'),
        clusters=args.speakers
    )
    return duration


def multimodal_analysis(video_path, output, tmp_dir, duration):

    video_frames_dir = os.path.abspath(os.path.join(tmp_dir, 'video_frames'))

    if not os.path.isdir(video_frames_dir):
        extract_images_from_video(
            os.path.abspath(video_path),
            video_frames_dir
        )

    generate_face_based_segmentation(
        output,
        os.path.abspath(os.path.join(tmp_dir, 'video_frames')),
        os.path.abspath('.'),
        args.speakers,
        os.path.abspath('models/shape_predictor_68_face_landmarks.dat'),
        os.path.abspath('models/dlib_face_recognition_resnet_model_v1.dat'),
        tmp_dir
    )

    # fusion
    mapping_face_to_voice = calculate_fusion(
        output,
        os.path.abspath('.'),
        Util.read_audacity_labels(
            os.path.abspath('%s.audio.txt' % output)
        ),
        Util.read_audacity_labels(
            os.path.abspath('%s.image.txt' % output)
        ),
        duration, step=0.05, neighbours_before_after=40, times_greater=4
    )
    with open(output+".mapping_face_to_voice.json", 'w') as f:
        json.dump(mapping_face_to_voice, f)


def main(args):

    # INPUT TYPE
    if args.youtube_video_url is not None:
        video_path, audio_path, output, tmp_dir = get_youtube_data(args.youtube_video_url, args.no_cache)

    elif args.audio_file is not None:
        if args.analysis_type == "multimodal":
            raise Exception("Multimodal analysis not supported for audio only input")
        audio_path, output, tmp_dir = get_audio_data(args.audio_file, args.no_cache)

    elif args.video_file is not None:
        pass

    # ANALYSIS
    duration = unimodal_analysis(audio_path, output)

    if args.analysis_type == "multimodal":
        multimodal_analysis(video_path, output, tmp_dir, duration)

    # GROUND TRUTH METRICS
    if args.ground_truth is not None:
        calculate_metrics(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", required=True, type=int)
    parser.add_argument("--ground-truth")
    parser.add_argument("--analysis-type", default="unimodal", choices=["unimodal", "multimodal"])
    parser.add_argument("--no-cache", action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-file")
    group.add_argument("--video-file")
    group.add_argument("--youtube-video-url")
    args = parser.parse_args()

    main(args)
