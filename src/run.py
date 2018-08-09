import argparse
import json
import shutil

import subprocess
from tempfile import gettempdir
import os
import librosa
import uuid

from sac.util import Util

from steps.audio_based_segmentation import generate_audio_based_segmentation
from steps.face_based_segmentation import extract_images_from_video, generate_face_based_segmentation
from steps.fusion import calculate_fusion


def main(args):
    tmp_dir = os.path.join(gettempdir(), "%s" % uuid.uuid4())
    try:
        youtube_dl_response = subprocess.check_output(['youtube-dl', '-f', '18', '-o', os.path.join(tmp_dir, '%(id)s'),
                               args.youtube_url_input, "--print-json"])
        youtube_dl_response = json.loads(youtube_dl_response)
        youtube_video_id = youtube_dl_response['id']
        youtube_video_filename = youtube_dl_response['_filename']

        subprocess.check_call(['ffmpeg', '-y', '-i', youtube_video_filename, '-ar', '16000', '-ac', '1',
                               youtube_video_filename+".wav"])
        wav_file = youtube_video_filename+".wav"

        y, sr = librosa.load(wav_file, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration > 600:
            raise Exception("Video duration greater than 10 min (600 sec)")

        # audio based segmentation
        generate_audio_based_segmentation(
            wav_file, 15, 20, 256, 128, 0.2,
            os.path.abspath('models/weights.h5'),
            os.path.abspath('models/scaler.pickle'),
            1024, 3, 1024, youtube_video_id,
            os.path.abspath('.'),
            clusters=args.speakers
        )

        # face based segmentation
        extract_images_from_video(
            os.path.abspath(youtube_video_filename),
            os.path.abspath(os.path.join(tmp_dir, 'video_frames'))
        )
        generate_face_based_segmentation(
            youtube_video_id,
            os.path.abspath(os.path.join(tmp_dir, 'video_frames', youtube_video_id)),
            os.path.abspath('.'),
            args.speakers,
            os.path.abspath('models/shape_predictor_68_face_landmarks.dat'),
            os.path.abspath('models/dlib_face_recognition_resnet_model_v1.dat')
        )

        # fusion
        mapping_face_to_voice = calculate_fusion(
            youtube_video_id,
            os.path.abspath('.'),
            Util.read_audacity_labels(
                os.path.abspath('%s.audio.txt' % youtube_video_id)
            ),
            Util.read_audacity_labels(
                os.path.abspath('%s.image.txt' % youtube_video_id)
            ),
            duration
        )

        shutil.rmtree(tmp_dir)

    except Exception as e:
        shutil.rmtree(tmp_dir)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", required=True, type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-input")
    group.add_argument("--audio-input")
    group.add_argument("--youtube-url-input")
    args = parser.parse_args()

    main(args)