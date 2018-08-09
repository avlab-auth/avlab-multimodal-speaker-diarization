from subprocess import check_call
import os
import dlib
from skimage import io
from glob import glob
import numpy
from sklearn.cluster import KMeans
from sac.util import Util
import pandas as pd
import shutil
import json

FRAMES_PER_STEP = 5

def extract_images_from_video(input_video_file_path, output_dir_path):
    video_id = os.path.basename(input_video_file_path).split('.')[0]
    if os.path.isdir(os.path.join(output_dir_path, video_id)):
        shutil.rmtree(os.path.join(output_dir_path, video_id))
    os.makedirs(os.path.join(output_dir_path, video_id))

    check_call(['ffmpeg', '-i', input_video_file_path, '-qscale:v', '2',
                os.path.join(output_dir_path, video_id, 'img%6d.jpg')])


def generate_face_based_segmentation(youtube_video_id, images_dir, lbls_dir, faces, predictor_path,
                                     face_rec_model_path):

    images_raw = glob(os.path.join(images_dir, "*.jpg"))
    images_raw.sort()
    # images_raw = images_raw[0:100]
    images = [images_raw[i] for i in range(0, len(images_raw), FRAMES_PER_STEP)]
    print(images)
    timestamps = [i * (FRAMES_PER_STEP/25.0) for i in range(0, len(images))]
    print(timestamps)

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    embeddings = []
    embeddings_timestamps = []
    landmarks_parts = []
    landmarks_rect = []

    for frame_no, f in enumerate(images):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            embeddings.append(face_descriptor)
            embeddings_timestamps.append(timestamps[frame_no])
            landmarks_parts.append(shape.parts())
            landmarks_rect.append(shape.rect)

    embeddings = numpy.array(embeddings)
    embeddings_timestamps = numpy.array(embeddings_timestamps)

    print(embeddings.shape)
    print(embeddings_timestamps.shape)

    if len(embeddings) == 0:
        Util.write_audacity_labels([], os.path.join(lbls_dir, youtube_video_id + ".image.txt"))
        return

    kmeans = KMeans(n_clusters=faces)
    kmeans.fit(embeddings)

    predictions = numpy.array(kmeans.labels_.tolist())
    df = pd.DataFrame({"timestamps": embeddings_timestamps.tolist(), "predictions": predictions})

    timestamps = []
    classes = []

    for key, group in df.groupby(['timestamps']):
        timestamps.append(key)
        classes.append(",".join([str(i) for i in sorted(group['predictions'].tolist())]))

    lbls = Util.generate_labels_from_classifications(classes, timestamps)
    json_lbls = []
    for lbl in lbls:
        json_lbls.append({
            "start_seconds": lbl.start_seconds,
            "end_seconds": lbl.end_seconds,
            "label": lbl.label
        })
    with open(os.path.join(lbls_dir, youtube_video_id + ".image.json"), 'w') as outfile:
        json.dump(json_lbls, outfile)

    Util.write_audacity_labels(lbls, os.path.join(lbls_dir, youtube_video_id + ".image.txt"))


# if __name__ == '__main__':
#
#     extract_images_from_video("/Users/nicktgr15/workspace/speaker_diarisation_poc/src/videos/Unamij6z1io.mp4",
#                               "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/video_frames")
#
#     generate_face_based_segmentation(
#         "Unamij6z1io",
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/video_frames/Unamij6z1io",
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/static/lbls/image",
#         4,
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/models/shape_predictor_68_face_landmarks.dat",
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/models/dlib_face_recognition_resnet_model_v1.dat"
#     )