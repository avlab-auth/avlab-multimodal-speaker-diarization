import os
import json
from collections import defaultdict, Counter
import operator
from sac.util import Util


class Item:
    def __init__(self, start_seconds, end_seconds, image_class, audio_class):
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.image_class = image_class
        self.audio_class = audio_class

    def __repr__(self):
        return "[%s %s %s %s]" % (self.start_seconds, self.end_seconds, self.image_class, self.audio_class)


def create_pairs(audio_lbls, image_lbls, duration, step):

    pairs = []
    timestamps = []

    i = 0
    while i < duration:

        image_lbl_code = None
        audio_lbl_code = None

        for image_lbl in image_lbls:

            if image_lbl.end_seconds >= i:
                # print("%s >= %s" % (image_lbl.start_seconds, i) )
                image_lbl_code = image_lbl.label
                break

        for audio_lbl in audio_lbls:
            if audio_lbl.end_seconds >= i:
                audio_lbl_code = audio_lbl.label
                break

        pairs.append(
            Item(i, i+step, image_lbl_code, audio_lbl_code)
        )
        timestamps.append(i)

        i += step

    return pairs, timestamps


def apply_mapping_to_pairs(pairs, mapping_face_to_voice):
    for p in pairs:
        if p.image_class is not None:
            classes = p.image_class.split(",")
            if len(classes) == 1:
                c = classes[0]
                p.image_class = mapping_face_to_voice[c]

    return pairs


def detect_face_voice_mapping(pairs):

    mapping_face_to_voice = {}

    ## find correlation
    correlation_pairs = []
    for p in pairs:
        if p.image_class is not None and p.audio_class is not None and p.audio_class != 'non_speech':
            classes = p.image_class.split(",")
            if len(classes) == 1:
                c = classes[0]
                if c != "non_speech":
                    # print(p)
                    correlation_pairs.append(p)

    # print(correlation_pairs)

    classes_set = list(set([i.image_class for i in correlation_pairs]))
    # print(classes_set)
    for c in classes_set:
        classes_probability = defaultdict(int)
        for p in correlation_pairs:
            if p.image_class == c:
                classes_probability[p.audio_class] += 1

        mapping_face_to_voice[c] = max(classes_probability.items(), key=operator.itemgetter(1))[0]

    # print(mapping_face_to_voice)

    return mapping_face_to_voice


def calculate_fusion(youtube_video_id, lbls_dir, audio_lbls, image_lbls, duration, step=0.1): # 100ms

    pairs, timestamps = create_pairs(audio_lbls, image_lbls, duration, step)
    mapping_face_to_voice = detect_face_voice_mapping(pairs)
    print(mapping_face_to_voice)
    pairs = apply_mapping_to_pairs(pairs, mapping_face_to_voice)
    print(pairs)

    for k, pair in enumerate(pairs):
        if pair.image_class is None:
            # when image is None
            continue
        classes = pair.image_class.split(",")
        if len(classes) == 1 and pair.audio_class != 'non_speech':
            if pair.image_class != pair.audio_class:
                # print("%s != %s" % (pair.image_class, pair.audio_class))
                nearest_neighbour_class = find_nearest_neighbours_class(k, pairs)
                pair.audio_class = nearest_neighbour_class

    print(pairs)

    lbls = Util.generate_labels_from_classifications([p.audio_class for p in pairs], timestamps)
    lbls = filter(lambda x: x.label is not None, lbls)

    json_lbls = []
    for lbl in lbls:
        json_lbls.append({
            "start_seconds": lbl.start_seconds,
            "end_seconds": lbl.end_seconds,
            "label": lbl.label
        })
    with open(os.path.join(lbls_dir, youtube_video_id + ".fusion.json"), 'w') as outfile:
        json.dump(json_lbls, outfile)

    Util.write_audacity_labels(lbls, os.path.join(lbls_dir, youtube_video_id + ".fusion.txt"))
    return mapping_face_to_voice

def find_nearest_neighbours_class(position, pairs, neighbours_before_after=6):

    neighbour_image_classes = [
        p.image_class for p in pairs[position-neighbours_before_after:position+neighbours_before_after]
    ]

    neighbour_image_classes = list(filter(lambda x: x != 'non_speech', neighbour_image_classes))

    if len(neighbour_image_classes) == 0:
        return pairs[position].audio_class

    neighbour_image_classes = Counter(neighbour_image_classes)

    most_popular_class = max(neighbour_image_classes.items(), key=operator.itemgetter(1))[0]
    classes = most_popular_class.split(",")
    if len(classes) == 1:
        return most_popular_class
    else:
        return pairs[position].audio_class


# if __name__ == '__main__':
#
#     audio_lbls = Util.read_audacity_labels(
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/static/lbls/audio/Unamij6z1io.txt")
#     image_lbls = Util.read_audacity_labels(
#         "/Users/nicktgr15/workspace/speaker_diarisation_poc/src/static/lbls/image/Unamij6z1io.txt")
#
#     calculate_fusion(audio_lbls, image_lbls, 443.129625)
