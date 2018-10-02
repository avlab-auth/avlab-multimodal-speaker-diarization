import os
from sac.util import Util

from run import calculate_der
from steps.fusion import calculate_fusion


def eval_der(youtube_video_id, step, neighbours_before_after, times_greater):
    duration = 443.0

    mapping_face_to_voice = calculate_fusion(
        youtube_video_id,
        os.path.abspath('.'),
        Util.read_audacity_labels(
            os.path.abspath('%s.wav.audio.txt' % youtube_video_id)
        ),
        Util.read_audacity_labels(
            os.path.abspath('%s.wav.image.txt' % youtube_video_id)
        ),
        duration, step=step, neighbours_before_after=neighbours_before_after, times_greater=times_greater
    )

    ground_truth = "sequence_5_lbls_fixed.txt"

    # fusion
    hypothesis_filename = os.path.abspath('%s.fusion.txt' % youtube_video_id)
    fusion_der = calculate_der(
        os.path.abspath(ground_truth), hypothesis_filename
    )
    # print("fusion der: %s" % )
    # audio
    hypothesis_filename = os.path.abspath("%s.wav.audio.txt" % youtube_video_id)
    # print("audio der: %s" % )
    audio_der = calculate_der(
        os.path.abspath(ground_truth), hypothesis_filename
    )
    return audio_der, fusion_der

# videos = [
#     "0KUoPilGyHE",
#     '4Z3Pwygta58'
#     "lr2a17Iu9qM"
# ]
step = [0.05, 0.1, 0.2]
window_size = [0.2, 0.5, 1.0, 2.0, 5.0]
times_greater = [1, 4, 8]

with open("fusion_results.txt", "w") as f:
    for s in step:
        for w in window_size:
            for t in times_greater:
                # print("step=%s window_size=%s times_greater=%s" % (s, w, t))
                # f.write("%s %s %s %s\n" % (s, w, t, int(w/s)))
                audio_der, fusion_der = eval_der("0KUoPilGyHE", s, int(w/s), t)
                audio_der2, fusion_der2 = eval_der("lr2a17Iu9qM", s, int(w/s), t)
                f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (s, w, t, int(w/s), fusion_der, fusion_der2))
