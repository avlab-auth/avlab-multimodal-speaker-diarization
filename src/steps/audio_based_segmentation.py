import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sac.util import Util
import numpy

from random import seed

import librosa
import numpy as np
from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization, Conv1D, \
    MaxPooling1D, Lambda, LSTM, regularizers
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from librosa.feature import melspectrogram, mfcc
from sklearn.cluster import KMeans
import pickle
import os
from sac.methods.vad import Vad
from keras.layers.wrappers import Bidirectional

seed(12345)
np.random.seed(42)

MARGIN = 1.0


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # same = 1
    # different = 0
    # 1, 1.5<2.0 => 1
    # 0, 2.25 > 2.0 => 0
    return K.mean(K.equal(y_true, K.cast(y_pred < MARGIN / 2.0, y_true.dtype)))


def eucl_dist_output_shape(shapes):
    # because it's going to be a vector of distances :, 128 we need to transform it to :,1
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    # same=1 different=0
    # when same => minimize distance => K.square(d)
    # when different => get the max(margin-d, 0) .. this is optimal (i.E. 0) when the distance is smaller or equal to the margin
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(MARGIN - d, 0)))


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def get_lstm_siamese(input_shape, feature_vector_size, lstm_nodes, dropout):
    def create_base_network(input_shape):
        model = Sequential()

        model.add(Bidirectional(LSTM(lstm_nodes, return_sequences=True, dropout=dropout), input_shape=input_shape))
        model.add(Bidirectional(LSTM(lstm_nodes, return_sequences=False, dropout=dropout)))

        model.add(Dense(feature_vector_size, activation='relu'))
        model.add(Dense(feature_vector_size, activation='relu'))

        print(model.summary())

        return model

    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    print(model.summary())

    optimiser = Adam()

    model.compile(loss=contrastive_loss, optimizer=optimiser, metrics=[accuracy])

    intermediate_layer_model = Model(input=base_network.layers[0].input, output=base_network.layers[-1].output)

    return model, intermediate_layer_model


class FeatExtractorMFCC(object):
    def __init__(self, window_size, hop_size, w, sr, mfcc_no, step=None, vad=False):
        self.window_size = window_size
        self.hop_size = hop_size
        self.mfcc_no = mfcc_no
        self.w = w
        self.sr = sr
        if step is None:
            self.step = self.w
        else:
            self.step = step

    def extract(self, audio_file):
        y, sr = librosa.load(audio_file, sr=self.sr)
        D = mfcc(y, sr=self.sr, n_mfcc=self.mfcc_no + 2, n_fft=self.window_size, hop_length=self.hop_size)
        D = D[2:, :]

        feats = []
        timestamps = []
        current_time = 0
        for i in range(0, D.shape[1], self.step):
            d = D[:, i:i + self.w]

            d = d.transpose()

            if d.shape[0] == self.w:
                feats.append(d)
                timestamps.append(current_time)
                current_time = current_time + self.step * self.hop_size / float(self.sr)

        feats = np.array(feats)
        return feats, timestamps


def generate_audio_based_segmentation(audio_file, w, h, embedding_size, lstm_nodes, dropout, weights_filename,
                                      scaler_filename, window_size, step, hop_size, youtube_video_id, lbls_dir,
                                      clusters=4, sr=16000):
    vad = Vad()
    vad_lbls = vad.detect_voice_segments(audio_file)
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)

    model, intermediate = get_lstm_siamese((w, h), embedding_size, lstm_nodes, dropout)
    model.load_weights(weights_filename)
    feature_extractor = FeatExtractorMFCC(window_size, hop_size, w, sr, h, step=step)
    X, timestamps = feature_extractor.extract(audio_file)
    timestamps = numpy.array(timestamps)
    window = timestamps[1] - timestamps[0]

    frame_predictions = []

    for k, timestamp in enumerate(timestamps):
        found = False
        for lbl in vad_lbls:
            if lbl.start_seconds <= timestamp <= lbl.end_seconds - window:  # need the window end to fit in the label
                frame_predictions.append(lbl.label)
                found = True
                break
        if not found:
            frame_predictions.append('non_speech')

    frame_predictions = numpy.array(frame_predictions)
    print(frame_predictions.shape)
    print(timestamps.shape)

    speech_indices = numpy.where(frame_predictions == 'speech')

    X_speech = X[speech_indices]

    X = X_speech.reshape((X_speech.shape[0] * w, h))
    X = scaler.transform(X)
    X = X.reshape(-1, w, h)

    original_embeddings = intermediate.predict(X)

    clustering_algorithm = KMeans(n_clusters=clusters)

    reducted_embeddings = original_embeddings
    predictions = clustering_algorithm.fit_predict(reducted_embeddings)

    for k, speech_index in enumerate(speech_indices[0]):
        frame_predictions[speech_index] = predictions[k]

    lbls = Util.generate_labels_from_classifications(frame_predictions, timestamps)

    Util.write_audacity_labels(lbls, os.path.join(lbls_dir, youtube_video_id + ".audio.txt"))
