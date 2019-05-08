import itertools
import cv2
import os
import shutil
import json
import kenlm
import base64
import numpy as np
import tensorflow as tf
import networkx as nx
from tensorflow.python import keras
from time import time
from math import inf
from evaluation import evaluate_word_accuracy, mrr, correct_transcr_count
from absl import app, flags

"""
HELPER FUNCTIONS
"""


def _map_class_to_chars(char_class, all_mappings=None):
    if all_mappings == None:
        all_mappings = {
                '0_b_stroke': "b", # -
                '0_con': "con",
                '0_curl': "us",
                '0_d_stroke': "d", # -
                '0_l_stroke': "l", # -
                '0_nt': "et",
                '0_per': "per",
                '0_pro': "pro",
                '0_qui': "qui",
                '0_rum': "rum",
                '0_semicolon': ';',
                'a': 'a',
                'b': 'b',
                'c': 'c',
                'd': 'd',
                'd_1': 'd',
                'd_2': 'd',
                'e': 'e',
                'f': 'f',
                'g': 'g',
                'h': 'h',
                'i': 'i',
                'l': 'l',
                'm': 'm',
                'n': 'n',
                'o': 'o',
                'p': 'p',
                'q': 'q',
                'r': 'r',
                's_1': "s",
                's_2': "s",
                's_3': "s",
                's_alta': "s",
                's_ending': "s",
                't': 't',
                'u': 'u',
                'x': 'x'
            }
    return all_mappings[char_class]


def _dst_suffix():
    dst = ''
    for k, v in sorted(flags.FLAGS.flag_values_dict().items()):
        if k not in ['h', 'help', 'helpfull', 'helpshort'] and '_dir' not in k:
            dst += '.' + k + '=' + str(v)
    return dst


def _compute_bbx(stats):
    x1 = min([x for x, _, _, _, _ in stats[1:]])
    y1 = min([y for _, y, _, _, _ in stats[1:]])
    x2 = max([x + w for x, _, w, _, _ in stats[1:]])
    y2 = max([y + h for _, y, _, h, _ in stats[1:]])

    return x1, y1, x2, y2


def _compute_segments_and_centroids(word_img):
    hist = cv2.calcHist([word_img], [0, 1, 2], None, [256] * 3, [0, 256] * 3)
    colors = [
        np.array([b, g, r]) for b, g, r in np.argwhere(hist > 0)
        if (b, g, r) != (255, 255, 255)
    ]
    centroids = []
    segments = []

    for color in colors:
        mask = cv2.inRange(word_img, lowerb=color, upperb=color)
        _, _, stats, ctds = cv2.connectedComponentsWithStats(mask)

        x1, y1, x2, y2 = _compute_bbx(stats)
        w = x2 - x1
        h = y2 - y1

        if w * h > 9:
            centroids.append(tuple(ctds[1]))
            segments.append(mask)

    return segments, centroids


def _make_sample(segments, sample_shape=56):
    word_mask = np.zeros(segments[0].shape, dtype='uint8')

    for s in segments:
        word_mask = cv2.bitwise_or(word_mask, s)

    _, _, stats, _ = cv2.connectedComponentsWithStats(word_mask)

    x1 = min([x for x, _, _, _, _ in stats[1:]])
    y1 = min([y for _, y, _, _, _ in stats[1:]])
    x2 = max([x + w for x, _, w, _, _ in stats[1:]])
    y2 = max([y + h for _, y, _, h, _ in stats[1:]])
    w = x2 - x1
    h = y2 - y1

    bbx_crop = word_mask[y1:y2, x1:x2]

    top = max((sample_shape - h) // 2, 0)
    bottom = max(sample_shape - (h + top), 0)
    left = max((sample_shape - w) // 2, 0)
    right = max(sample_shape - (w + left), 0)

    sample_img = cv2.copyMakeBorder(
        bbx_crop,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    if sample_img.shape != (sample_shape, sample_shape):
        sample_img = cv2.resize(
            sample_img,
            (sample_shape, sample_shape),
            interpolation=cv2.INTER_NEAREST
        )

    return sample_img.reshape((sample_shape,sample_shape,1))


"""
LATTICE TRAVERSAL
"""


def multidag_dfs_kenlm(graph, start, end, path_so_far=[], threshold=-inf):
    if len(graph.out_edges(start)) < 1:  # start == end:
        textgram = '#'
        score = 0.0
        for _, _, prev_data in path_so_far:
            textgram += _map_class_to_chars(prev_data['transcription'])
            score += np.log(prev_data['weight'])

        if textgram[-2:] == 'b;':
            textgram = textgram[:-2] + 'bus'
        if textgram[-2:] == 'q;':
            textgram = textgram[:-2] + 'que'

        textgram += '#'

        if model_LM:
            score = model_LM.score(' '.join(list(textgram)), bos=False, eos=False)

        if score > threshold:
            yield path_so_far, score
    else:
        for u, v, data in graph.out_edges(start, data=True):
            textgram = '#'
            score = 0.0
            for _, _, prev_data in path_so_far:
                textgram += _map_class_to_chars(prev_data['transcription'])
                score += np.log(prev_data['weight'])

            if model_LM:
                score = model_LM.score(' '.join(list(textgram)), bos=False, eos=False)

            if score > threshold:
                for path in multidag_dfs_kenlm(
                        graph, v, end, path_so_far + [(u, v, data)], threshold):
                    yield path

def custom_loss(y_true, y_pred):
    # Compute the binary loss (is char / not char)
    loss_nochar = keras.losses.binary_crossentropy(y_true[:, 0:1], y_pred[:, 0:1], from_logits=True)

    # These are the locations of chars inside the current batch
    idx_chars = tf.where(1 - y_true[:, 0])[:, 0]

    # Compute the cross-entropy loss only for chars
    loss_chars = keras.losses.categorical_crossentropy(
            tf.gather(y_true[:, 1:], idx_chars),
            tf.gather(y_pred[:, 1:], idx_chars), from_logits=True)

    # Sum the two losses (weighted)
    return tf.reduce_sum(loss_nochar) + tf.reduce_sum(loss_chars)*5

def char_accuracy(y_true, y_pred):
    # Returns the accuracy of recognition for the characters
    idx_chars = tf.where(1 - y_true[:, 0])[:, 0]
    return keras.metrics.categorical_accuracy(
            tf.gather(y_true[:, 1:], idx_chars),
            tf.gather(y_pred[:, 1:], idx_chars))

def nochar_accuracy(y_true, y_pred):
    # Returns the accuracy of recognizing chars vs. no-chars
    return keras.metrics.binary_accuracy(y_true[:, 0:1], y_pred[:, 0:1])

"""
APP MAIN
"""

import matplotlib.pyplot as plt

def main(unused_argv):
    # set destination folder names
    # sfx = _dst_suffix()
    tsc_dir, dag_dir, eval_fnm = 'tsc', 'dag', 'eval'

    # make necessary directories
    if os.path.isdir(tsc_dir):
        shutil.rmtree(tsc_dir)
    os.mkdir(tsc_dir)

    if os.path.isdir(dag_dir):
        shutil.rmtree(dag_dir)
    os.mkdir(dag_dir)

    all_classes = [ # '1_not_char', '0_con',
        '0_b_stroke', '_curl', '0_d_stroke', '0_l_stroke',
        '0_nt',       '0_per',  '0_pro', '0_qui',      '0_rum',     '0_semicolon',
        'a', 'b', 'c', 'd', 'e',      'f',        'g', 'h', 'i', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's_alta', 's_ending', 't', 'u', 'x'
    ]
    clrs = cv2.imread('palette2.png')[0]


    icr_classifier = keras.models.load_model(
        flags.FLAGS.ocr_dir,
        custom_objects={
            'custom_loss':custom_loss,
            'char_accuracy':char_accuracy,
            'nochar_accuracy':nochar_accuracy
        }
    )

    # load the Language Model
    global model_LM
    if flags.FLAGS.n_gram == 0:
        model_LM = None
    else:
        model_LM = kenlm.Model(
            os.path.join(
                flags.FLAGS.lm_dir,
                'corpus_%dgram.arpa' % flags.FLAGS.n_gram
            )
        )

    start_time = time()
    words = os.listdir(flags.FLAGS.word_dir)

    for word in words:
        word_img = cv2.imread(os.path.join(flags.FLAGS.word_dir, word))
        _, word_img_bin = cv2.threshold(cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)
        _, _, stats, _ = cv2.connectedComponentsWithStats(word_img_bin)
        x1, y1, x2, y2 = _compute_bbx(stats)
        word_img_crop = word_img[y1:y2,x1:x2]

        segments, centroids = _compute_segments_and_centroids(word_img_crop)

        sorted_segments, sorted_centroids = zip(
            *sorted(
                zip(segments, centroids),
                key=lambda x: (x[1][0], x[1][1])
            )
        )

        # generate all possible segment combinations
        segments_and_centroids_combinations = [
            (sorted_segments[sorted_centroids.index((x1, y1)):sorted_centroids.index((x2, y2)) + 1],
             sorted_centroids[sorted_centroids.index((x1, y1)):sorted_centroids.index((x2, y2)) + 1])
            for (x1, y1), (x2, y2) in itertools.combinations_with_replacement(sorted_centroids, 2)
            if (x2 - x1) < 25
        ]

        grouped_segments, centroid_ids = zip(*segments_and_centroids_combinations)

        # create samples for prediction
        X_test = np.array([_make_sample(s) for s in grouped_segments], dtype='float32') / 255

        logit_preds = icr_classifier.predict(X_test)

        char_preds = tf.nn.softmax(logit_preds[:, 1:], axis=1).numpy()
        notchar_preds = tf.nn.sigmoid(logit_preds[:, :1]).numpy()

        # keep top3 classification results and notchar score
        top3_preds = [
            (notchar_preds[ix], np.argsort(p)[::-1][:3]) for ix, p in enumerate(char_preds)
        ]

        # filter segment combinations according to classification
        filtered_combinations = []

        for i, cc in enumerate(centroid_ids):
            prob_dist = 0.0
            potential_edges = []
            nchar_prob, top3_pred = top3_preds[i]

            if nchar_prob < flags.FLAGS.notchar_thr:
                for pred_ix in top3_pred:
                    prob_dist += char_preds[i][pred_ix]
                    if all_classes[pred_ix] == '_curl':
                        potential_edges.append((
                            cc,
                            '0_curl',
                            char_preds[i][pred_ix],
                            grouped_segments[i]
                        ))
                        potential_edges.append((
                            cc,
                            '0_con',
                            char_preds[i][pred_ix],
                            grouped_segments[i]
                        ))
                    else:
                        potential_edges.append((
                            cc,
                            all_classes[pred_ix],
                            char_preds[i][pred_ix],
                            grouped_segments[i]
                        ))

                    if prob_dist > flags.FLAGS.pdist_thr:
                        break

            for pe in potential_edges:
                if pe[2] > flags.FLAGS.char_thr:
                    filtered_combinations.append(pe)

        print(
            '{}:\nKept: {} out of {} potential edges ({:.2f}%)'.format(
                word,
                len(filtered_combinations),
                len(centroid_ids) * 3,
                (len(filtered_combinations) / (len(centroid_ids) * 3)) * 100
            )
        )

        # creation of the word lattice: segment combinations represent
        # edges. Nodes are segments consumed up to a certain point.
        lattice = nx.MultiDiGraph()

        filtered_combinations = sorted(filtered_combinations, key=lambda x: x[0][0])
        potential_nodes = [
            set(sorted_centroids[:i])
            for i in range(len(sorted_centroids) + 1)
        ]

        for u, v in itertools.combinations(potential_nodes, 2):
            edge = v - u
            for fc, char, prob, segment in filtered_combinations:
                if set(fc) == edge:
                    s_mask = np.zeros(segment[0].shape, dtype='uint8')
                    for s in segment:
                        s_mask = cv2.bitwise_or(s_mask, s)

                    lattice.add_edge(
                        tuple(sorted(u)),
                        tuple(sorted(v)),
                        transcription=char,
                        weight=prob,
                        image=s_mask
                    )

        print("nodes: {},\tedges: {}\n".format(len(lattice.nodes()), len(lattice.edges())))

        # save a .js file with the lattice structure
        dict_nodes = [
            str([sorted_centroids.index(c) for c in node])
            for node in lattice.nodes()
        ]

        lattice_dict = {
            'nodes': sorted(dict_nodes, reverse=True),
            'edges': []
        }

        graph_as_dict = nx.to_dict_of_dicts(lattice)

        for src_node in graph_as_dict:
            for dst_node in graph_as_dict[src_node]:
                labels = ''
                for edge_ix in graph_as_dict[src_node][dst_node]:
                    labels += '%s: %.4f\n' % (graph_as_dict[src_node][dst_node][edge_ix]['transcription'],
                                              graph_as_dict[src_node][dst_node][edge_ix]['weight'])
                dict_src_node = str([sorted_centroids.index(c) for c in src_node])
                dict_dst_node = str([sorted_centroids.index(c) for c in dst_node])
                lattice_dict['edges'].append((dict_src_node, dict_dst_node, labels))

        with open(os.path.join(dag_dir, word.replace('/', '_').split('.')[0] + '.js'), 'w') as f:
            f.write('var graph = ' + json.dumps(lattice_dict, indent=2))

        # Transcription generation:
        if len(lattice.nodes()) == 0 or len(lattice.edges()) == 0:
            all_paths = []
        else:
            start = list(lattice.nodes())[np.argmin([len(n) for n in lattice.nodes()])]
            end = list(lattice.nodes())[np.argmax([len(n) for n in lattice.nodes()])]
            all_paths = multidag_dfs_kenlm(lattice, start, end, threshold=flags.FLAGS.lm_thr)

        transcriptions = []

        for path, prob in all_paths:
            transcript = ''
            w_segmentation = np.zeros(path[0][2]['image'].shape + (3,), dtype='uint8')

            for c_ix, (u, v, data) in enumerate(path):
                transcript += _map_class_to_chars(data['transcription'])

                data_img = cv2.cvtColor(data['image'], cv2.COLOR_GRAY2RGB)
                data_img = np.where(data_img == [255, 255, 255],
                                    clrs[c_ix % len(clrs)],
                                    [0, 0, 0]
                                    )

                w_segmentation = w_segmentation + data_img

            if transcript[-2:] == 'b;':
                transcript = transcript[:-2] + 'bus'
            if transcript[-2:] == 'q;':
                transcript = transcript[:-2] + 'que'
            if len(transcript) * 27 >= word_img_crop.shape[1]:
                # encode image as string
                _, buffer = cv2.imencode('.png', w_segmentation)
                png_as_str = base64.b64encode(buffer)
                transcriptions.append((prob, transcript, png_as_str))

        transcriptions = sorted(set(transcriptions), reverse=True)

        with open(os.path.join(tsc_dir, word.replace('/', '_').split('.')[0] + '.txt'), 'w') as f:
            for t in transcriptions:
                f.write(str(t) + '\n')

    end_time = time()

    # evaluation
    elapsed = end_time - start_time
    print('time elapsed:', elapsed)
    print(tsc_dir)

    MRR = mrr(tsc_dir, flags.FLAGS.gt_dir)
    correct_count = correct_transcr_count(tsc_dir, flags.FLAGS.gt_dir)
    evaluation = evaluate_word_accuracy(tsc_dir, flags.FLAGS.gt_dir)
    evaluation['mrr'] = MRR
    evaluation['time'] = elapsed
    evaluation['correct_overall'] = correct_count

    print(json.dumps(evaluation, indent=2))
    json.dump(evaluation, open(eval_fnm+'.json','w'), indent=2)


if __name__ == '__main__':
    flags.DEFINE_integer('n_gram', 6, 'Language Model order')
    flags.DEFINE_float('lm_thr', -16.0, 'LM probability pruning threshold')
    flags.DEFINE_float('char_thr', 0.1, 'character probability pruning threshold')
    flags.DEFINE_float('notchar_thr', 0.7, 'not character probability pruning threshold')
    flags.DEFINE_float('pdist_thr', 0.8, 'probability distribution pruning threshold')
    flags.DEFINE_integer('alt_top_n', 0, 'top n transcriptions to submit to alternative generation')
    flags.DEFINE_string('lm_dir', 'lm_model', 'Language model folder')
    flags.DEFINE_string('ocr_dir', 'ocr_model/new_multiout.hdf5', 'character classifier model folder')
    flags.DEFINE_string('word_dir', 'word_imgs/best_segmentation_riccardo/050v_573_280_1344_1805/images', 'word image source folder')
    flags.DEFINE_string('gt_dir', 'word_imgs/best_segmentation_riccardo/050v_573_280_1344_1805/transcriptions', 'ground truth source folder')

    app.run(main)
