import itertools
import cv2
import pathlib
# import os
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


def _compute_segments_and_centroids(word_img, bg_color=(255, 255, 255)):
    hist = cv2.calcHist([word_img], [0, 1, 2], None, [256] * 3, [0, 256] * 3)
    colors = [
        np.array([b, g, r]) for b, g, r in np.argwhere(hist > 0)
        if (b, g, r) != bg_color
    ]
    centroids = []
    segments = []

    for color in colors:
        mask = cv2.inRange(word_img, lowerb=color, upperb=color)
        _, _, stats, ctds = cv2.connectedComponentsWithStats(mask)

        x1, y1, x2, y2 = _compute_bbx(stats)
        w = x2 - x1
        h = y2 - y1

        if w * h > 0:
            cx, cy = ctds[1]
            centroids.append(
                (int(np.rint(cx)), int(np.rint(cy)))
            )
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


IX2TSC = {
    0:'b', # with stroke
    1:'con', # con?
    2:'d', # with stroke
    3:'l', # with stroke
    4:'et',
    5:'per',
    6:'pro',
    7:'qui',
    8:'rum',
    9:';',
    10:'a',
    11:'b',
    12:'c',
    13:'d',
    14:'e',
    15:'f',
    16:'g',
    17:'h',
    18:'i',
    19:'l',
    20:'m',
    21:'n',
    22:'o',
    23:'p',
    24:'q',
    25:'r',
    26:'s',
    27:'s',
    28:'t',
    29:'u',
    30:'x'
}

from collections import OrderedDict

def _edges_from(G, visited, width, lm_th):
    multiedges = []
    u, _, _ = visited[-1]

    prob = np.sum(score for _, _, score in visited)

    if prob > lm_th:
        for v in G.successors(u):
            ocr_predictions = np.log10(G.get_edge_data(u, v)['preds'])
            tsc = ''.join(c for _, c, _ in visited)

            tsc_score = model_LM.score(
                ' '.join(
                    list(tsc.replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que'))
                ),
                bos=False,
                eos=False
            )

            if len(G[v]) < 1:
                lm_predictions = np.array([
                    model_LM.score(
                        ' '.join(list(
                            (tsc + IX2TSC[i] + '#').replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                        )),
                        bos=False,
                        eos=False
                    ) - tsc_score for i in IX2TSC
                ])
            else:
                lm_predictions = np.array([
                    model_LM.score(
                        ' '.join(list(
                            (tsc + IX2TSC[i]).replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                        )),
                        bos=False,
                        eos=False
                    ) - tsc_score for i in IX2TSC
                ])

            predictions = np.sum(
                [ocr_predictions, lm_predictions],
                axis=0
            )
            multiedges += [(v, IX2TSC[tsc_ix], pred) for tsc_ix, pred in enumerate(predictions)]

        multiedges = sorted(multiedges, key=lambda x: x[-1], reverse=True)[:width]

    return iter(multiedges)


def paths_beam(G, source, targets, width, lm_th):
    visited = OrderedDict.fromkeys([(source, '#', 0.0)])
    stack = [_edges_from(G, list(visited), width, lm_th)]

    while stack:
        children = stack[-1]
        child = next(children, None)

        if child is None:
            stack.pop()
            visited.popitem()
        else:
            if child in visited:
                continue
            if child[0] in targets:
                yield list(visited) + [child]

            visited[child] = None
            visited_nodes = {n for n, _, _ in visited.keys()}

            if targets - visited_nodes:  # expand stack until find all targets
                stack.append(_edges_from(G, list(visited), width, lm_th))
            else:
                visited.popitem()  # maybe other ways to child



"""
APP MAIN
"""

import matplotlib.pyplot as plt

def main(unused_argv):
    src_dir = pathlib.Path(flags.FLAGS.src_dir)
    dst_dir = pathlib.Path(flags.FLAGS.dst_dir)

    # all_classes = [ # '1_not_char', '0_con',
    #     '0_b_stroke', '_curl', '0_d_stroke', '0_l_stroke',
    #     '0_nt',       '0_per',  '0_pro', '0_qui',      '0_rum',     '0_semicolon',
    #     'a', 'b', 'c', 'd', 'e',      'f',        'g', 'h', 'i', 'l', 'm', 'n',
    #     'o', 'p', 'q', 'r', 's_alta', 's_ending', 't', 'u', 'x'
    # ]
    # clrs = cv2.imread('palette2.png')[0]

    icr_classifier = keras.models.load_model(
        flags.FLAGS.ocr,
        custom_objects={
            'custom_loss':custom_loss,
            'char_accuracy':char_accuracy,
            'nochar_accuracy':nochar_accuracy
        }
    )

    # load the Language Model
    global model_LM
    if flags.FLAGS.lm == 'noLM':
        model_LM = None
    else:
        model_LM = kenlm.Model(flags.FLAGS.lm)

    for page_path in src_dir.glob('*'):
        start_time = time()

        tsc_dir = dst_dir / page_path.stem
        tsc_dir.mkdir(parents=True, exist_ok=True)

        for word_path in page_path.glob(flags.FLAGS.img_dir+'/*'):
            word_img = cv2.imread(str(word_path))
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

            # filter segment combinations according to classification
            filtered_combinations = []
            for i, cc in enumerate(centroid_ids):
                if notchar_preds[i] <= flags.FLAGS.notchar_thr:
                    s_mask = np.zeros(grouped_segments[i][0].shape, dtype='uint8')
                    for s in grouped_segments[i]:
                        s_mask = cv2.bitwise_or(s_mask, s)
                    filtered_combinations.append((cc, char_preds[i], s_mask))

            print(
                '{}:\nKept: {} out of {} potential edges ({:.2f}%)'.format(
                    str(word_path),
                    len(filtered_combinations),
                    len(centroid_ids),
                    (len(filtered_combinations)/len(centroid_ids)) * 100
                )
            )

            # creation of the word lattice: segment combinations represent
            # edges. Nodes are segments consumed up to a certain point.
            lattice = nx.DiGraph()

            filtered_combinations = sorted(filtered_combinations, key=lambda x: x[0][0])
            filtered_centroids =  sorted({c for ctds, _, _ in filtered_combinations for c in ctds})

            nodes = [
                set(filtered_centroids[:i])
                for i in range(len(filtered_centroids) + 1)
            ]
            edges = [
                set(ctds)
                for ctds, _, _ in filtered_combinations
            ]

            for u, v in itertools.combinations(nodes, 2):
                if v - u in edges:
                    _, preds, sgmt_img = filtered_combinations[edges.index(v-u)]
                    lattice.add_edge(
                        tuple(sorted(u)),
                        tuple(sorted(v)),
                        preds=preds,
                        image=sgmt_img
                    )

            print("nodes: {},\tedges: {}\n".format(len(lattice.nodes()), len(lattice.edges())))

            # # save a .js file with the lattice structure
            # dict_nodes = [
            #     str([sorted_centroids.index(c) for c in node])
            #     for node in lattice.nodes()
            # ]
            #
            # lattice_dict = {
            #     'nodes': sorted(dict_nodes, reverse=True),
            #     'edges': []
            # }
            #
            # graph_as_dict = nx.to_dict_of_dicts(lattice)
            #
            # for src_node in graph_as_dict:
            #     for dst_node in graph_as_dict[src_node]:
            #         labels = ''
            #         for edge_ix in graph_as_dict[src_node][dst_node]:
            #             labels += '%s: %.4f\n' % (graph_as_dict[src_node][dst_node][edge_ix]['transcription'],
            #                                       graph_as_dict[src_node][dst_node][edge_ix]['weight'])
            #         dict_src_node = str([sorted_centroids.index(c) for c in src_node])
            #         dict_dst_node = str([sorted_centroids.index(c) for c in dst_node])
            #         lattice_dict['edges'].append((dict_src_node, dict_dst_node, labels))
            #
            # with open(os.path.join(dag_dir, word.replace('/', '_').split('.')[0] + '.js'), 'w') as f:
            #     f.write('var graph = ' + json.dumps(lattice_dict, indent=2))

            # Transcription generation:

            transcriptions = set()

            if len(lattice.nodes()) > 0:
                for path in paths_beam(
                        lattice,
                        source=sorted(lattice.nodes())[0],
                        targets={sorted(lattice.nodes())[-1]},
                        width=3,
                        lm_th=flags.FLAGS.lm_thr):
                    transcript = ''
                    prob = 0.0
                    for _, char, score in path[1:]:
                        transcript += char
                        prob += score
                    transcript = transcript.replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                    transcript = transcript.strip(';')
                    transcriptions.add((prob, transcript, str(path)))

            transcriptions = sorted(transcriptions, reverse=True)

            transcriptions_filtered = []

            for line in transcriptions:
                if len(transcriptions_filtered) > 0:
                    _, tfil, _ = zip(*transcriptions_filtered)
                else:
                    tfil = []
                if line[1] not in tfil:
                    transcriptions_filtered.append(line)

            dst_tsc = tsc_dir / (word_path.stem  + '.txt')
            with open(str(dst_tsc), 'w') as f:
                for t in transcriptions_filtered:
                    f.write(str(t) + '\n')

        end_time = time()

        # evaluation
        elapsed = end_time - start_time
        print('time elapsed:', elapsed)

        MRR = mrr(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
        correct_count = correct_transcr_count(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
        evaluation = evaluate_word_accuracy(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
        evaluation['mrr'] = MRR
        evaluation['time'] = elapsed
        evaluation['correct_overall'] = correct_count

        print(json.dumps(evaluation, indent=2))
        json.dump(evaluation, open(str(tsc_dir)+'_eval.json','w'), indent=2)

# con moltiplicazione:
# 050v 0.34 mrr: 0.17
# 100r 0.39 mrr: 0.25
# 150v 0.29 mrr: 0.12
# 200r 0.45 mrr: 0.27



# su tutta la pagina:
# 050v: 0.80 mrr: 0.62
# 100r: 0.79 mrr: 0.64
# 150v: 0.79 mrr: 0.63
# 200r: 0.81 mrr: 0.67

# slvr: 0.92 mrr: 0.81

# 042v: 0.81 mrr: 0.71
# 043r: 0.77 mrr: 0.64
# 043v: 0.80 mrr: 0.65
# 044r: 0.83 mrr: 0.68
# 044v: 0.81 mrr: 0.64


if __name__ == '__main__':
    # flags.DEFINE_integer('n_gram', 6, 'Language Model order')
    # flags.DEFINE_float('char_thr', 0.1, 'character probability pruning threshold')
    # flags.DEFINE_float('pdist_thr', 0.8, 'probability distribution pruning threshold')
    # flags.DEFINE_integer('alt_top_n', 0, 'top n transcriptions to submit to alternative generation')
    flags.DEFINE_string('src_dir', 'word_imgs/all_pages_clean', 'source folder')
    flags.DEFINE_string('dst_dir', 'tsc_gen_clean_ncthr=0.5', 'destination folder')
    flags.DEFINE_string('img_dir', 'images', 'word image folder name')
    flags.DEFINE_string('tsc_dir', 'transcriptions', 'word transcription folder name')
    flags.DEFINE_string('lm', 'lm_model/corpus_6gram.arpa', 'Language model file')
    flags.DEFINE_string('ocr', 'ocr_model/new_multiout.hdf5', 'character classifier file')
    flags.DEFINE_float('notchar_thr', 0.5, 'not character probability pruning threshold')
    flags.DEFINE_float('lm_thr', -16.0, 'LM probability pruning threshold')

    app.run(main)
