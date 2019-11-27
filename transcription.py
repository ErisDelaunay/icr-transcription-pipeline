import itertools
import cv2
import pathlib
import json
import numpy as np
import networkx as nx
from time import time
from evaluation import evaluate_word_accuracy, mrr, correct_transcr_count
from absl import app, flags
from cnn_wrapper import CNN
from dfs import WordDFS, LineDFS
import matplotlib.pyplot as plt


"""
HELPER FUNCTIONS
"""


def _compute_segments_and_centroids(word_img, offset=(0, 0), bg_color=(255, 255, 255)):
    colors = np.unique(word_img.reshape(-1, word_img.shape[2]), axis=0) if word_img.size > 0 else []
    colors = [
        np.array((b, g, r)) for b, g, r in colors if (b, g, r) != bg_color
    ]

    centroids = []
    segments = []

    for color in colors:
        mask = cv2.inRange(word_img, lowerb=color, upperb=color)
        _, _, w, h = cv2.boundingRect(mask)

        if w * h > 0:
            cy, cx = np.average(np.argwhere(mask != 0), axis=0)
            centroids.append(
                (int(np.rint(cx)) + offset[0], int(np.rint(cy)) + offset[1])
            )
            segments.append(mask)

    return segments, centroids


def _make_sample(segments, baseline=None, x_line=None, sample_shape=56, draw_lines=False):
    word_mask = np.zeros(segments[0].shape, dtype='uint8')

    for s in segments:
        word_mask = cv2.bitwise_or(word_mask, s)

    x, y, w, h = cv2.boundingRect(word_mask)

    bbx_crop = word_mask[y:y+h, x:x+w]

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

    if draw_lines:
        new_xline = min(max(top - (y - x_line), 0), sample_shape - 1)
        new_baseline = min(new_xline + (baseline - x_line), sample_shape - 1)
        sample_img[[new_baseline, new_xline], :] = 127


    if sample_img.shape != (sample_shape, sample_shape):
        sample_img = cv2.resize(
            sample_img,
            (sample_shape, sample_shape),
            interpolation=cv2.INTER_NEAREST
        )

    return sample_img.reshape((sample_shape,sample_shape,1))


def merge_graphs(graphs, class_n=32, add_space_edge=True, space_prob=0.82):
    line_graph = nx.compose_all(graphs) if len(graphs)> 0 else nx.DiGraph()
    p = [(1 - space_prob) / (class_n - 1)] * (class_n - 1) + [space_prob]
    assert sum(p) == 1.0

    for prev_word_graph, next_word_graph in zip(graphs, graphs[1:]):
        src_node = sorted(next_word_graph.nodes)[0]
        sink_nodes = list(
                filter(
                lambda v: len(prev_word_graph.out_edges(v)) == 0,
                prev_word_graph.nodes
            )
        )

        avg_sink_len = np.average([len(s) for s in sink_nodes])

        sink_nodes_filter = filter(lambda v: len(v) > avg_sink_len*0.66, sink_nodes)

        for sn in sink_nodes_filter:
            # add space edges
            if add_space_edge:
                line_graph.add_edge(sn, src_node, preds=np.array(p))

            # connect sinks predecessors to source
            sn_inedges = prev_word_graph.in_edges(sn, data=True)
            for sn_pred, _, data in sn_inedges:
                line_graph.add_edge(sn_pred, src_node)
                line_graph[sn_pred][src_node].update(data)

    return line_graph


"""
APP MAIN
"""


def main(unused_argv):
    src_dir = pathlib.Path(flags.FLAGS.src_dir)
    dst_dir = pathlib.Path(flags.FLAGS.dst_dir)

    # load the OCR model
    icr_classifier = CNN(flags.FLAGS.ocr)

    # load Decoder + Language Model
    word_decoder = WordDFS(flags.FLAGS.lm)
    line_decoder = LineDFS(word_decoder)

    for page_path in list(src_dir.glob('*'))[:]:
        start_time = time()

        tsc_dir = dst_dir / page_path.stem / 'tsc'
        # dag_dir = dst_dir / page_path.stem / 'dag'
        tsc_dir.mkdir(parents=True, exist_ok=True)
        # dag_dir.mkdir(parents=True, exist_ok=True)

        line_paths = sorted(
            page_path.glob('*'),
            key=lambda x: int(x.stem.split('-')[0])
        )

        for line_path in line_paths:
            print(line_path)

            line_img = cv2.imread(str(line_path))
            _, line_img_bin = cv2.threshold(
                cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV
            )

            nlabels, labels, stats, ctds = cv2.connectedComponentsWithStats(line_img_bin)

            cc_order = ctds[:,0].argsort()
            cc_order = cc_order[cc_order>0]

            if len(cc_order) == 0:
                print("WARNING: Empty line!")
                continue

            prev_cc_x, _, prev_cc_w, _, _ = stats[cc_order[0]]
            cc_ix_groups = []
            cc_ix_group = []
            closeness_th = 5

            for cc_ix, (cc_x, cc_y, cc_w, _, _) in zip(cc_order, stats[cc_order]):
                if cc_x - (prev_cc_x + prev_cc_w) > closeness_th:
                    cc_ix_groups.append(cc_ix_group)
                    cc_ix_group = []
                cc_ix_group.append(cc_ix)
                prev_cc_x = cc_x
                prev_cc_w = cc_w
            if len(cc_ix_group) > 0:
                cc_ix_groups.append(cc_ix_group)

            line_pieces = []
            for cc_group in cc_ix_groups:
                cc_graph = nx.DiGraph()
                for u, v in itertools.combinations((tuple(cc_group[:i]) for i in range(len(cc_group) + 1)), 2):
                    mask_elements = set(v) - set(u)
                    mask = np.zeros(labels.shape, dtype='uint8')
                    for m_e in mask_elements:
                        mask = cv2.bitwise_or(
                            mask, np.uint8(np.where(labels == m_e, 255, 0))
                        )
                    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(mask)
                    color_group = cv2.bitwise_not(cv2.bitwise_and(line_img, line_img, mask=mask))
                    color_group = color_group[y_crop:y_crop+h_crop,x_crop:x_crop+w_crop]

                    segments, centroids = _compute_segments_and_centroids(color_group, offset=(x_crop, y_crop))

                    if len(segments) == 0 or len(centroids) == 0:
                        continue

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

                    char_preds, notchar_preds = icr_classifier.predict(X_test)

                    # filter segment combinations according to classification
                    filtered_combinations = []
                    for i, cc in enumerate(centroid_ids):
                        if notchar_preds[i] <= flags.FLAGS.notchar_thr:
                            s_mask = np.zeros(grouped_segments[i][0].shape, dtype='uint8')
                            for s in grouped_segments[i]:
                                s_mask = cv2.bitwise_or(s_mask, s)
                            filtered_combinations.append((cc, char_preds[i], s_mask))

                    filtered_combinations = sorted(filtered_combinations, key=lambda x: x[0][0])
                    filtered_centroids = sorted({c for ctds, _, _ in filtered_combinations for c in ctds})

                    # creation of the word lattice: segment combinations represent
                    # edges. Nodes are segments consumed up to a certain point.
                    word_lattice = nx.DiGraph()

                    nodes = [
                        set(filtered_centroids[:i])
                        for i in range(len(filtered_centroids) + 1)
                    ]
                    edges = [
                        set(ctds)
                        for ctds, _, _ in filtered_combinations
                    ]

                    for w_u, w_v in itertools.combinations(nodes, 2):
                        if w_v - w_u in edges:
                            _, preds, sgmt_img = filtered_combinations[edges.index(w_v - w_u)]
                            word_lattice.add_edge(
                                tuple(sorted(w_u)),
                                tuple(sorted(w_v)),
                                preds=preds,
                                image=sgmt_img
                            )
                    if len(word_lattice.nodes) > 0:
                        word_lattice = nx.relabel_nodes(
                            word_lattice, {(): ((x_crop, y_crop),)}
                        )

                    cc_graph.add_edge(u, v, img=color_group, graph=word_lattice)
                line_pieces.append(cc_graph)

            line_tsc = ''
            for p_n, piece in enumerate(line_pieces):
                transcriptions = set()
                for path in line_decoder.paths_beam(
                            piece,
                            source=tuple(),
                            targets={n for n in piece.nodes if piece.out_degree(n) < 1},
                            width=5,
                            lm_th=flags.FLAGS.lm_thr
                        ):
                    transcript = ''
                    prob = 0.0
                    for _, char, score in path[1:]:
                        transcript += '#' + char
                        prob += score
                    transcript = transcript.replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                    transcript = transcript.replace(';', '')
                    # print(prob, '\t', transcript)
                    transcriptions.add((prob, transcript+'#', tuple(path)))

                transcriptions = sorted(transcriptions, reverse=True)
                transcriptions_filtered = []

                for tsc_line in transcriptions:
                    if len(transcriptions_filtered) > 0:
                        _, tfil, _ = zip(*transcriptions_filtered)
                    else:
                        tfil = []
                    if tsc_line[1] not in tfil:
                        transcriptions_filtered.append(tsc_line)

                tsc_piece_path = tsc_dir / (line_path.stem + '_' + str(p_n)+'.txt')
                with tsc_piece_path.open('w') as lp:
                    for t_f in transcriptions_filtered:
                        lp.write(str(t_f)+'\n')

                line_tsc += transcriptions_filtered[0][1] if len(transcriptions_filtered) > 0 else ''

            print(line_tsc)

            with open(page_path.stem+'.txt', 'a') as f:
                f.write(line_tsc.replace('#', ' ').strip() + '\n')

        end_time = time()

        # evaluation
        elapsed = end_time - start_time
        print('time elapsed:', elapsed)

        if flags.FLAGS.eval:
            MRR = mrr(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
            correct_count = correct_transcr_count(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
            evaluation = evaluate_word_accuracy(str(tsc_dir), str(page_path / flags.FLAGS.tsc_dir))
            evaluation['mrr'] = MRR
            evaluation['time'] = elapsed
            evaluation['correct_overall'] = correct_count

            print(json.dumps(evaluation, indent=2))
            json.dump(evaluation, open(str(tsc_dir)+'_eval.json','w'), indent=2)


if __name__ == '__main__':
    flags.DEFINE_string('src_dir', 'word_imgs/e2e_lines/', 'source folder')
    flags.DEFINE_string('dst_dir', 'prova_linee', 'destination folder')
    flags.DEFINE_string('img_dir', '*', 'word image folder name')
    flags.DEFINE_string('tsc_dir', 'transcriptions', 'word transcription folder name')
    flags.DEFINE_string('lm', 'lm_model/6grams_bul_abbr.arpa', 'Language model file')
    flags.DEFINE_string('ocr', 'ocr_model/new_multiout.hdf5', 'character classifier file')
    flags.DEFINE_float('notchar_thr', 0.4, 'not character probability pruning threshold')
    flags.DEFINE_float('lm_thr', -16.0, 'LM probability pruning threshold')
    flags.DEFINE_bool('eval', False, 'run evaluation')

    app.run(main)
