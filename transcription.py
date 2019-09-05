import itertools
import cv2
import pathlib
import json
# import base64
import numpy as np
import networkx as nx
from time import time
from evaluation import evaluate_word_accuracy, mrr, correct_transcr_count
from absl import app, flags
from cnn_wrapper import CNN
from dfs import DFS
from imgutils import find_midline_bounds
import matplotlib.pyplot as plt


"""
HELPER FUNCTIONS
"""


def _compute_segments_and_centroids(word_img, bg_color=(255, 255, 255)):
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
                (int(np.rint(cx)), int(np.rint(cy)))
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


"""
APP MAIN
"""


def main(unused_argv):
    # all_classes = [ # '1_not_char', '0_con',
    #     '0_b_stroke', '_curl', '0_d_stroke', '0_l_stroke',
    #     '0_nt',       '0_per',  '0_pro', '0_qui',      '0_rum',     '0_semicolon',
    #     'a', 'b', 'c', 'd', 'e',      'f',        'g', 'h', 'i', 'l', 'm', 'n',
    #     'o', 'p', 'q', 'r', 's_alta', 's_ending', 't', 'u', 'x'
    # ]
    # clrs = cv2.imread('palette2.png')[0]

    src_dir = pathlib.Path(flags.FLAGS.src_dir)
    dst_dir = pathlib.Path(flags.FLAGS.dst_dir)

    # load the OCR model
    icr_classifier = CNN(flags.FLAGS.ocr)

    # load Decoder + Language Model
    dfs_decoder = DFS(flags.FLAGS.lm)

    # keras.models.load_model(
    #     flags.FLAGS.ocr,
    #     custom_objects={
    #         'custom_loss':custom_loss,
    #         'char_accuracy':char_accuracy,
    #         'nochar_accuracy':nochar_accuracy
    #     }
    # )

    # global model_LM
    # if flags.FLAGS.lm == 'noLM':
    #     model_LM = None
    # else:
    #     model_LM = kenlm.Model(flags.FLAGS.lm)

    for page_path in src_dir.glob('*'):
        print(page_path)
        start_time = time()

        tsc_dir = dst_dir / page_path.stem / 'tsc'
        # dag_dir = dst_dir / page_path.stem / 'dag'
        tsc_dir.mkdir(parents=True, exist_ok=True)
        # dag_dir.mkdir(parents=True, exist_ok=True)

        line_tsc_top1 = ''

        word_paths = sorted(
            page_path.glob(flags.FLAGS.img_dir+'/*'),
            key=lambda x: (x.parts[2], int(x.parts[3]), int(x.stem.split('_')[0]))
        )

        prev_line_path = None

        for word_path in word_paths:
            word_img = cv2.imread(str(word_path))
            _, word_img_bin = cv2.threshold(cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(word_img_bin)
            word_img_crop = word_img[y:y+h,x:x+w]
            word_img_bin_crop = word_img_bin[y:y+h,x:x+w]

            x_line, baseline = find_midline_bounds(word_img_bin_crop, margin=3, slice=1)
            x_line, baseline = x_line[0][0], baseline[0][0]

            # word_img_bin_crop[[x_line, baseline],:] = 127
            #
            # plt.imshow(word_img_bin_crop)
            # plt.show()

            segments, centroids = _compute_segments_and_centroids(word_img_crop)

            if len(segments) > 0 and len(centroids) > 0:

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
                X_test = np.array([_make_sample(s, baseline, x_line, draw_lines=False) for s in grouped_segments], dtype='float32') / 255 #TODO parametri

                # fig = plt.figure(figsize=(8, 8))
                # columns = 10
                # rows = 10
                # for i in range(1, columns * rows + 1):
                #     if i >= len(X_test):
                #         break
                #     fig.add_subplot(rows, columns, i)
                #     plt.imshow(X_test[i][:,:,0])
                # plt.show()


                char_preds, notchar_preds = icr_classifier.predict(X_test)

                # char_preds = tf.nn.softmax(logit_preds[:, 1:], axis=1).numpy()
                # notchar_preds = tf.nn.sigmoid(logit_preds[:, :1]).numpy()

                # filter segment combinations according to classification
                filtered_combinations = []
                for i, cc in enumerate(centroid_ids):
                    if notchar_preds[i] <= flags.FLAGS.notchar_thr:
                        s_mask = np.zeros(grouped_segments[i][0].shape, dtype='uint8')
                        for s in grouped_segments[i]:
                            s_mask = cv2.bitwise_or(s_mask, s)
                        filtered_combinations.append((cc, char_preds[i], s_mask))

                # print(
                #     '{}:\nKept: {} out of {} potential edges ({:.2f}%)'.format(
                #         str(word_path),
                #         len(filtered_combinations),
                #         len(centroid_ids),
                #         (len(filtered_combinations)/len(centroid_ids)) * 100
                #     )
                # )

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

                # print("nodes: {},\tedges: {}\n".format(len(lattice.nodes()), len(lattice.edges())))

                # # save a .js file with the lattice structure
                # dict_nodes = [
                #     str([filtered_centroids.index(c) for c in node])
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
                # dst_dag = dag_dir / (word_path.stem + '.js')
                # with open(str(dst_dag), 'w') as f:
                #     f.write('var graph = ' + json.dumps(lattice_dict, indent=2))

                # Transcription generation:
                transcriptions = set()

                if len(lattice.nodes()) > 0:
                    for path in dfs_decoder.paths_beam(
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

                if word_path.parts[3] != prev_line_path:
                    line_tsc_top1 += '\n'
                if len(transcriptions_filtered) > 0:
                    line_tsc_top1 += transcriptions_filtered[0][1] + ' '
                else:
                    line_tsc_top1 += '* '

                prev_line_path = word_path.parts[3]

        print('\n'+line_tsc_top1.strip()+'\n')

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
    flags.DEFINE_string('src_dir', 'word_imgs/words_by_line/', 'source folder')
    flags.DEFINE_string('dst_dir', 'prova_linee', 'destination folder')
    flags.DEFINE_string('img_dir', '*', 'word image folder name')
    flags.DEFINE_string('tsc_dir', 'transcriptions', 'word transcription folder name')
    flags.DEFINE_string('lm', 'lm_model/6grams_bul_abbr.arpa', 'Language model file')
    flags.DEFINE_string('ocr', 'ocr_model/new_multiout.hdf5', 'character classifier file')
    flags.DEFINE_float('notchar_thr', 0.4, 'not character probability pruning threshold')
    flags.DEFINE_float('lm_thr', -16.0, 'LM probability pruning threshold')
    flags.DEFINE_bool('eval', False, 'run evaluation')

    app.run(main)
