import numpy as np
import kenlm
from collections import OrderedDict


class WordDFS:
    IX2TSC = {
        0: 'b',  # with stroke
        1: 'con',  # con?
        2: 'd',  # with stroke
        3: 'l',  # with stroke
        4: 'et',
        5: 'per',
        6: 'pro',
        7: 'qui',
        8: 'rum',
        9: ';',
        10: 'a',
        11: 'b',
        12: 'c',
        13: 'd',
        14: 'e',
        15: 'f',
        16: 'g',
        17: 'h',
        18: 'i',
        19: 'l',
        20: 'm',
        21: 'n',
        22: 'o',
        23: 'p',
        24: 'q',
        25: 'r',
        26: 's',
        27: 's',
        28: 't',
        29: 'u',
        30: 'x',
        31: '#',
        # 32: 'b#',  # with stroke
        # 33: 'con#',  # con?
        # 34: 'd#',  # with stroke
        # 35: 'l#',  # with stroke
        # 36: 'et#',
        # 37: 'per#',
        # 38: 'pro#',
        # 39: 'qui#',
        # 40: 'rum#',
        # 41: ';#',
        # 42: 'a#',
        # 43: 'b#',
        # 44: 'c#',
        # 45: 'd#',
        # 46: 'e#',
        # 47: 'f#',
        # 48: 'g#',
        # 49: 'h#',
        # 50: 'i#',
        # 51: 'l#',
        # 52: 'm#',
        # 53: 'n#',
        # 54: 'o#',
        # 55: 'p#',
        # 56: 'q#',
        # 57: 'r#',
        # 58: 's#',
        # 59: 's#',
        # 60: 't#',
        # 61: 'u#',
        # 62: 'x#',
    }

    # IX2TSC = {
    #     0: 'b',  # with stroke
    #     1: 'con',
    #     2: 'us',
    #     3: 'd',  # with stroke
    #     4: 'l',  # with stroke
    #     5: 'et',
    #     6: 'per',
    #     7: 'pro',
    #     8: 'qui',
    #     9: 'rum',
    #     10: ';',
    #     11: 'a',
    #     12: 'b',
    #     13: 'c',
    #     14: 'd',
    #     15: 'e',
    #     16: 'f',
    #     17: 'g',
    #     18: 'h',
    #     19: 'i',
    #     20: 'l',
    #     21: 'm',
    #     22: 'n',
    #     23: 'o',
    #     24: 'p',
    #     25: 'q',
    #     26: 'r',
    #     27: 's',
    #     28: 's',
    #     29: 't',
    #     30: 'u',
    #     31: 'x'
    # }

    def __init__(self, lm_path):
        self.lm = kenlm.Model(lm_path)

    def _edges_from(self, G, visited, width, lm_th):
        multiedges = []
        u, _, _ = visited[-1]

        prob = np.sum(score for _, _, score in visited)

        if prob > lm_th:
            for v in G.successors(u):
                ocr_predictions = G.get_edge_data(u, v)['preds'] # * (1 - 1/(len(visited)+1))

                # for rank_ix, pos_ix in enumerate(np.argsort(-ocr_predictions)):
                #     ocr_predictions[pos_ix] = ocr_predictions[pos_ix] / (rank_ix+2)

                ocr_predictions = np.log10(ocr_predictions)

                tsc = ''.join(c for _, c, _ in visited)

                tsc_score = self.lm.score(
                    ' '.join(
                        list(tsc.replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que'))
                    ),
                    bos=False,
                    eos=False
                )

                if len(G[v]) < 1:
                    lm_predictions = np.array([
                        self.lm.score(
                            ' '.join(list(
                                (tsc + self.IX2TSC[i] + '#').replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                            )),
                            bos=False,
                            eos=False
                        ) - tsc_score for i in self.IX2TSC
                    ])
                else:
                    lm_predictions = np.array([
                        self.lm.score(
                            ' '.join(list(
                                (tsc + self.IX2TSC[i]).replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                            )),
                            bos=False,
                            eos=False
                        ) - tsc_score for i in self.IX2TSC
                    ])

                predictions = np.sum(
                    [ocr_predictions, lm_predictions], # np.hstack((ocr_predictions, ocr_predictions[:-1]))
                    axis=0
                )
                multiedges += [(v, self.IX2TSC[tsc_ix], pred) for tsc_ix, pred in enumerate(predictions)]

            multiedges = sorted(multiedges, key=lambda x: x[-1], reverse=True)
            multiedges = multiedges[:width]

        return iter(multiedges)

    def paths_beam(self, G, source, targets, width, lm_th):
        visited = OrderedDict.fromkeys([(source, '#', 0.0)])
        stack = [self._edges_from(G, list(visited), width, lm_th)]

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
                    stack.append(self._edges_from(G, list(visited), width, lm_th))
                else:
                    visited.popitem()  # maybe other ways to child


class LineDFS:
    def __init__(self, word_dfs):
        self.wdfs = word_dfs

    def _edges_from(self, G, visited, width, lm_th):
        multiedges = []
        u, _, _ = visited[-1]

        prob_so_far = np.sum(score for _, _, score in visited)
        tsc_so_far = '#'.join(t for _, t, _ in visited)

        # print(prob_so_far, '\t',tsc_so_far)

        if prob_so_far > -35.0:
            for v in G.successors(u):
                transcriptions = set()
                wg = G[u][v]['graph']

                if len(wg.nodes()) > 0:
                    for path in self.wdfs.paths_beam(
                            wg,
                            source=sorted(wg.nodes())[0],
                            targets={sorted(wg.nodes())[-1]},
                            width=3,
                            lm_th=lm_th):
                        transcript = ''
                        prob = 0.0
                        for _, char, score in path[1:]:
                            transcript += char
                            prob += score
                        transcript = transcript.replace('b;', 'bus').replace('q;', 'que').replace('qui;', 'que')
                        transcript = transcript.strip(';')

                        transcriptions.add((prob, transcript))

                transcriptions = sorted(transcriptions, reverse=True)
                transcriptions_filtered = []

                for w in transcriptions:
                    if len(transcriptions_filtered) > 0:
                        _, tfil = zip(*transcriptions_filtered)
                    else:
                        tfil = []
                    if w[1] not in tfil:
                        transcriptions_filtered.append(w)

                for _, tsc in transcriptions_filtered[:3]:
                    tsc_score = self.wdfs.lm.score(
                        ' '.join(list('#'+tsc_so_far+'#'+tsc+'#')), bos=False, eos=False
                    ) - prob_so_far
                    multiedges.append((v, tsc, tsc_score))

            multiedges = sorted(multiedges, key=lambda x: x[-1], reverse=True)
            multiedges = multiedges[:width]

        return iter(multiedges)

    def paths_beam(self, G, source, targets, width, lm_th):
        visited = OrderedDict.fromkeys([(source, '', 0.0)])
        stack = [self._edges_from(G, list(visited), width, lm_th)]

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
                    stack.append(self._edges_from(G, list(visited), width, lm_th))
                else:
                    visited.popitem()  # maybe other ways to child
