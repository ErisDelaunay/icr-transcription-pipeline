import numpy as np
import kenlm
from collections import OrderedDict


class DFS:
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
        30: 'x'
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
                ocr_predictions = np.log10(G.get_edge_data(u, v)['preds'])
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
                    [ocr_predictions, lm_predictions],
                    axis=0
                )
                multiedges += [(v, self.IX2TSC[tsc_ix], pred) for tsc_ix, pred in enumerate(predictions)]

            multiedges = sorted(multiedges, key=lambda x: x[-1], reverse=True)[:width]

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
