import os
import ast
import re
from math import inf
from collections import defaultdict
import editdistance as ed
import pathlib


GT_FOLDER = 'correct_transcriptions'


def _remove_abbreviations(word):
    tsc = re.sub(
        r"\(.*?\)|\'.*?\'|\[.*?\]|\{.*?\}",
        "", 
        word.lower()
            .replace('v', 'u')
            .replace('j', 'i')
            .replace('r(um)', 'rum')
            .replace('p(er)', 'per')
            .replace('p(ro)', 'pro')
            .replace('q(ui)', 'qui')
            .replace('q(ue)', 'que')
            .replace('b(us)', 'bus')
            .replace('(rum)', 'rum')
            .replace('(per)', 'per')
            .replace('(pro)', 'pro')
            .replace('(qui)', 'qui')
            .replace('(que)', 'que')
            .replace('(bus)', 'bus')
            .replace('(con)', 'con')
            .replace('(et)', 'et')
            .replace('-', '')
            .replace(',', '')
            .replace('.', '')
    )
    return tsc


def mrr(gen_folder, gt_folder=GT_FOLDER):
    words_filenames = os.listdir(gt_folder)

    rank_score = 0.0
    dist_ixs = defaultdict(int)

    for w_fnm in words_filenames:
        with open(os.path.join(gt_folder, w_fnm), 'r') as gt_file:
            gt_word = _remove_abbreviations(gt_file.readline().strip())
        with open(os.path.join(gen_folder, w_fnm), 'r') as gen_file:
            gen_words = [ast.literal_eval(w.strip())[1] for w in gen_file.readlines()]

        if gt_word in gen_words:
            rank_score += 1 / (gen_words.index(gt_word) + 1)
            dist_ixs[gen_words.index(gt_word)] += 1
        else:
            dist_ixs[-1] += 1

    mean_rr = rank_score/len(words_filenames)

    return mean_rr


def evaluate_word_accuracy(gen_folder, gt_folder=GT_FOLDER):
    words_filenames = os.listdir(gt_folder)

    ed_thresholds = [0,1,2]
    topNs = [1,2,3,5,10]

    results = {'top-'+str(topn):{'ed-'+str(ed): 0.0 for ed in ed_thresholds} for topn in topNs}

    for ed_threshold in ed_thresholds:
        for topN in topNs:
            correct = 0
            for w_fnm in words_filenames:
                with open(os.path.join(gt_folder, w_fnm), 'r') as gt_file:
                    gt_word = _remove_abbreviations(gt_file.readline().strip())
                with open(os.path.join(gen_folder, w_fnm), 'r') as gen_file:
                    gen_words = [ast.literal_eval(w.strip())[1] for w in gen_file.readlines()]

                if len(gen_words) > 0:
                    min_ed = min([ed.eval(gt_word, gen_word) for gen_word in gen_words[:topN]])
                else:
                    min_ed = inf

                if min_ed <= ed_threshold:
                    correct += 1
            results['top-'+str(topN)]['ed-'+str(ed_threshold)] = correct/len(words_filenames)

    return results
                

def correct_transcr_count(gen_folder, gt_folder=GT_FOLDER):
    fnms = os.listdir(gt_folder)
    correct_transcr = 0

    for fnm in fnms:
        with open(os.path.join(gt_folder, fnm), 'r') as gt_file:
            gt_word = _remove_abbreviations(gt_file.readline().strip())
        with open(os.path.join(gen_folder, fnm), 'r') as gen_file:
            gen_words = [ast.literal_eval(w.strip())[1] for w in gen_file.readlines()]
        if gt_word in gen_words:
            correct_transcr += 1
    
    return correct_transcr/len(fnms)

import json
if __name__ == '__main__':
    tsc_folder = pathlib.Path('evals/eval_multiout-0.4/043r_206_219_1382_1832/tsc')

    eds = []
    for tsc_path in tsc_folder.glob('*'):
        gt_path = pathlib.Path(
            'word_imgs/all_pages_clean/' + tsc_path.parts[-3] + '/transcriptions/' + tsc_path.name
        )

        with tsc_path.open('r') as f:
            top1 = f.readline()
        with gt_path.open('r') as g:
            gt_tsc = _remove_abbreviations(g.readline())

        if len(top1) > 0:
            _, gen_tsc, _ = ast.literal_eval(top1)
        else:
            gen_tsc = ''

        eval_ed = ed.eval(gen_tsc, gt_tsc)
        print(gen_tsc, gt_tsc)
        print(eval_ed)
        eds.append(eval_ed)

    eds += [6]*30
    print(len(eds))
    print(sum(eds) / len(eds))


    # tsc_json = json.load(open('tesseract_evaluation.json', 'r'))
    #
    # ixs_by_y = zip(
    #     sorted(tsc_json['lines'].keys(), key=lambda x: int(x)),
    #     sorted(tsc_folder.glob('*'), key=lambda x: int(x.stem.split('_')[1]))
    # )
    #
    # # tscs_by_y = sorted(tsc_folder.glob('*'), key=lambda p: int(p.stem.split('_')[1]))
    # #
    # lines = []
    # line = []
    # prev_y = 0
    # line_size = 30
    # for ix, t in ixs_by_y:
    #     cur_y = int(t.stem.split('_')[1])
    #     if (cur_y - prev_y >= line_size) and len(line)>0:
    #         lines.append(line)
    #         line = []
    #     line.append((ix, t))
    #     prev_y = cur_y
    #
    # lines = [sorted(l, key=lambda p: int(p[1].stem.split('_')[0])) for l in lines]
    #
    # genvsgt_tscs = []
    # for l in lines:
    #     gen_line_tsc = ''
    #     gt_line_tsc = ''
    #     for ix, _ in l:
    #         gen_tsc = tsc_json['lines'][ix]['pred']
    #         gen_line_tsc += gen_tsc + ' '
    #         gt_tsc = tsc_json['lines'][ix]['gt']
    #         gt_line_tsc += gt_tsc + ' '
    #     genvsgt_tscs.append((gen_line_tsc, gt_line_tsc))
    #
    #
    # # # with open('tsc_040r_195_275_1355_1807.txt', 'r') as tsc:
    # # #     gen_tscs = tsc.readlines()
    # #
    # # with open('200r.txt', 'r') as gt:
    # #     gt_tscs = gt.readlines()
    # #
    # eds = []
    #
    # for gen_line, gt_line in genvsgt_tscs:
    #     # gt_line = _remove_abbreviations(gt_line)
    #     # gt_line = ' '.join(gt_line.split())
    #     eval_ed = ed.eval(gen_line, gt_line)
    #     print(gen_line.strip())
    #     print(gt_line.strip())
    #     print(eval_ed)
    #     print()
    #     eds.append(eval_ed)
    #
    # print(sum(eds)/len(eds))
