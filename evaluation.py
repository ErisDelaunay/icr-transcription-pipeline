import os
import ast
import re
from math import inf
from collections import defaultdict
import editdistance as ed


GT_FOLDER = 'correct_transcriptions'


def _remove_abbreviations(word):
    tsc = re.sub(
        r"\(.*?\)|\'.*?\'", 
        "", 
        word.lower()
            .replace('(et)', 'et')
            .replace('(rum)', 'rum')
            .replace('p(er)', 'per')
            .replace('p(ro)', 'pro')
            .replace('q(ui)', 'qui')
            .replace('q(ue)', 'que')
            .replace('b(us)', 'bus')
            .replace('-', '')
            .replace(',', '')
            .replace('.', '')
            .replace('v', 'u')
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
