import os
import ast
import re
import json
from math import inf
from collections import defaultdict, Counter, OrderedDict
import editdistance as ed
import pathlib
import difflib


GT_FOLDER = 'correct_transcriptions'


def _remove_abbreviations_icr(word):
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

def _remove_abbreviations_tess(word):
    tsc = re.sub(
        r"\(.*?\)|\'.*?\'|\[.*?\]|\{.*?\}",
        "",
        word.replace('s ', '6 ') \
            .replace('s.', '6.') \
            .replace('s' + '\n', '6' + '\n') \
            .replace('d(is)', '$') \
            .replace('d(u)', '$') \
            .replace('d(um)', '$') \
            .replace('(uo)d', '$') \
            .replace('d(inis)', '$') \
            .replace('d(iaconus)', '$') \
            .replace('d(em)', '$') \
            .replace('d(as)', '$') \
            .replace('d(inali)', '$') \
            .replace('q(uod)', '%') \
            .replace('(quod)', '%') \
            .replace('b(is)', '0') \
            .replace('b(er)', '0') \
            .replace('(res)b', '0') \
            .replace('b(ris)', '0') \
            .replace('b(at)', '0') \
            .replace('b(ate)', '0e') \
            .replace('r(um)', '1') \
            .replace('p(er)', '2') \
            .replace('p(ropter)', '22') \
            .replace('p(ar)', '2') \
            .replace('p(ro)', '3p') \
            .replace('q(ui)', '4') \
            .replace('q(uem)', '4') \
            .replace('q(ue)', 'q5') \
            .replace('b(us)', 'b5') \
            .replace('l(er)', '8') \
            .replace('l(es)', '8') \
            .replace('(e)l', '8') \
            .replace('(osto)l', '8') \
            .replace('l(endis)', '8') \
            .replace('l(endas)', '8') \
            .replace('l(ius)', '8') \
            .replace('l(ite)', '8') \
            .replace('(u)l', '8') \
            .replace('l(ecti)', '8') \
            .replace('l(ii)', '8') \
            .replace('l(esi)', '8') \
            .replace('l(is)', '8') \
            .replace('l(ectus)', '8') \
            .replace('li(ite)', '8') \
            .replace('l(ur)', '8') \
            .replace('XPO', 'Xo') \
            .replace('(Christ)', 'X')
            .replace('(rum)', '1') \
            .replace('(per)', '2') \
            .replace('(par)', '2') \
            .replace('(pro)', '3p') \
            .replace('(qui)', '4') \
            .replace('(que)', 'q5') \
            .replace('(bus)', 'b5') \
            .replace('(it)', '5') \
            .replace('(ed)', '5') \
            .replace('(et)', '7') \
            .replace('(ur)', '9') \
            .replace('(uius)', '9') \
            .replace('(imus)', '9') \
            .replace('(mus)', '9') \
            .replace('(us)', '9')
    )
    return tsc


def mrr(gen_folder, gt_folder=GT_FOLDER):
    words_filenames = os.listdir(gt_folder)

    rank_score = 0.0
    dist_ixs = defaultdict(int)

    for w_fnm in words_filenames:
        with open(os.path.join(gt_folder, w_fnm), 'r') as gt_file:
            gt_word = _remove_abbreviations_icr(gt_file.readline().strip())
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
                    gt_word = _remove_abbreviations_icr(gt_file.readline().strip())
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
            gt_word = _remove_abbreviations_icr(gt_file.readline().strip())
        with open(os.path.join(gen_folder, fnm), 'r') as gen_file:
            gen_words = [ast.literal_eval(w.strip())[1] for w in gen_file.readlines()]
        if gt_word in gen_words:
            correct_transcr += 1
    
    return correct_transcr/len(fnms)

def evaluate_full_transcriptions(gen_dir, gt_dir, evalfnm, mode='icr'):
    ok_modes = ['icr', 'tess']
    if mode not in ok_modes:
        raise ValueError("Unknown mode: use "+' or '.join(ok_modes))

    tsc_folder = pathlib.Path(gen_dir)

    evaluation = dict()

    eds = []
    for tsc_path in tsc_folder.glob('*'):
        print()
        print(tsc_path)
        evaluation[tsc_path.stem] = defaultdict(dict)
        gt_path = pathlib.Path(gt_dir) / (tsc_path.stem.split('_')[0] + '.txt')

        with tsc_path.open('r') as f:
            gen_lines = f.readlines()
        with gt_path.open('r') as g:
            gt_lines = g.readlines()

        page_eds = []

        for i, (gen_line, gt_line) in enumerate(zip(gen_lines, gt_lines)):
            if mode == 'icr':
                gt_line = _remove_abbreviations_icr(gt_line.strip())
            else:
                gt_line = _remove_abbreviations_tess(gt_line.strip())
            gen_line = gen_line.strip()
            eval_ed = ed.eval(gen_line, gt_line)

            evaluation[tsc_path.stem][i]['gt'] = gt_line
            evaluation[tsc_path.stem][i]['gen'] = gen_line
            evaluation[tsc_path.stem][i]['ed'] = eval_ed
            print(gen_line)
            print(gt_line)
            print(eval_ed)
            eds.append(eval_ed)
            page_eds.append(eval_ed)

        evaluation[tsc_path.stem]['avg_page_ed'] = sum(page_eds) / len(page_eds)

    print(len(eds))
    print(sum(eds) / len(eds))

    evaluation['avg_tot_ed'] = sum(eds) / len(eds)

    json.dump(evaluation, open(evalfnm+'.json', 'w'), indent=2)

def get_diffs_count(eval_fnm):
    tsc_eval = json.load(open(eval_fnm, 'r'))
    diffs = []

    for pagename in tsc_eval:
        if pagename != 'avg_tot_ed':
            for line in tsc_eval[pagename]:
                if line != 'avg_page_ed':
                    gt_tsc = tsc_eval[pagename][line]['gt']
                    gen_tsc = tsc_eval[pagename][line]['gen']

                    matcher = difflib.SequenceMatcher(None, gt_tsc, gen_tsc, autojunk=False)
                    diff = list(matcher.get_matching_blocks())

                    if len(diff) > 0:
                        diffs.append((gt_tsc[:diff[0].a], gen_tsc[:diff[0].b]))
                        for (a1, b1, size1), (a2, b2, size2) in zip(diff, diff[1:]):
                            diffs.append((gt_tsc[a1 + size1:a2], gen_tsc[b1 + size1:b2]))

    return Counter(diffs)



if __name__ == '__main__':
    # exts = ['_line', '_word', '_lineword', '_wordline']
    #
    # for ext in exts:
    evaluate_full_transcriptions(
        'page_transcription_icr_lmfix_lineth42',
        'page_transcriptions_gt',
        'page_tsc_eval_icr_lmfix_lineth42',
        mode='icr'
    )
    # count = get_diffs_count('page_tsc_eval_icr_all.json')
    #
    # countdict = OrderedDict([(str(k), c) for k, c in count.most_common(len(count))])
    #
    # json.dump(countdict, open('count_diffs_icr_old.json', 'w'), indent=2)

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
    #     # gt_line = _remove_abbreviations_icr(gt_line)
    #     # gt_line = ' '.join(gt_line.split())
    #     eval_ed = ed.eval(gen_line, gt_line)
    #     print(gen_line.strip())
    #     print(gt_line.strip())
    #     print(eval_ed)
    #     print()
    #     eds.append(eval_ed)
    #
    # print(sum(eds)/len(eds))
