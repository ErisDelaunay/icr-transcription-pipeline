import cv2
import numpy as np
import find_text_area as ta
import binarization as bin
import word_segmentation as ws
from cnn_wrapper import CNN
from dfs import DFS


def page_to_words(path):
    img = ta.find_text_area(str(path))
    print("text area ok")
    master_binarized_img = bin.master_binarization(img)
    print("master binarization ok")
    slave_binarized_img = bin.slave_binarization(
        master_binarized_img,
        img
    )
    print("slave binarization ok")
    word_8_add_binarization = ws.find_words_add_8_binarization(
        cv2.bitwise_not(slave_binarized_img), img
    )
    word_diveded_color = ws.find_words(
        cv2.bitwise_not(word_8_add_binarization)
    )
    print("word segmentation ok")
    return word_diveded_color


def page_to_lines(page_img):
    colors = np.unique(page_img.reshape(-1, page_img.shape[2]), axis=0)
    colors = [
        np.array((b, g, r)) for b, g, r in colors if (b, g, r) != (0, 0, 0)
    ]

    centroids = []

    for color in colors:
        mask = cv2.inRange(page_img, color, color)
        argmask = np.argwhere(mask != 0)
        y, x = np.average(argmask, axis=0)

        centroids.append((x, y, color))

    centroids = sorted(centroids, key=lambda e: (e[1], e[0]))

    prev_y = 0
    th = 10  # px
    lines = []
    line = []

    for x, y, c in centroids:
        if (y - prev_y) > th:
            if len(line) > 1:
                lines.append(line)
            line = []
        line.append((x, y, c))
        prev_y = y
    if len(line) > 1:
        lines.append(line)

    sorted_lines = []
    for l in lines:
        sorted_lines.append(sorted(l, key=lambda e: e[0]))

    return  sorted_lines


if __name__ == '__main__':
    color_page_path = ''
    # 1) segmentazione in parole
    ws_page = page_to_words(color_page_path)
    # 2) individuazione righe
    lines = page_to_lines(ws_page)

    icr_classifier = CNN('')
    dfs = DFS('')

    for line in lines:
        for _, _, color in line:
            word = cv2.inRange(ws_page, color, color)
            x, y, w, h = cv2.boundingRect(word)
            word_crop = word[y:y+h,x:x+w]

            word_segm = word_crop # TODO metodo segmentazione
            # 3) segmentazione parole

    # 4) trascrizione riga parola per parola