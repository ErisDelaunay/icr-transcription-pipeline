import numpy as np
import cv2
import matplotlib.pyplot as plt


def group_consecutive_values(values, threshold=2):
    """
        raggruppa valori interi consecutivi crescenti in una lista (a distanza threshold).
        ad es.: [1,2,4,6] -> [[1,2],[4],[6]] con threshold = 1
                [1,2,4,6] -> [[1,2,4,6]] con threshold = 2
    """
    run = []
    result = [run]
    last = values[0]
    for v in values:
        if v-last <= threshold:
            run.append(v)
        else:
            run = [v]
            result.append(run)
        last = v
    return result


def remove_left_margin(page_img):
    black_px_per_column = np.count_nonzero(
        cv2.bitwise_not(page_img), axis=0
    )[:page_img.shape[1]//2]
    left_margins = np.argwhere(
        black_px_per_column < np.average(black_px_per_column)*0.5
    )
    left_margin = int(
        max([min(g) for g in group_consecutive_values(left_margins)])
    )
    return page_img[:, left_margin:], left_margin



def page_to_lines(page_img):

    black_px_per_row = np.count_nonzero(cv2.bitwise_not(page_img), axis=1)

    threshold = np.average(black_px_per_row)*1.1
    ixs = np.argwhere(black_px_per_row < threshold)
    ixs_grouped = group_consecutive_values(ixs, threshold=7)

    min_ixs = [int(np.average(ixs)) for ixs in ixs_grouped]
    lines = []
    # page_img[min_ixs,:] = 127
    # plt.imshow(page_img)
    # plt.show()

    for i in range(0, len(min_ixs)-1):
        top_y = min_ixs[i]
        bottom_y = min_ixs[i+1]
        line = page_img[top_y:bottom_y]
        lines.append((top_y,bottom_y))

    return np.array(lines)




def page_to_lines2(page_img):

    black_px_per_row = np.count_nonzero(cv2.bitwise_not(page_img), axis=1)
    #0.85
    threshold = np.average(black_px_per_row)*0.85
    ixs = np.argwhere(black_px_per_row < threshold)
    ixs_grouped = group_consecutive_values(ixs, threshold=7)

    min_ixs = [int(np.average(ixs)) for ixs in ixs_grouped]
    lines = []

    for i in range(0, len(min_ixs)-1):
        top_y = min_ixs[i]
        bottom_y = min_ixs[i+1]
        lines.append((top_y,bottom_y))
    #lines = np.array(lines)
    #lines[0][0] = 0
    #lines[-1][1] = page_img.shape[0]

    return np.array(lines)

def line_to_words(line, top_y):
    """
        suddivide una linea di testo in parole.
        input: la linea e la sua posizione y
        output: una lista di triple (parola, top_y, left_x)
    """
    kernel = np.ones((2,2), np.uint8)
    dilation = cv2.dilate(line, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=4)

    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(erosion))

    midline = np.argmax(np.count_nonzero(cv2.bitwise_not(erosion), axis=1))

    # line_cp = line.copy()

    # for x,y,w,h,_ in stats[1:]:
    #     if y <= midline <= y+h:
    #         cv2.rectangle(line_cp, (x-1,y-1), (x-1+w,y-1+h), 127, 1)
    # line_cp[midline] = 127
    # plt.imshow(line_cp)
    # plt.show()

    words = sorted(
        [(line[:, x:x+w], x, top_y)
            for x, y, w, h, _ in stats[1:] if (y <= midline <= y+h)],
        key=lambda x: x[1]
    )

    # for word, x, y in words:
    #     print(x,y)
    #     plt.imshow(word)
    #     plt.show()

    return words

def find_local_minima(a):
    """
        ritorna gli indici corrispondenti ai minimi locali in un'immagine
    """
    local_minima = []

    # un punto e' minimo locale se e' minore dei suoi punti adiacenti;
    # nello specifico, diciamo che deve essere minore strettamente almeno di uno dei due
    # (mentre puo' essere minore o uguale all'altro)

    local_minima.append(0)

    for i in range(1, len(a) - 1):

        is_local_minimum = (
                (a[i] <= a[i - 1] and a[i] < a[i + 1]) or
                (a[i] < a[i - 1] and a[i] <= a[i + 1])
        )

        if is_local_minimum:
            local_minima.append(i)

    local_minima.append(len(a) - 1)

    return np.array(local_minima)



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise( ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise( ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(int(window_len/2)-1):-(int(window_len/2))]

def remove_small_component(img,THRESHOLD = 6):
	img = cv2.bitwise_not(img)

	n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
	for i in range(1, n_labels):
		x, y, w, h, a= stats[i]
		if a<=THRESHOLD:
			img = np.where(labels == i, 0, img)
	img = cv2.bitwise_not(img)
	return img

def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise( ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise( ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(window_len//2):(window_len//2)+len(x)]

def get_polygons(img_th):
    img_th = cv2.bitwise_not(img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(0,len(contours)):

        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i]))

    drawing = np.zeros((img_th.shape[0], img_th.shape[1]), np.uint8)

    # draw contours and hull points
    for i in range(0, len(contours)):
        #cv2.drawContours(drawing, contours, i, 255, 1, 8, hierarchy)

        cv2.drawContours(drawing, hull, i, 255, -1)

    return drawing

def is_local_maxima(seq):
    rising = np.r_[True, seq[1:] > seq[:-1]] & np.r_[seq[:-1] >= seq[1:], True]
    falling = np.r_[True, seq[1:] >= seq[:-1]] & np.r_[seq[:-1] > seq[1:], True]
    return np.bitwise_or(rising, falling)

def find_midline_bounds(img, margin=7, slice=4):
    """
    Find the x-line and baseline from a line of text image
    :param img: image of a line of text
    :param margin: margin to apply to the writing line (to account for a certain tolerance)
    :param slice: how many slices per line (take into account skewed lines)
    :return: x-line and baseline for each slice of the img
    """



    splits = np.array_split(img, slice, axis=1)

    x_line, baseline = [], []

    splt_start, splt_end = 0, 0


    for splt in splits:
        splt_end += splt.shape[1]
        black_px_per_row = np.count_nonzero(splt, axis=1)
        black_px_per_row_smooth = smooth(black_px_per_row, splt.shape[0]//7)
        maxima_ixs = np.argwhere(is_local_maxima(black_px_per_row_smooth))

        candidate_midlines = []
        for start_ix, end_ix in zip(maxima_ixs, maxima_ixs[1:]):
            start_ix_margin = max(start_ix[0] - margin, 0)
            end_ix_margin = min(end_ix[0] + margin, splt.shape[0] - 1)

            candidate_midlines.append(
                (np.count_nonzero(splt[start_ix_margin:end_ix_margin, :]),
                 start_ix_margin,
                 end_ix_margin)
            )
        if len(candidate_midlines) == 0:
            x_line.append((0, splt_start, splt_end)) # += [0] * splt.shape[1]
            baseline.append((splt.shape[0]-1, splt_start, splt_end)) # += [splt.shape[0]-1] * splt.shape[1]
        else:
            _, x_line_val, baseline_val = sorted(candidate_midlines, reverse=True)[0]
            x_line.append((x_line_val, splt_start, splt_end)) # += [x_line_val] * splt.shape[1]
            baseline.append((baseline_val, splt_start, splt_end)) # += [baseline_val] * splt.shape[1]
        splt_start = splt_end

    return x_line, baseline
