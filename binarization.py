import cv2
import numpy as np
import imgutils as utils


def remove_small_component_and_abbreviations(img,THRESHOLD):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, n_labels):
        x, y, w, h, a= stats[i]
        if a<=THRESHOLD or (h<12 and w>=10) or (h<=2 and w >=7):
            img = np.where(labels == i, 0, img)
    return img



def get_all_horizontal_projection(text_area_img,numbers_windows,windows_size):
    all_orizontal_projection = []
    j = 1
    while (j<=numbers_windows):

        text_area = text_area_img[0:text_area_img.shape[0], windows_size*(j-1):windows_size*j]

        black_px_per_row = np.count_nonzero(cv2.bitwise_not(text_area), axis=1)
        black_px_per_row = utils.smooth(black_px_per_row, 3, 'flat')

        threshold = np.average(black_px_per_row) * 1.1
        ixs = np.argwhere(black_px_per_row < threshold)

        ixs_grouped = utils.group_consecutive_values(ixs, threshold=7)
        min_ixs = [int(np.average(ixs)) for ixs in ixs_grouped]
        #del min_ixs[0]
        #del min_ixs[-1]
        all_orizontal_projection.append(min_ixs)
        j = j+1
    return all_orizontal_projection


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def connecting_minima_projection(all_projection):
    starting_points = (all_projection[0])
    j = 0
    points = []
    points.append(starting_points)
    while j<len(all_projection):
        if j!=0: #indice dello starting points
            points_tmp = []
            last_points = points[-1]
            array = all_projection[j]
            for i in range(0,len(last_points)):
                nearest = find_nearest(array,last_points[i])
                if (abs(nearest-last_points[i]))<=10:
                    points_tmp.append(nearest)
                else:
                    points_tmp.append(last_points[i])
            points.append(points_tmp)
        j = j+1
    return points

def remove_abbreviations_coordinates(line):
    img = line.astype('uint8')
    img = cv2.bitwise_not(img)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    x_line, baseline = utils.find_midline_bounds(img, margin=1)
    list_coordinate  = []
    for i in range(1, n_labels):
        x, y, w, h, _ = stats[i]
        x_line_val = [xlv for xlv, start, end in x_line if start <= x <= end][0]
        if ((y + h <= x_line_val)or (h<=5 and w>=12) ):
            list_coordinate.append((x,y,w,h))
    return list_coordinate


def from_page_to_lines_remove_abbreviations(polygons_page, first_th,second_th):

    thresh_img = cv2.bitwise_not(polygons_page)
    black_px_per_column = np.sum(thresh_img == 0, axis=0)
    black_px_per_column = utils.smooth(black_px_per_column, 5, 'flat')
    avg = (np.average(black_px_per_column) * 0.75)

    black = []
    for i in black_px_per_column:
        black.append(int(i))
    ixs = np.argwhere(black < avg)
    index = []
    for i in ixs:
        index.append(i[0])
    values = utils.group_consecutive_values(index, 12)
    # (values)
    new_thres = []
    for i in values:
        if (len(i)) > 40:
            new_thres.append((min(i), max(i)))
    threshold_value = 0
    if (len(new_thres) > 1):
        sorted_th = sorted(new_thres, key=lambda tup: tup[0])
        if sorted_th[0][1]>10 and sorted_th[0][1] < 200:
            threshold_value = sorted_th[0][1]-10
        else:
            threshold_value = 0
        #print (threshold_value)
        thresh_img = thresh_img[0:thresh_img.shape[0], threshold_value:thresh_img.shape[1]]


    N_WINDOWS = 3
    WINDOWS_SIZE = int(thresh_img.shape[1] / N_WINDOWS)
    all_horizontal_projeciton = get_all_horizontal_projection(thresh_img, N_WINDOWS, WINDOWS_SIZE)
    all_horizontal_projeciton = connecting_minima_projection(all_horizontal_projeciton)


    empty_img = np.zeros((second_th.shape[0], second_th.shape[1]), np.uint8)
    empty_img.fill(255)
    for i in range(1, len(all_horizontal_projeciton[0])):

        for k in range(0,3):
            if k == 0:
                img1 = (first_th[all_horizontal_projeciton[0][i - 1]:all_horizontal_projeciton[k][i], 0:WINDOWS_SIZE + threshold_value])
                img2 = (second_th[all_horizontal_projeciton[0][i - 1]:all_horizontal_projeciton[k][i], 0:WINDOWS_SIZE + threshold_value])

                coordinates = np.array(remove_abbreviations_coordinates(img1))
                for j in coordinates:
                    img2[j[1]:j[1] + j[3], j[0]:j[0] + j[2]] = 255

                empty_img[all_horizontal_projeciton[0][i - 1]:all_horizontal_projeciton[0][i], 0:WINDOWS_SIZE + threshold_value] = img2

            else:
                img1 = (first_th[all_horizontal_projeciton[k][i - 1]:all_horizontal_projeciton[k][i],(WINDOWS_SIZE*k) + threshold_value:(WINDOWS_SIZE * (k+1)) + threshold_value])
                img2 = (second_th[all_horizontal_projeciton[k][i - 1]:all_horizontal_projeciton[k][i],(WINDOWS_SIZE*k) + threshold_value:(WINDOWS_SIZE * (k+1)) + threshold_value])
                coordinates = np.array(remove_abbreviations_coordinates(img1))
                for j in coordinates:
                    img2[j[1]:j[1] + j[3], j[0]:j[0] + j[2]] = 255
                empty_img[all_horizontal_projeciton[k][i - 1]:all_horizontal_projeciton[k][i],(WINDOWS_SIZE*k) + threshold_value:(WINDOWS_SIZE * (k+1)) + threshold_value] = img2

    return empty_img



def master_binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    threshold = (int(np.mean(img_gray)))

    _, first_img = cv2.threshold(img_gray, threshold * 0.60, 255, cv2.THRESH_BINARY)
    first_img = cv2.copyMakeBorder(first_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

    _, second_img = cv2.threshold(img_gray, threshold * 0.75, 255, cv2.THRESH_BINARY)
    second_img = cv2.copyMakeBorder(second_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)


    first_img_polygons = utils.get_polygons(first_img)
    second_img_no_abbreviations = from_page_to_lines_remove_abbreviations(first_img_polygons,first_img,second_img)
    second_img_no_abbreviations = remove_small_component_and_abbreviations(cv2.bitwise_not(second_img_no_abbreviations),8)
    return second_img_no_abbreviations


def connected_to_binary(img):
    arr = img
    black_pixels_background = np.array(np.where(arr == 0))
    black_pixel_coordinates_background = list(zip(black_pixels_background[1], black_pixels_background[0]))
    connected = []
    for (row, colonna) in black_pixel_coordinates_background:
        cut_x1, cut_y1, cut_x2, cut_y2 = (
            max(row - 1, 0),
            max(colonna - 1, 0),
            min(row + 2, img.shape[1] - 1),
            min(colonna + 2, img.shape[0] - 1)
        )
        pixel_around = img[cut_y1:cut_y2, cut_x1:cut_x2]
        n_white_pix = np.sum(pixel_around == 255)
        if n_white_pix > 0:
            connected.append((row, colonna))
    return connected

def add_perimeter(threshold, grey_img, binary_img):
    component_contour = connected_to_binary(binary_img)
    for point in component_contour:
        value = grey_img[point[1], point[0]]
        if value <= threshold+5:
            grey_img[point[1], point[0]] = 0
    return binary_img

def slave_binarization(master_binary,img_gray):
    img_gray =  cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(img_gray, 5, 5, 5, 5, cv2.BORDER_REPLICATE)

    #TROVA LE COMPONENTI DENTRO LA MASTER:
    sgmt_nlabels, sgmt_labels, sgmt_stats, _ = cv2.connectedComponentsWithStats(master_binary, connectivity=8)

    final_img = np.zeros((master_binary.shape))

    # SLAVE BINARIZATION
    for i in range(1, sgmt_nlabels):
        cut_x, cut_y, cut_w, cut_h, a = sgmt_stats[i]
        if a >9:
            cut_x1, cut_y1, cut_x2, cut_y2 = (
                max(cut_x - 1, 0),
                max(cut_y - 1, 0),
                min(cut_x + cut_w +1, sgmt_labels.shape[1] - 1),
                min(cut_y + cut_h + 1, sgmt_labels.shape[0] - 1)
            )
            gray_componenet = img_gray[cut_y1:cut_y2, cut_x1:cut_x2]

            threshold = (int(np.mean(gray_componenet)))

            ret2, th3 = cv2.threshold(gray_componenet, threshold, 255, cv2.THRESH_BINARY_INV)

            th3 = add_perimeter(threshold, gray_componenet, th3)

            final_img[cut_y1:cut_y2, cut_x1:cut_x2] = th3
    final_img = final_img.astype('uint8')
    img_polygons = utils.get_polygons(cv2.bitwise_not(final_img))
    final_img = from_page_to_lines_remove_abbreviations(img_polygons,cv2.bitwise_not(final_img),cv2.bitwise_not(final_img.copy()))

    final_img = remove_small_component_and_abbreviations(cv2.bitwise_not(final_img),8)

    return final_img

def add_8_simple(binary_word, grey_word):
    max_iteration = 3
    number_interation = 0
    binary = binary_word.copy()
    while True:

        connessi = connected_to_binary(binary)

        for punto in connessi:

            x = punto[0]
            y = punto[1]
            cut_x1, cut_y1, cut_x2, cut_y2 = (
                max(x - 3, 0),
                max(y - 3, 0),
                min(x + 4, binary_word.shape[1] - 1),
                min(y + 4, binary_word.shape[0] - 1)
            )
            around_seven_gray = grey_word[cut_y1:cut_y2, cut_x1:cut_x2]

            around_seven_binary = binary[cut_y1:cut_y2, cut_x1:cut_x2]

            arr = np.asarray(around_seven_binary)


            black_pixels_background = np.array(np.where(arr == 0))
            black_pixel_coordinates_background_windows = list(zip(black_pixels_background[1], black_pixels_background[0]))

            fore = []
            back = []
            for i in range(0,around_seven_binary.shape[1]):
                for j in range(0,around_seven_binary.shape[0]):

                    if (i,j) in black_pixel_coordinates_background_windows:

                        back.append(around_seven_gray[j,i])
                    else:

                        fore.append(around_seven_gray[j,i])
            if (len(back)>0 and len(fore)>0):
                diff_foreground = abs(grey_word[punto[1], punto[0]] - int(sum(fore) / len(fore)))
                diff_background = abs(grey_word[punto[1], punto[0]] - int(sum(back) / len(back)))


                diff_minor = min(diff_foreground,diff_background)

                if diff_minor == diff_foreground:

                    binary_word[punto[1], punto[0]] = 255

        if (np.sum(cv2.bitwise_xor(binary, binary_word) == 255)) == 0 or (number_interation == max_iteration):
            break

        else:

            number_interation = number_interation + 1
            binary = binary_word.copy()

    return binary_word