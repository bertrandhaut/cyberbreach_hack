import logging

logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
from itertools import product, cycle
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def plot_images(images):
    nx = images.shape[0]
    ny = images.shape[1]

    fig, axes = plt.subplots(nx, ny)
    for i, j in product(range(nx), range(ny)):
        if nx == 1 or ny == 1:
            ax = axes[j]
        else:
            ax = axes[i, j]
        ax.imshow(images[i, j].T)


def extract_grid_images(nx, ny, x_start, lx, y_start, ly, dx, dy, X):
    coords = np.empty((nx, ny), dtype=object)
    for i, j in np.ndindex(coords.shape):
        coords[i, j] = (x_start + i * lx, y_start + j * ly)

    # extract subimages
    images = np.empty((nx, ny, dx, dy), dtype=float)
    for i, j in np.ndindex(coords.shape):
        x, y = coords[i, j]
        images[i, j] = X[x:x + dx, y:y + dy]

    return images


def build_reference_from_images(images, labels_coord):
    reference = {l: images[i, j] for l, (i, j) in labels_coord.items()}
    # normalize
    for v in reference.values():
        v -= np.mean(v)
        v /= np.std(v)
    return reference


REFERENCE_FILENAME = Path(__file__).parent.parent / 'data/reference.pickle'


def save_reference(reference):
    with open(REFERENCE_FILENAME, 'wb') as f:
        pickle.dump(reference, f)


def load_references():
    with open(REFERENCE_FILENAME, 'rb') as f:
        references = pickle.load(f)
    return references


def build_references():
    # build the reference
    filename = Path(__file__).parent.parent / 'data/ref.png'
    X = parse_file(filename)
    images = extract_M_images_from_X(X)
    reference = build_reference_from_images(images=images,
                                            labels_coord={'bd': (0, 0),
                                                          '55': (0, 1),
                                                          '1c': (0, 2),
                                                          'e9': (2, 0)})
    save_reference(reference=reference)


def find_best_match(im, reference):
    im -= np.mean(im)
    im /= np.std(im)

    ref_size = list(reference.values())[0].shape

    # target size in function of im
    if im.shape != ref_size:
        img = Image.fromarray(im)
        img = img.resize(size=(ref_size[1], ref_size[0]))
        im = np.asarray(img)

    best_label, best_corr = None, -float('inf')
    for k, v in reference.items():
        corr = np.sum(im * v)
        if corr > best_corr:
            best_corr = corr
            best_label = k
    norm_corr = best_corr / (ref_size[0] * ref_size[1])

    logger.debug(f'Found {best_label} with a best correlation of {norm_corr}')
    if norm_corr > 0.2:
        return best_label
    else:
        return None


def parse_image_matrix(images, reference):
    # identify all elements of images
    nx, ny = images.shape[:2]
    labels = np.empty((nx, ny), dtype=object)
    for i, j in product(range(ny), range(ny)):
        im = images[i, j]
        label = find_best_match(im, reference)
        labels[i, j] = label
    return labels


def parse_file(filename):
    image = Image.open(filename)
    X = np.asarray(image)

    # grey scale
    X = np.mean(X, axis=2).T
    return X


def extract_M_images_from_X(X):
    # grid extract
    nx = 5
    ny = 5
    x_start = 347
    lx = 64
    y_start = 366
    ly = 64
    dx = 30
    dy = 20

    images = extract_grid_images(nx, ny, x_start, lx, y_start, ly, dx, dy, X)
    return images


def extract_T_images_from_X(X):
    """
    T position are different for 1 target and "2 or more" targets

    :param X:
    :return:
    """
    only_one = X[1170, 436] > 30

    x_start = 844
    lx = 42
    dx = 26
    ly = 71
    dy = 19
    if only_one:
        nx = 3  #TODO managed correclty nx/ny
        ny = 3
        y_start = 361
    else:
        nx = 3
        ny = 3
        y_start = 347
    images = extract_grid_images(nx, ny, x_start, lx, y_start, ly, dx, dy, X)

    # plot_images(images)
    return images


def extract_M_from_X(X, references):
    images = extract_M_images_from_X(X=X)
    M = parse_image_matrix(images=images,
                           reference=references)
    return M


def extract_T_from_X(X, references):
    images = extract_T_images_from_X(X=X)
    T = parse_image_matrix(images=images,
                           reference=references)
    # a column of full None --> is not a real line
    to_drop = np.all(T == None, axis=0)
    T = T[:, ~to_drop]
    assert T.shape[1] > 0  # at least 1 target

    return T


def compute_buffer_length(X):
    l = X[810:1270, 212]
    peaks_idx = np.where(l >= 0.95 * np.max(l))[0]
    n_buffer = int(round((peaks_idx[-1] - peaks_idx[0]) / 45))
    return n_buffer


def is_subsequence(lst1, lst2):
    """
        *   Finds if a list is a subsequence of another.

        *   Params:
            *   `lst1` (`list`): The candidate subsequence.
            *   `lst2` (`list`): The parent list.
        *   Return:
            *   (`bool`): A boolean variable indicating whether `lst1` is a subsequence of `lst2`.
    """
    # https://codereview.stackexchange.com/questions/215324/find-if-one-list-is-a-subsequence-of-another

    l1, l2 = len(lst1), len(lst2)
    if l1 > l2:  # `l1` must be <= `l2` for `lst1` to be a subsequence of `lst2`.
        return False
    i = j = 0
    d1, d2 = l1, l2
    while i < l1 and j < l2:
        while lst1[i] != lst2[j]:
            j += 1
            d2 -= 1
            if d1 > d2:  # At this point, `lst1` cannot a subsequence of `lst2`.
                return False
        i, j, d1, d2 = i + 1, j + 1, d1 - 1, d2 - 1
        if d1 > d2:
            return False
    return True


def convert_x_to_symbol(x, M):
    c = 0
    r = x[0]
    v = [M[r, c]]

    col_sel = True
    for i in x[1:]:
        if col_sel:
            c = i
        else:
            r = i
        v.append(M[r, c])
        col_sel = not col_sel
    return v


def gain(x: np.array, M: np.array, T: np.array) -> float:
    """Compute gain associated to x.
    x : [row sel idx, col sel idx, ...]
    """
    if len(x) == 0:
        return 0.

    v = convert_x_to_symbol(x=x, M=M)

    # compute associated gain
    value = 0
    for t_idx, t in enumerate(T.T):
        t = [tt for tt in t if tt is not None]  # filter None

        # check if t is a subsequence of v
        match = is_subsequence(t, v)

        if match:
            value += (t_idx + 1)
    return value - len(x) * 0.001


# assert gain([1,2,2, 1, 4, 4], M, T) == 6


def find_best_path(M, T, n_buffer):
    """
    Find best path of length n_buffer in M:
    - row idx in first col
    - col idx in selected row
    - ...

    Gain given by 3 * "col[2] of T satisfied" + 2 * "col[1] of T satisfied" +  "col[0] of T satisfied"

    :param np.array M:
    :param np.array T:
    :param float n_buffer:
    :return: optimal path
    """

    # first implementation, brute force approach
    av_rows = set(range(M.shape[0]))  # available choice among rows indexes
    av_cols = set(range(M.shape[1])) - {0}  # first column already selected

    x = []  # solution found so far
    g = 0  # total gain

    def best_sol(x, av_rows, av_cols, n):
        """Find the best sol starting from x and potentially n additional tokens chosen form av_rows, av_cols

        """
        actual_gain = gain(x=x, M=M, T=T)
        if n == 0:
            return actual_gain, x

        if len(x) % 2 == 0:  # select row index
            av_indexes = av_rows
        else:
            av_indexes = av_cols

        g_best = actual_gain
        x_best = x
        for idx in av_indexes:  # try select
            if len(x) % 2 == 0:
                av_rows = av_rows.copy()
                av_rows.remove(idx)
            else:
                av_cols = av_cols.copy()
                av_cols.remove(idx)

            x_cand = x.copy()
            x_cand.append(idx)
            g_cand, x_cand = best_sol(x=x_cand,
                                      av_rows=av_rows,
                                      av_cols=av_cols,
                                      n=n - 1)
            if g_cand > g_best:
                g_best = g_cand
                x_best = x_cand

        return g_best, x_best

    g, x_opt = best_sol(x, av_rows=av_rows, av_cols=av_cols, n=n_buffer)

    return x_opt, g


def analyze_file(filename):
    references = load_references()

    X = parse_file(filename)
    M = extract_M_from_X(X, references=references)
    logger.info(f'M:\n{M.T}')
    T = extract_T_from_X(X, references=references)
    logger.info(f'T:\n{T.T}')

    n_buffer = compute_buffer_length(X)
    logger.info(f'n_buffer: {n_buffer}')

    # find optimal trajectory
    x_opt, g = find_best_path(M=M, T=T, n_buffer=n_buffer)

    x_opt_str = ', '.join(f'{o}{i + 1}' for o, i in zip(cycle('CR'), x_opt))
    logger.info(x_opt_str)

    v = convert_x_to_symbol(x=x_opt, M=M)
    logger.info(f'Symbol selection: {v}')
    logger.info(f'gain: {g}')

    return x_opt_str, g


if __name__ == '__main__':
    # build_references()

    # analyze_file('../data/ref.png')
    analyze_file('../tests/data/3.png')

    plt.show()
