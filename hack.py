from itertools import product, cycle
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def extract_grid_images(nx, ny, x_start, lx, y_start, ly, dx, dy, X, plot_debug=False):
    coords = np.empty((nx, ny), dtype=object)
    for i, j in np.ndindex(coords.shape):
        coords[i, j] = (x_start + i * lx, y_start + j * ly)

    # extract subimages
    images = np.empty((nx, ny, dx, dy), dtype=float)
    for i, j in np.ndindex(coords.shape):
        x, y = coords[i, j]
        images[i, j] = X[x:x + dx, y:y + dy]

    # show all images
    if plot_debug:
        fig, axes = plt.subplots(nx, ny)
        for i, j in product(range(ny), range(ny)):
            axes[i, j].imshow(images[i, j].T)

    return images


def build_reference(images, labels_coord):
    reference = {l: images[i,j] for l, (i, j) in labels_coord.items()}
    # normalize
    for v in reference.values():
        v -= np.mean(v)
        v /= np.std(v)
    return reference


def save_reference(reference, ref_name):
    with open(ref_name, 'wb') as f:
        pickle.dump(reference, f)


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
    if norm_corr > 0.15:
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

# plt.imshow(X)
X = parse_file(r'ref.png')


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


M = parse_image_matrix(images=images,
                       reference=reference)

print(M.T)

# Build target
nx = 3
ny = 3

x_start = 844
lx = 42
dx = 26
y_start = 347
ly = 71
dy = 19
images = extract_grid_images(nx, ny, x_start, lx, y_start, ly, dx, dy, X, plot_debug=False)

T = parse_image_matrix(images=images,
                       reference=reference)
print(T.T)


def compute_buffer_length(X):
    l = X[810:1270, 212]
    peaks_idx = np.where(l >= 0.95 * np.max(l))[0]
    n_buffer = int(round((peaks_idx[-1] - peaks_idx[0]) / 45))
    return n_buffer


# compute buffer length
n_buffer = compute_buffer_length(X)
print(f'n_buffer: {n_buffer}')


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
    if l1 > l2:  #`l1` must be <= `l2` for `lst1` to be a subsequence of `lst2`.
        return False
    i = j = 0
    d1, d2 = l1, l2
    while i < l1 and j < l2:
        while lst1[i] != lst2[j]:
            j += 1
            d2 -= 1
            if d1 > d2:  #At this point, `lst1` cannot a subsequence of `lst2`.
                return False
        i, j, d1, d2 = i+1, j+1, d1-1, d2-1
        if d1 > d2:
            return False
    return True


def gain(x: np.array, M: np.array, T: np.array, debug=False) -> float:
    """Compute gain associated to x.
    x : [row sel idx, col sel idx, ...]
    """
    if len(x) == 0:
        return 0.

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
    # V is the symbol selection associated to x
    if debug:
        print(v)

    # compute associated gain
    value = 0
    for t_idx, t in enumerate(T.T):
        t = [tt for tt in t if tt is not None]  # filter None

        # check if t is a subsequence of v
        match = is_subsequence(t, v)

        if match:
            value += (t_idx+1)
    return value

assert gain([1,2,2, 1, 4, 4], M, T) == 6


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
    av_cols = set(range(M.shape[1]))

    x = []  # solution found so far
    g = 0  # total gain

    def best_sol(x, av_rows, av_cols, n):
        """Find the best sol starting from x and potentially n additional tokens chosen form av_rows, av_cols

        """
        if n == 0:
            g = gain(x=x, M=M, T=T)
            return g, x

        if len(x) % 2 == 0:  # select row index
            av_indexes = av_rows
        else:
            av_indexes = av_cols

        g_best = 0
        x_best = None
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
                                      n=n-1)
            if g_cand > g_best:
                g_best = g_cand
                x_best = x_cand

        return g_best, x_best

    g, x_opt = best_sol(x, av_rows=av_rows, av_cols=av_cols, n=n_buffer)

    return x_opt, g

# find optimal trajectory
x_opt, g = find_best_path(M=M, T=T, n_buffer=n_buffer)

for o, i in zip(cycle('CR'), x_opt):
    print(f'{o}{i+1}, ')

gain(x=x_opt, M=M, T=T, debug=True)

plt.show()


def build_ref():
    # build the reference
    filename = 'ref.png'
    X = parse_file(filename)
    reference = build_reference(images=images,
                                labels_coord={'bd': (0, 0),
                                              '55': (0, 1),
                                              '1c': (0, 2),
                                              'e9': (2, 0)})

    save_reference(reference=reference,
                   ref_name='reference.pickle')


if __name__ == '__main__':
    build_ref()