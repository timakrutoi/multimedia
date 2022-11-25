import numpy as np


def corr2d(A, B):
    A_e = expectation2d(A)
    B_e = expectation2d(B)

    sig_A = deviation2d(A, A_e)
    sig_B = deviation2d(B, B_e)

    res = expectation2d((A - A_e) * (B - B_e)) / (sig_A * sig_B)

    return res


def expectation2d(v):
    H, W = v.shape[0], v.shape[1]
    res = np.sum(v) / (W * H)
    return res


def deviation2d(v, v_e=None):
    H, W = v.shape[0], v.shape[1]
    if v_e is None:
        v_e = expectation2d(v)

    res = (np.sum((v - v_e) ** 2)) / (W * H)

    res = np.sqrt(res)
    return res


def get_slices(s, size):
    slices = []
    if s < 0:
        slices.append(slice(abs(s), size))
        slices.append(slice(0, size + s))
    else:
        slices.append(slice(0, size - s))
        slices.append(slice(s, size))
    return slices


def autocorr2d(img, x, y):
    H, W = img.shape[0], img.shape[1]

    res = np.array([])

    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]

    x_array = np.array([i for i in x])
    y_array = np.array([i for i in y])

    for i in y_array:
        row = np.array([])
        for j in x_array:

            s1 = get_slices(i, H)
            s2 = get_slices(j, W)

            c1 = img[s1[0], s2[0]]
            c2 = img[s1[1], s2[1]]

            row = np.append(row, corr2d(c1, c2))
        if res.size == 0:
            res = np.append(res, row)
        else:
            res = np.vstack([res, row])

    return res.T


def correlation_between_channels(img):
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    cor_rb = corr2d(img_r, img_b)
    cor_rg = corr2d(img_r, img_g)
    cor_gb = corr2d(img_g, img_b)

    return cor_rb, cor_rg, cor_gb


def autocorr(img, x, y):
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    autocorr_r = autocorr2d(img_r, x, y)
    autocorr_g = autocorr2d(img_g, x, y)
    autocorr_b = autocorr2d(img_b, x, y)

    return autocorr_r, autocorr_g, autocorr_b


def rgb2ycbcr(img):
    new_img = np.empty(img.shape, dtype='float')
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    new_img[:, :, 0] = 0.2990 * r + 0.587 * g + 0.114 * b
    new_img[:, :, 1] = 0.5643 * (b - new_img[:, :, 0]) + 128
    new_img[:, :, 2] = 0.7132 * (r - new_img[:, :, 0]) + 128

    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0

    return new_img.astype('uint8')


def ycbcr2rgb(img):
    new_img = np.empty(img.shape, dtype='float')
    y, cb, cr = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # blue
    new_img[:, :, 0] = y + 1.772 * (cb - 128.)
    # green
    new_img[:, :, 1] = y - 0.714 * (cr - 128.) - 0.334 * (cb - 128.)
    # red
    new_img[:, :, 2] = y + 1.402 * (cr - 128.)

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    return new_img.astype('uint8')


def psnr(a, b, l=8):
    if a.shape != b.shape:
        return None

    error = np.sum((a - b)**2)
    res = a.shape[0] * a.shape[1] * (((2**l) - 1)**2) / error
    return 10 * np.log10(res)


def decimation_avg(img, channels=3, k=2):
    new_img = np.array(img)

    for i in range(1, channels):
        for x in range(0, img.shape[0], k):
            for y in range(0, img.shape[1], k):
                avg = np.sum(img[x:x+k, y:y+k, i], dtype='float')
                avg /= k**2

                new_img[x:x+k, y:y+k, i] = avg

    return new_img


def decimation_max(img, channels=3, k=2):
    new_img = np.array(img)

    for i in range(1, channels):
        for x in range(0, img.shape[0], k):
            for y in range(0, img.shape[1], k):
                new_img[x:x+k, y:y+k, i] = new_img[x, y, i]

    return new_img


def f_neig(x, r):
    if r == 1:
        return x[1, 1] - x[1, 0]
    if r == 2:
        return x[1, 1] - x[0, 1]
    if r == 3:
        return x[1, 1] - x[0, 0]
    if r == 4:
        return x[1, 1] - (x[0, 0] + x[0, 1] + x[1, 0]) / 3
    return None


def get_D(x, r):
    D = np.empty((x.shape[0] - 1, x.shape[1] - 1), dtype='int')
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            D[i, j] = f_neig(x[i:i+2, j:j+2].astype('int'), r)

    return D


def get_H(x):
    r = 0
    f = get_pr(x)
    for i in f.values():
        if i == 0:
            continue
        r += i * np.log2(i)

    return -int(r) + 1


def get_pr(x):
    # res = np.zeros(256, dtype='float')
    res = {}

    for i in x.flat:
        # print(i)
        try:
            res[i] += 1
        except KeyError:
            res[i] = 1

    for k in res.keys():
        res[k] /= x.size

    return res


def get_subframe(_y, i=0, j=0):
    res = np.empty((int(_y.shape[0]/2), int(_y.shape[1]/2)))

    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            res[x, y] = _y[2*x + i, 2*y + j]

    return res.astype('uint8')
