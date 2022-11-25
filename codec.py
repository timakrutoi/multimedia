#!/usr/bin/python3
#encode=utf-8
from scipy.fft import dctn, idctn
import numpy as np

import av
import matplotlib.pyplot as plt
from my_utils import *
from argparse import ArgumentParser

from utils.dct import dct_2d
from utils.idct import idct_2d

from tqdm import tqdm


def get_quant_table(R, s):
    res = np.empty(s)
    W, H = s[0], s[1]
    for i in range(W):
        for j in range(H):
            if len(s) == 3:
                res[i, j, :] = 1 + (i%8 + j%8) * R
            else:
                res[i, j] = 10 + (i%8 + j%8) * R
    return res


def quantization(G, m):
    # return (np.divide(G, m).round().astype(np.float64))
    return (G / m).round()#.astype('uint8')


def dequantization(G, m):

    return (np.multiply(G, m))


def my_dct_3d(blocks):
    """Discrete Cosine Transform 3D"""
    return dct(dct(dct(blocks, axis=0), axis=1), axis=2)


def my_idct_3d(blocks):
    """Inverse Discrete Cosine Transform 3D"""
    return idct(idct(idct(blocks, axis=0), axis=1), axis=2)


def my_dct_2d(blocks):
    """Discrete Cosine Transform 2D"""
    return dct(dct(blocks.T, axis=0).T, axis=1)


def my_idct_2d(blocks):
    """Inverse Discrete Cosine Transform 2D"""
    return idct(idct(blocks.T, axis=0).T, axis=1)


def get_score(cur, ref, r=1):

    # return np.sum((cur - ref)**2)

    return np.sum((np.abs(cur - ref) < r))


def motion_estimation(cur, ref, r, x_size=16, y_size=16, dx=16, dy=16, dx1=4, dy1=4):
    W, H = cur.shape[0], cur.shape[1]

    res = []

    for x_s in range(0, W, dx):
        # print(f'stage {x_s / W}')
        res.append([])
        for y_s in range(0, H, dy):
            cur_area = ref[x_s:x_s+x_size, y_s:y_s+y_size]

            best_score = None

            for x in range(0, W, dx):
                for y in range(0, H, dy):
                    try:
                        score = get_score(cur_area, cur[x:x+x_size, y:y+y_size], r)
                        # print(score)
                    except ValueError:
                        pass
                    cur_score = (x, y, score)
                    # score_map[x, y] = score
                    try:
                        best_score = best_score if best_score[2] < cur_score[2] else cur_score
                    except TypeError:
                        best_score = cur_score

                    if best_score[2] == 0:
                        break

                if best_score[2] == 0:
                    break

            for x in range(x_s - dx, x_s + dx + 1, dx1):
                for y in range(y_s - dy, y_s + dy + 1, dy1):
                    try:
                        score = get_score(cur_area, cur[x:x+x_size, y:y+y_size], r)
                    except ValueError:
                        pass
                    cur_score = (x, y, score)
                    # score_map[x, y] = score
                    try:
                        best_score = best_score if best_score[2] <= cur_score[2] else cur_score
                    except TypeError:
                        best_score = cur_score

            vec = (best_score[0] - x_s, best_score[1] - y_s)
            # vec = local_search(cur_area, cur, vec[0], vec[1], x_size, y_size, best_score[2])
            res[x_s // dx].append(vec)

    return res


def local_search(ref_b, cur, x, y, x_size, y_size, min_score):
    vec = (x, y)

    for i in range(x - x_size, x + x_size):
        for j in range(y - y_size, y + y_size):
            try:
                cur_score = get_score(ref_b, cur[x:x+x_size, y:y+y_size], r)
            except ValueError:
                continue

            if cur_score < min_score:
                min_score = cur_score
                vec = (i - x, j - y)

    return vec


def motion_compensation(ref, motion_vecs, x_size=16, y_size=16):
    res = np.zeros(ref.shape)

    for i, vec_row in enumerate(motion_vecs):
        # print(vec_row)
        for j, vec in enumerate(vec_row):
            x, y = i*x_size, j*y_size
            xr, yr = i*x_size + vec[0], j*y_size + vec[1]

            try:
                try:
                    res[x:x+x_size, y:y+y_size, :] = ref[xr:xr+x_size, yr:yr+y_size, :]
                except IndexError:
                    res[x:x+x_size, y:y+y_size] = ref[xr:xr+x_size, yr:yr+y_size]                
            except ValueError:
                continue

    return res


def bits_required(n):
    res = np.abs(n)
    try:
        res[res == 0] = 2
    except TypeError:
        if res == 0:
            return 1

    return np.log2(res).astype('int') + 1


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):

    return [item for sublist in lst for item in sublist]


def run_length_encode_old(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a LENGTH tuple
    runs = []

    # values are binary representations of array elements using SIZE bits
    levels = []

    run = 1
    level = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            runs.append(0)
            levels.append(0)
            break
        else:
            if elem == level:
                # if run == 15:
                #     runs.append(run)
                #     levels.append(level)
                #     run = 0
                #     level = elem
                run += 1
            else:
                runs.append(run)
                levels.append(level)
                run = 1
                level = elem

    return runs, levels


def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a LENGTH tuple
    runs = []

    # values are binary representations of array elements using SIZE bits
    levels = []

    run = 0
    level = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero+1:
            runs.append(0)
            levels.append(0)
            break
        else:
            if elem == level:
                # if run == 15:
                #     runs.append(run)
                #     levels.append(level)
                #     run = 0
                #     level = elem
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                    levels.append(level)
                run = 1
                level = elem

    return runs, levels


def get_bits(frame, vecs):
    b1, b2 = 0, 0

    x, y = frame.shape[0], frame.shape[1]

    zigzag = [#(0, 0), 
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6), (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

    last_dc = 0
    dc = []

    ac = 0

    for i in range(0, x, 8):
        for j in range(0, y, 8):
            block = frame[i:i+8, j:j+8]
            dc.append(block[0, 0] - last_dc)
            last_dc = block[0, 0]
            ac_s = [block[i] for i in zigzag]
            run, level = run_length_encode(ac_s)
            # if block[0, 0] == 131:
            #     print(ac_s)
            #     print(list(zip(run, level)))
            #     print((get_H(np.array(run)) + get_H(bits_required(level))) * len(run) + sum(bits_required(level)))
            ac += (get_H(np.array(run)) + get_H(bits_required(level))) * len(run) + sum(bits_required(level))
            # exit()

    b1 = get_H(bits_required(dc)) * len(np.array(dc)) + sum(bits_required(dc)) + ac
    b2 = get_H(np.array(flatten(vecs))) * len(flatten(vecs))

    # print(vecs[0])
    # print(b1, b2)

    return b1 + b2


def proccess_frame(cur, ref, q_t, r=1, plot_frame=False, jpeg=False):
    if not jpeg:
        motion_vec = motion_estimation(cur, ref, r)
        comp_frame = motion_compensation(ref, motion_vec)
        diff_frame = cur - comp_frame + 128

        diff_frame[diff_frame < 0] = 0
        diff_frame[diff_frame > 255] = 255
        re_frame = diff_frame.astype('uint8')
    else:
        diff_frame = cur
        re_frame = cur
        motion_vec = []

    my_dct = dctn
    my_idct = idctn

    tmp = np.empty(re_frame.shape)

    for i in range(0, re_frame.shape[0], 8):
        for j in range(0, re_frame.shape[1], 8):
            tmp[i:i+8, j:j+8] = my_dct(re_frame[i:i+8, j:j+8], norm='ortho')#, type=4)
    # a = 32
    # print(tmp[a:a+8, a:a+8])
    re_frame = quantization(tmp, q_t)

    entropy = get_bits(re_frame, motion_vec)

    # print(re_frame)
    # plt.imshow(re_frame)
    # plt.show()

    re_frame = dequantization(re_frame, q_t)
    for i in range(0, re_frame.shape[0], 8):
        for j in range(0, re_frame.shape[1], 8):
            re_frame[i:i+8, j:j+8] = my_idct(re_frame[i:i+8, j:j+8], norm='ortho')#, type=4)

    p = psnr(re_frame, diff_frame)

    if not jpeg:
        re_frame = (comp_frame + re_frame - 128)
    re_frame[re_frame < 0] = 0
    re_frame[re_frame > 255] = 255
    re_frame = re_frame.astype('uint8')

    if plot_frame:
        # print(f'PSNR = {p}')
        plt.imshow(re_frame)
        plt.title(f'PSNR = {p}, comp f = {(re_frame.size*8) / (entropy)}')
        plt.show()

    try:
        return ycbcr2rgb(re_frame), entropy
    except IndexError:
        return re_frame, entropy, p


def create_new_file(source_list, in_stream, file_name, fmt='gray8'):
    output = av.open(file_name, "w")
    out_stream = output.add_stream('mpeg4')

    for i, img in enumerate(source_list):
        frame = av.VideoFrame.from_ndarray(img, format=fmt)
        for packet in out_stream.encode(frame):
            output.mux(packet)

    for img_packet in out_stream.encode():
        output.mux(img_packet)

    output.close()


def save_diff_frames(in_file, out_file='diff_frames.avi', plotting=False):
    output = av.open(out_file, "w")
    out_stream = output.add_stream('mpeg4')

    container = av.open(in_file, )
    in_stream = container.streams.video[0]

    frames = list(container.demux(in_stream))
    last_frame = frames.pop(0)

    deviation = []

    for frame in frames:
        try:
            cur = rgb2ycbcr(np.asarray(frame.decode()[0].to_rgb().to_image()))[:,:,0]
            ref = rgb2ycbcr(np.asarray(last_frame.decode()[0].to_rgb().to_image()))[:,:,0]
        except IndexError:
            break

        f = av.VideoFrame.from_ndarray(cur - ref, format='gray8')
        last_frame = frame
        deviation.append(deviation2d(cur, ref))
        for packet in out_stream.encode(f):
            output.mux(packet)

    for img_packet in out_stream.encode():
        output.mux(img_packet)

    output.close()

    if plotting:
        plt.plot(deviation, label=in_file[len('../../vid/lr1_'):])


def proccess_video(file_name, R=10, r=1, verbose='None', plot_frame=False, flimit=-1, use_jpeg=False, channel='y'):
    container = av.open(file_name)

    in_stream = container.streams.video[0]

    frames = list(container.demux(in_stream))
    frames.pop()
    tmp = frames.pop(0)
    if flimit > 0:
        frames = frames[:flimit]
    last_frame = None
    # last_frame = frames.pop(0)
    template = np.asarray(tmp.decode()[0].to_rgb().to_image())
    fmt = 'rgb24'

    if channel == 'y':
        template = template[:,:,0]
        fmt = 'gray8'

    quant_table = get_quant_table(R, template.shape)

    new_frames = []

    ent = 0
    psnr_ = 0

    if verbose == 'all':
        it = tqdm(frames)
    else:
        it = frames

    for i, frame in enumerate(it):
        cur = rgb2ycbcr(np.asarray(frame.decode()[0].to_rgb().to_image()))[:,:,0]

        if last_frame is not None:
            ref = rgb2ycbcr(np.asarray(last_frame.decode()[0].to_rgb().to_image()))[:,:,0]
            res, e, p = proccess_frame(cur, ref, quant_table, r, plot_frame, use_jpeg)

        else:
            res, e, p = proccess_frame(cur, None, quant_table, r, plot_frame, jpeg=True)
        
        new_frames.append(res)
        ent += e
        psnr_ += p
        last_frame = frame
    try:
        it.close()
    except AttributeError:
        pass

    # print(template.shape)
    if verbose == 'all':
        print(f'Total orig size: {len(new_frames) * template.size * 8}')
        print(f'Total comp size: {ent}')
        print(f'Compress factor: {(len(new_frames) * template.size * 8) / ent}')
        print(f'Average psnr   : {psnr_ / len(new_frames)}')

    if verbose == 'stats':
        print(R, (len(new_frames) * template.size * 8) / ent, psnr_ / len(new_frames))


    create_new_file(new_frames, in_stream, 'converted/diff_frames.avi', fmt)


if __name__ == '__main__':
    # add task : make plots with different parameter of get_score(R) for all

    p = ArgumentParser()
    p.add_argument('-f', '--file', default=0, type=int, help='File index')
    p.add_argument('-q', default=1, type=int, help='Quantize factor')
    p.add_argument('-r', default=1, type=int, help='Score metric factor')
    p.add_argument('-j', '--use-jpeg', action='store_true', default=False, help='Use only jpeg coding')
    p.add_argument('-v', choices=['none', 'stats', 'all'], default='None', help='Verbosity level')
    p.add_argument('-p', action='store_true', default=False, help='Plot every frame')
    p.add_argument('-c', type=int, default=-1, help='Limit of frames to code')
    a = p.parse_args()

    name = '../vid/re_vid.avi'
    name = ['../vid/lr1_1.AVI', '../vid/lr1_2.AVI', '../vid/lr1_3.avi']
    # name = '../vid/lr1_3.avi'

    proccess_video(name[a.file], a.q, a.r, a.v, a.p, a.c, a.use_jpeg)
