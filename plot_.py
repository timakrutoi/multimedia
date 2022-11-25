#!/usr/bin/python3
#encode=utf-8
import matplotlib.pyplot as plt
import numpy as np
from os import sep


def plot(name, axs, use_jpeg):
    q = []
    comp = []
    psnr = []

    data = np.loadtxt(name)

    for i, j, k in data:
        q.append(i)
        comp.append(j)
        psnr.append(k)
    # print(q, comp, psnr)

    axs[0].plot(q, comp, label=name)
    axs[1].plot(q, psnr, label=name)
    axs[2].plot(psnr, comp, label=name)
    # plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()

    p.add_argument('--file-name', nargs='+', default=[], help='File name to plot')
    p.add_argument('-f', nargs='+', type=int, default=[-1], help='Index of film to plot')
    p.add_argument('-j', action='store_true', default=False, help='Also plot jpeg')

    a = p.parse_args()

    fig, axs = plt.subplots(1, 3)
    axs[0].set_title('Compression factor')
    axs[1].set_title('PSNR')
    axs[2].set_title('Compression factor to PSNR')

    if len(a.file_name) == 0:
        prefix = '.'
        names = ['vid0', 'vid1', 'vid2']
        files = []

        for i, j in enumerate(names):
            if a.f[0] == -1 or i in a.f:
                files.append(j)
                if a.j:
                    files.append(j + 'j')
        # print(files)


        for j in files:
            plot(sep.join([prefix, j]), axs, a.j)
    else:
        for name in a.file_name:
            plot(name, axs, a.j)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()
