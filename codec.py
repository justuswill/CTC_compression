import os
import sys
import argparse
import numpy as np

import torch
# from setuptools.sandbox import save_path

from models import model_CTC
from utils import path2torch, torch2img, psnr

torch.autograd.set_detect_anomaly(True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("--mode", type=str, choices=["enc", "dec"], default="enc")
    parser.add_argument("--save-path", type=str, default="results")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--recon-level", type=int, choices=list(range(1, 161)), default=160)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--chunk_id", type=int, default=-1)

    args = parser.parse_args(argv)
    return args


def _enc(args, net):
    x = path2torch(args.input_file).to(args.device)
    save_path_enc = os.path.join(args.save_path, "bits")
    if not os.path.exists(save_path_enc): os.mkdir(save_path_enc)
    net.encode_and_save_bitstreams_ctc(x, save_path_enc)


def _dec(args, net):
    if hasattr(args, 'bits_path'):
        save_path_dec = args.save_path
        if not os.path.exists(save_path_dec): os.mkdir(save_path_dec)
        dec_time, x_rec, bpp = net.reconstruct_ctc(args)
        torch2img(x_rec).save(f"{save_path_dec}/{os.path.basename(args.input_file)}")
    else:
        save_path_dec = os.path.join(args.save_path, "recon")
        if not os.path.exists(save_path_dec): os.mkdir(save_path_dec)
        dec_time, x_rec, bpp = net.reconstruct_ctc(args)
        torch2img(x_rec).save(f"{save_path_dec}/q{args.recon_level:04d}.png")

    print(f"dec time: {dec_time:.3f}, bpp: {bpp:.5f}", end=" ")

    if args.input_file is not None:
        x_in = path2torch(args.input_file).to(args.device)
        metric = psnr(x_in, x_rec)
        print(f", psnr: {metric:.4f}")

    return bpp, metric


def test(argv):
    args = parse_args(argv)
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = model_CTC(N=192).to(args.device)
    ckpt = torch.load("ctc.pt")["state_dict"]
    net.load_state_dict(ckpt)
    net.update()

    if args.mode == "enc":
        _enc(args, net)

    elif args.mode == "dec":
        _dec(args, net)

    else:
        raise ValueError(f"{args.mode} error: choose 'enc' or 'dec'.")


if __name__ == "__main__":
    # test(sys.argv[1:])

    args = parse_args(sys.argv[1:])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model_CTC(N=192).to(args.device)
    ckpt = torch.load("ctc.pt")["state_dict"]
    net.load_state_dict(ckpt)
    net.update()

    # qs = [1, 9, 17, 33, 65, 95, 129, 145, 160]
    qs = [1, 17, 33, 65, 129, 160]
    # qs = range(160, 0, -1)
    bpps = [[] for _ in range(len(qs))]
    psnrs = [[] for _ in range(len(qs))]
    # data_root = '/mnt/c/Users/Justus/PycharmProjects/compress_uqdm/torch_datasets/imagenet_png/data'
    data_root = '/home/jcwill/PycharmProjects/compress_uqdm/torch_datasets/imagenet_png/data'
    img_idxs = sorted([int(f.split('.')[0]) for f in os.listdir(data_root)])
    for img in img_idxs[10*args.chunk_id:10*(args.chunk_id+1)] if args.chunk_id >= 0 else img_idxs:
    # for img in ['sample/sample.png']:
        print(img)
        try:
            # blacklist = [226, 1004, 1094, 1101, 1431, 1460, 1912, 2110, 2258, 2765, 3075, 3384, 3507, 3930]
            # 32, 365
            blacklist = []
            if img in blacklist:
                continue
            args.input_file = data_root + '/%s.png' % img
            args.bits_path = 'imagenetpng/bits/%s' % img
            args.save_path = 'imagenetpng/bits/%s' % img
            # args.input_file = img
            # args.save_path = img[:-4]
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)

            if not os.path.exists(args.bits_path + '/bits/z.bin'):
                _enc(args, net)
            for q in range(len(qs)):
                q_path = 'imagenetpng/q_%d' % qs[q]
                if os.path.exists(q_path + '/%s.png' % img):
                    continue
                elif not os.path.exists(q_path):
                    os.mkdir(q_path)
                args.save_path = q_path
                args.recon_level = qs[q]
                bpp, psnr_ = _dec(args, net)
                bpps[q] += [bpp]
                psnrs[q] += [psnr_]
        except Exception as e:
            print('Error on image %s: \n %s' % (img, e))

    bpps = [np.mean(bpps[q]) for q in range(len(qs))]
    psnrs = [np.mean(psnrs[q]) for q in range(len(qs))]
    np.savez_compressed('cdc.npz', psnrs=psnrs, bpps=bpps)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(bpps, psnrs)
    ax.set(xlabel='bpp', ylabel='psnr', title='bpp vs psnr')
    ax.grid()
    fig.savefig("bpp_vs_psnr.png")
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(bpps[:3], fids[:3])
    # ax.plot(bpps[3:], fids[3:])
    # ax.set(xlabel='bpp', ylabel='fid', title='bpp vs fid')
    # ax.grid()
    # fig.savefig("../results/bpp_vs_fid.png")
    # plt.show()

    # srun --nodes=1 --ntasks=1 --mem=12G --time=3-00 --gres=gpu:1 -p ava_m.p -w "ava-m0" --pty /bin/bash --login
    # for i in {501..1000}; do /home/jcwill/miniconda/envs/UQDM-local/bin/python codec.py --chunk_id ${i}; done