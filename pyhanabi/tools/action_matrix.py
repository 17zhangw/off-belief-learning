# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from tqdm import tqdm
import gc
from pathlib import Path
import glob
import os, sys
import argparse
import pprint
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import common_utils
from parse_handshake import *


plt.rc("image", cmap="viridis")
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)
plt.rc("axes", labelsize=10)
plt.rc("axes", titlesize=10)


def plot(mat, title, *, fig=None, ax=None, savefig=None, num_cards=5):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(mat, vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylim(0, 1)
    ax.set_title(title)

    actions = idx2action if num_cards == 5 else idx2action_3card
    assert num_cards == 5 or num_cards == 3
    ticks = 12 if num_cards == 3 else 20

    ax.set_xticks(range(ticks))
    ax.set_xticklabels(actions)
    ax.set_yticks(range(ticks))
    ax.set_yticklabels(actions)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_game", type=int, default=10000)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--num_cards", type=int, default=5)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    dataset, _, _ = create_dataset_new(args.model, num_game=args.num_game, num_thread=args.num_thread, num_cards=args.num_cards)
    normed_p0_p1, _ = analyze(dataset, 2, num_cards=args.num_cards)
    plot(normed_p0_p1, "action_matrix", savefig=args.output, num_cards=args.num_cards)
