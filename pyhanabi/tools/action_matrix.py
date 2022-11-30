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
    print("starting plot")
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    print("plotting>..")
    cax = ax.matshow(mat, vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylim(0, 1)
    ax.set_title(title)

    actions = idx2action if num_cards == 5 else idx2action_3card
    assert num_cards == 5 or num_cards == 3

    ax.set_xticks(range(20))
    ax.set_xticklabels(actions)
    ax.set_yticks(range(20))
    ax.set_yticklabels(actions)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
    print("done plotting..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_game", type=int, default=10000)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--num_cards", type=int, default=5)
    args = parser.parse_args()

    #models = sorted([f for f in glob.glob("/home/wz2/off-belief-learning/pyhanabi/exps/obl2/model0.pthw")])
    models = sorted([f for f in glob.glob("/home/wz2/off-belief-learning/models/icml_OBL5/*/model0.pthw")])

    for model in tqdm(models):
        model_gen = Path(model).parts[-2]
        parent = Path(model).parts[-1]
        (Path("exps") / f"act_{model_gen}").mkdir(parents=True, exist_ok=True)
        output = f"exps/act_{model_gen}/{parent}.png"
        dataset, _, _ = create_dataset_new(model, num_game=args.num_game, num_thread=args.num_thread, num_cards=args.num_cards)
        normed_p0_p1, _ = analyze(dataset, 2, num_cards=args.num_cards)
        assert False
        plot(normed_p0_p1, "action_matrix", savefig=output, num_cards=args.num_cards)

        del dataset
        del normed_p0_p1
        gc.collect()
        gc.collect()
