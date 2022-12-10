import glob
from tqdm import tqdm
from pathlib import Path
import pickle
import argparse
import getpass
import os
import sys
import logging
import copy

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import set_path
set_path.append_sys_path()

import torch
import rela
import hanalearn
import utils
import common_utils
import random
import r2d2
import glob
import pandas as pd
import pickle

from tools.symmetry_analysis import GameState

def diff_lists(l1, l2):
    if len(l1) != len(l2):
        return False

    for i, x in enumerate(l1):
        if l2[i] != x:
            return False

    return True

def diff_base_replay(base_replay):
    base = base_replay[0]
    m = [m.to_string() for m in base["played_moves"]]
    for replay in base_replay[1:]:
        r = [r.to_string() for r in replay["played_moves"]]
        if not diff_lists(m, r):
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--input-dir", type=str)
    args = parser.parse_args()

    with open(args.output, "w") as output:
        game_dirs = sorted([Path(f).parent for f in Path(args.input_dir).rglob("base")])
        for game_dir in tqdm(game_dirs):
            output.write(str(game_dir) + "\n")
            output.write("base:")
            with open(f"{game_dir}/base", "rb") as f:
                game = pickle.load(f)
                output.write(f"Life: {game.life}\n")
                output.write(f"Info: {game.info}\n")
                output.write(f"Score: {game.score}\n")

            output.write("base_replay:")
            ft = ["base_replay", "intervention_25.0_replay", "intervention_50.0_replay", "intervention_75.0_replay"]
            for fst in tqdm(ft, leave=False):
                with open(f"{game_dir}/{fst}", "rb") as f:
                    game = pickle.load(f)
                    game = pd.DataFrame(game)
                    output.write(fst + "\n")
                    output.write(game[["recolor_life_minus_base", "recolor_info_minus_base", "recolor_score_minus_base"]].describe().to_string())
                    output.write("\n")
            output.flush()
