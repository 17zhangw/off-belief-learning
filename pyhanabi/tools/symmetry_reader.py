from tqdm import tqdm
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
    import code
    code.interact(local=locals())

    #pickles = [f for f in glob.glob("exps/dual_obl2/fkpct50.0/output.pkl")]
    #res = []
    #for p in pickles:
    #    with open(p, "rb") as f:
    #        experiment = pickle.load(f)
    #        for exp in experiment:
    #            #for move in exp["old_moves"]:
    #            #    print(move.to_string())
    #            #remapper = {v:k for k, v in exp["remapper"].items()}
    #            #print(remapper)
    #            #for move in exp["played_moves"]:
    #            #    if move == "diverge":
    #            #        continue
    #            #    if move.color() != -1:
    #            #        new_color = remapper[move.color()]
    #            #        move.set_color(remapper[move.color()])
    #            #        assert move.color() == new_color
    #            #print("new Moves here")
    #            #for move in exp["played_moves"]:
    #            #    if move == "diverge":
    #            #        print("diverge")
    #            #    else:
    #            #        print(move.to_string())
    #            #assert False
    #            #print(exp["game_num"], exp["remapper"])
    #            #assert False
    #            res.append({k: v for k, v in exp.items() if "remapper" not in k and "move" not in k and "deck" not in k})
    #pd.DataFrame(res).to_feather("out_obl2.feather")
