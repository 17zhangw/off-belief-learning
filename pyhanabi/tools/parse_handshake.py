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
import argparse
import os
import sys
import pprint
from collections import defaultdict
import json
import numpy as np
import torch
import time

import matplotlib.pyplot as plt

plt.switch_backend("agg")

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

from create import *
import utils
import common_utils
import rela
import r2d2
import model_zoo


def create_dataset_new(
    weight_file,
    *,
    write_replay=None,
    eps=0,
    trinary=True,
    shuffle_obs=False,
    shuffle_color=False,
    num_game=1000,
    num_thread=10,
    random_start=1,
    num_cards=5,
):
    if not isinstance(weight_file, list):
        agent, config = utils.load_agent(
            weight_file,
            {
                "vdn": True,
                "boltzmann_act": False,
                "device": "cuda:0",
                "uniform_priority": True,
                "off_belief": False,
            },
        )
        runner = rela.BatchRunner(agent, "cuda:0", 100, ["act", "compute_priority"])
        runners = [runner]
        configs = [config]
    else:
        configs = []
        runners = []
        for w in weight_file:
            agent, config = utils.load_agent(
                w,
                {
                    "vdn": False,
                    "boltzmann_act": False,
                    "device": "cuda:0",
                    "uniform_priority": True,
                    "off_belief": False,
                },
            )

            configs.append(config)
            runner = rela.BatchRunner(agent, "cuda:0", 100, ["act", "compute_priority"])
            runners.append(runner)

    if write_replay is None:
        write_replay = [True for _ in runners]

    replay_buffer = rela.RNNPrioritizedReplay(
        num_game,  # args.dataset_size,
        1,  # args.seed,
        0,  # args.priority_exponent, uniform sampling
        1,  # args.priority_weight,
        0,  # args.prefetch,
    )

    actors = []
    game_per_thread = 1
    seed = 0
    vdn = len(runners) == 1
    for i in range(num_thread):
        thread_actor = []
        for player_idx, (runner, config) in enumerate(zip(runners, configs)):
            seed += 1
            actor = hanalearn.R2D2Actor(
                runner,
                num_cards,
                seed,
                config["num_player"],
                player_idx,
                [eps],
                [],  # boltzmann_temp
                vdn,  # vdn
                config["sad"],  # sad
                False,  # shuffle color
                config["hide_action"],
                trinary,
                replay_buffer if write_replay[player_idx] else None,
                1,  # mutli step, does not matter
                80,  # max_seq_len, default to 80
                0.999,  # gamma, does not matter
            )
            thread_actor.append(actor)
        actors.append([thread_actor])

    games = create_envs(
        num_thread * game_per_thread,
        1,  # seed
        config["num_player"],
        0,  # bomb
        80,  # config["max_len"],
        random_start_player=random_start,
        num_cards=num_cards,
    )
    context, threads = create_threads(num_thread, game_per_thread, actors, games)

    for runner in runners:
        runner.start()
    context.start()
    while replay_buffer.size() < num_game:
        time.sleep(0.2)

    context.pause()

    # remove extra data
    for _ in range(2):
        data, unif = replay_buffer.sample(num_game, "cpu")
        replay_buffer.update_priority(unif.detach().cpu())
        time.sleep(0.2)

    print("dataset size:", replay_buffer.size())

    scores = []
    for i in range(replay_buffer.size()):
        scores.append(replay_buffer.get(i).reward.sum())

    print(
        "done about to return, avg score (%d game): %.2f"
        % (len(scores), np.mean(scores))
    )
    return replay_buffer, agent, context


def marginalize(dataset):
    count = 0
    priv_s = None
    for i in range(dataset.size()):
        epsd = dataset.get(i)
        for t in range(int(epsd.seq_len.item()) - 1):
            if priv_s is None:
                priv_s = epsd.obs["priv_s"][0][0]
            else:
                priv_s += epsd.obs["priv_s"][0][0]
            count += 1
    return priv_s / count


def analyze(dataset, num_player=2, vdn=True, num_cards=5):
    if num_player == 2:
        p0_p1 = np.zeros((20, 20))
        if num_cards == 3:
            p0_p1 = np.zeros((12, 12))
    elif num_player == 3:
        assert False
        p0_p1 = np.zeros((30, 30))

    for i in range(dataset.size() if vdn else dataset.size() // 2):
        epsd = dataset.get(i)
        action = epsd.action["a"]

        if num_player == 2 and num_cards == 5 and action[0][0] == 20:
            action = action[:, [1, 0]]
        if num_player == 2 and num_cards == 3 and action[0][0] == 12:
            action = action[:, [1, 0]]
        if num_player == 3:
            while action[0][0] == 30:
                action = action[:, [1, 2, 0]]

        for t in range(int(epsd.seq_len.item()) - 1):
            p0 = t % num_player
            p1 = (t + 1) % num_player
            a0 = action[t][p0]  # This indexing allows to avoid no-ops with vdn
            a1 = action[t + 1][p1]
            p0_p1[a0][a1] += 1

    denom = p0_p1.sum(1, keepdims=True)
    denom[denom == 0] = 1
    normed_p0_p1 = p0_p1 / denom
    return normed_p0_p1, p0_p1


def analyze_action_distribution(dataset):
    p0 = np.zeros((20,))
    for i in range(dataset.size()):
        epsd = dataset.get(i)
        action = epsd.action["a"]
        for t in range(int(epsd.seq_len.item()) - 1):
            a = action[t].item()
            if a < 20:
                p0[a] += 1

    return p0


def transition_and_timestep(dataset):
    p0_p1 = np.zeros((20, 20, 80))
    for i in range(dataset.size()):
        epsd = dataset.get(i)
        action = epsd.action["a"]
        for t in range(int(epsd.seq_len.item()) - 1):
            if t % 2 == 0:
                a0 = int(action[t][0].item())
                a1 = int(action[t + 1][1].item())
            else:
                a0 = int(action[t][1].item())
                a1 = int(action[t + 1][0].item())

            p0_p1[a0][a1][t] += 1

    return p0_p1


idx2action = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
]

idx2action_3card = [
    "D1",
    "D2",
    "D3",
    "P1",
    "P2",
    "P3",
    "C1",
    "C2",
    "C3",
    "R1",
    "R2",
    "R3",
]
