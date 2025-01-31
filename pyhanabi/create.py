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
import set_path

set_path.append_sys_path()

import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


def create_envs(
    num_env,
    seed,
    num_player,
    bomb,
    max_len,
    *,
    num_cards=5,
    random_start_player=1,
):
    games = []
    for game_idx in range(num_env):
        params = {
            "players": str(num_player),
            "seed": str(seed + game_idx),
            "bomb": str(bomb),
            "colors": str(num_cards),
            "ranks": str(num_cards),
            "hand_size": str(num_cards),
            "random_start_player": str(random_start_player),
        }
        game = hanalearn.HanabiEnv(
            params,
            max_len,
            False,
        )
        games.append(game)
    return games


def flatten(s):
    if s == []:
        return s
    if isinstance(s[0], list):
        return flatten(s[0]) + flatten(s[1:])
    return s[:1] + flatten(s[1:])


def create_threads(num_thread, num_game_per_thread, actors, games):
    context = rela.Context()
    threads = []
    for thread_idx in range(num_thread):
        envs = games[
            thread_idx * num_game_per_thread : (thread_idx + 1) * num_game_per_thread
        ]
        thread = hanalearn.HanabiThreadLoop(envs, actors[thread_idx], False)
        threads.append(thread)
        context.push_thread_loop(thread)
    print(
        "Finished creating %d threads with %d games and %d actors"
        % (len(threads), len(games), len(flatten(actors)))
    )
    return context, threads
