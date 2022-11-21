from pathlib import Path
import itertools
import glob
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

class GameState(object):
    def __init__(self, params, game_moves, game_deck, life, info, score):
        self.params = params
        self.game_moves = game_moves
        self.game_deck = game_deck
        self.life = life
        self.info = info
        self.score = score

    params = None
    # list[hanalearn.HanabiMove]
    game_moves = None
    # list[hanalearn.HanabiCardValue]
    game_deck = None

    life = None
    info = None
    score = None


def run_game(p1_model, p2_model):
    # Load the agents.
    default_override = {"device": "cuda:0", "vdn": False}
    p1_agent, _ = utils.load_agent(p1_model, default_override)
    p2_agent, _ = utils.load_agent(p2_model, default_override)

    p1_agent.train(False)
    p2_agent.train(False)
    assert isinstance(p1_agent, r2d2.R2D2Agent)
    assert isinstance(p2_agent, r2d2.R2D2Agent)
    agents = [p1_agent, p2_agent]

    params = {
        "players": str(len(agents)),
        #"seed": str(-1),
        "colors": "5", # 5 colors.
        "ranks": "5", # 5 ranks.
        "hand_size": "5", # 5 cards hand size.
        "max_information_tokens": "8", # 8 observation tokens to start.
        "max_life_tokens": "3", # 3 life tokens to start.
        "bomb": str(0), # Start with 0 bombs.
        "random_start_player": str(0), # No random start.
    }
    game = hanalearn.HanabiEnv(params, -1, False)  # max_len  # verbose
    game.reset()

    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            h[k] = v.cuda().unsqueeze(0) # add batch dim

    moves = []
    while not game.terminated():
        actions = []
        new_hids = []

        for i, (agent, hid) in enumerate(zip(agents, hids)):
            # Note: argument here is (game_state, player_idx, hide_action)
            # make sure to specify the correct hide_action value
            obs = hanalearn.observe(game.get_hle_state(), i, False)

            priv_s = obs["priv_s"].cuda().unsqueeze(0)
            publ_s = obs["publ_s"].cuda().unsqueeze(0)
            legal_move = obs["legal_move"].cuda().unsqueeze(0)

            action, new_hid = agent.greedy_act(priv_s, publ_s, legal_move, hid)
            if i == 0:
                actions.append([action.item()])
            else:
                actions[-1].append(action.item())
            new_hids.append(new_hid)

        hids = new_hids
        cur_player = game.get_current_player()
        move = game.get_move(actions[-1][cur_player])

        moves.append(move)
        game.step(move)

    history = game.move_history()
    moves = []
    for move in history:
        moves.append(move.move)

    deck_card_history = game.deck_card_history()
    original_deck = [d for d in reversed(deck_card_history)]
    return GameState(params=params, game_moves=moves, game_deck=original_deck, life=game.get_life(), info=game.get_info(), score = game.get_score())


def replay_game(p1_model, p2_model, game_state, divergence_point):
    #### NOTE: the divergence_point is 1-indexed.

    # Load the agents.
    default_override = {"device": "cuda:0", "vdn": False}
    p1_agent, _ = utils.load_agent(p1_model, default_override)
    p2_agent, _ = utils.load_agent(p2_model, default_override)
    p1_agent.train(False)
    p2_agent.train(False)
    assert isinstance(p1_agent, r2d2.R2D2Agent)
    assert isinstance(p2_agent, r2d2.R2D2Agent)
    assert divergence_point <= len(game_state.game_moves)
    agents = [p1_agent, p2_agent]

    old_deck = game_state.game_deck
    old_actions = game_state.game_moves

    def recolor():
        color = [0, 1, 2, 3, 4]
        while color == [0, 1, 2, 3, 4]:
            random.shuffle(color)
        remapper = {i: color[i] for i in range(0, 5)}

        remapped_deck = []
        for card in game_state.game_deck:
            remapped_deck.append(hanalearn.HanabiCardValue(remapper[card.color()], card.rank()))
        game_state.game_deck = remapped_deck

        remapped_actions = []
        for action in game_state.game_moves:
            remapped_actions.append(hanalearn.HanabiMove(
                action.move_type(),
                action.card_index(),
                action.target_offset(),
                -1 if action.color() == -1 else remapper[action.color()],
                action.rank()))
        game_state.game_moves = remapped_actions
        return remapper

    remapper = recolor()

    # Load the deck with recolored cards.
    game = hanalearn.HanabiEnv(game_state.params, -1, False)  # max_len  # verbose
    game.reset_with_deck_no_chance(game_state.game_deck)

    # Make sure that we still deal cards in the right order.
    # 10 is so that we deal through the "initial" hand loading.
    deal_after_diverge = []
    deck_deal = []
    for move in game_state.game_moves[10+divergence_point-1:]:
        if move.move_type() == hanalearn.MoveType.Deal:
            deal_after_diverge.append(move)

    num_deal = len([c for c in game_state.game_moves if c.move_type() == hanalearn.MoveType.Deal])
    for card in reversed(game_state.game_deck[:-num_deal]):
        # Add moves to deal the remaining deck.
        deck_deal.append(hanalearn.HanabiMove(
            hanalearn.MoveType.Deal,
            -1,
            -1,
            card.color(),
            card.rank()))
    game_state.game_moves = game_state.game_moves[:10+divergence_point-1]

    # Seed the initial h0.
    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            h[k] = v.cuda().unsqueeze(0) # add batch dim

    seed_diverged = False
    seeded = False
    played_moves = []
    while not game.terminated():
        def advance():
            actions = []
            new_hids = []
            for i, (agent, hid) in enumerate(zip(agents, hids)):
                # Note: argument here is (game_state, player_idx, hide_action)
                # make sure to specify the correct hide_action value
                obs = hanalearn.observe(game.get_hle_state(), i, False)

                priv_s = obs["priv_s"].cuda().unsqueeze(0)
                publ_s = obs["publ_s"].cuda().unsqueeze(0)
                legal_move = obs["legal_move"].cuda().unsqueeze(0)

                action, new_hid = agent.greedy_act(priv_s, publ_s, legal_move, hid)
                if i == 0:
                    actions.append([action.item()])
                else:
                    actions[-1].append(action.item())
                new_hids.append(new_hid)
            return actions, new_hids

        def seed():
            nonlocal seed_diverged
            nonlocal game
            nonlocal played_moves
            nonlocal hids
            for i, move in enumerate(game_state.game_moves):
                if game.is_chance():
                    # See applyMove() in rlcc/utils.cc for why we don't try to
                    # advance the hidden state at all.
                    #
                    # Similarly, the OBL bot in bot/hanabi_client.py doesn't either.
                    game.apply_move(move)
                else:
                    actions, new_hids = advance()
                    hids = new_hids
                    cur_player = game.get_current_player()

                    # seed_diverged is a check to see whether the players might have
                    # wanted to act differently given the "alternate history".
                    #
                    # if the player is partial to the exact colors, arguably then
                    # the player would want to deviate?
                    infer_move = game.get_move(actions[-1][cur_player])
                    infer_move_uid = game.get_hle_game().get_move_uid(infer_move)
                    move_uid = game.get_hle_game().get_move_uid(move)
                    if infer_move_uid != move_uid:
                        seed_diverged = True
                    game.apply_move(move)
                played_moves.append(move)

        if not seeded:
            seed()
            played_moves.append("diverge")
            seeded = True

        if game.is_chance():
            # We need to deal a card, so deal in the deck order.
            assert len(deal_after_diverge) > 0 or len(deck_deal) > 0
            if len(deal_after_diverge) > 0:
                move = deal_after_diverge[0]
                deal_after_diverge = deal_after_diverge[1:]
            else:
                move = deck_deal[0]
                deck_deal = deck_deal[1:]
            # See applyMove() in rlcc/utils.cc for why we don't try to
            # advance the hidden state at all.
            game.apply_move(move)
            played_moves.append(move)
        else:
            actions, new_hids = advance()
            cur_player = game.get_current_player()
            move = game.get_move(actions[-1][cur_player])

            game.apply_move(move)
            played_moves.append(move)
            hids = new_hids

    assert game.terminated()
    return {
        "seed_diverged": seed_diverged,
        "recolor_life_delta": game.get_life() - game_state.life,
        "recolor_info_delta": game.get_info() - game_state.info,
        "recolor_score_delta": game.get_score() - game_state.score,
        "old_deck": old_deck,
        "old_moves": old_actions,
        "played_moves": played_moves,
        "remapper": remapper,
    }


def run_simulation(p1_model, p2_model, pickle_output, num_game, fake_point, fake_percent):
    sim_results = []
    for _ in tqdm(range(num_game), leave=False):
        gs = run_game(p1_model, p2_model)
        fake_spot = None
        if fake_point is not None:
            fake_spot = fake_point
            if fake_spot >= len(gs.game_moves):
                continue
        elif fake_percent is not None:
            fake_spot = int(len(gs.game_moves) * fake_percent)
            if fake_spot < 1:
                continue
        sim_results.append(replay_game(p1_model, p2_model, gs, fake_spot))

    with open(pickle_output, "wb") as f:
        pickle.dump(sim_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Number of games to run.
    parser.add_argument("--num-game", type=int, default=1000)
    # Point of each game to fake out. (# results is <= number of games if game prematurely ends)
    parser.add_argument("--fake-point", type=int, default=None)
    # Percent Point of each game to fake out.
    parser.add_argument("--fake-point-percent", type=float, default=None)
    parser.add_argument("--glob-pattern", type=str, required=True)
    args = parser.parse_args()
    assert args.fake_point is not None or args.fake_point_percent is not None

    models = sorted([f for f in glob.glob(f"/home/wz2/off-belief-learning/models/{args.glob_pattern}/*/model0.pthw")])
    for model1, model2 in tqdm(itertools.product(models, repeat=2), total=len(models)*2, leave=False):
        path = Path(f"/home/wz2/off-belief-learning/pyhanabi/exps/{args.glob_pattern}")
        if args.fake_point is not None:
            path = path / f"fkpt{args.fake_point}"
        elif args.fake_point_percent is not None:
            path = path / f"fkpct{args.fake_point_percent * 100}"

        model1_name = Path(model1).parts[-2]
        (path / model1_name).mkdir(parents=True, exist_ok=True)

        model2_name = Path(model2).parts[-2]
        output = f"{path}/{model1_name}/{model2_name}.pkl"
        run_simulation(model1, model2, output, num_game=args.num_game, fake_point=args.fake_point, fake_percent=args.fake_point_percent)
    assert False

