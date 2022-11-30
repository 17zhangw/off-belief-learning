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
    def __init__(self, params, player_options, game_moves, game_deck, life, info, score):
        self.params = params
        self.player_options = player_options
        self.game_moves = game_moves
        self.game_deck = game_deck
        self.life = life
        self.info = info
        self.score = score

    params = None
    # list[list[tuple(hid, obs, new_hid, action)]]
    player_options = None
    # list[hanalearn.HanabiMove]
    game_moves = None
    # list[hanalearn.HanabiCardValue]
    game_deck = None

    life = None
    info = None
    score = None


def run_game(p1_model, p2_model, num_cards):
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
        "colors": str(num_cards),
        "ranks": str(num_cards),
        "hand_size": str(num_cards),
        "max_information_tokens": "8", # 8 observation tokens to start.
        "max_life_tokens": "3", # 3 life tokens to start.
        "bomb": str(0), # Start with 0 bombs.
        "random_start_player": str(0), # No random start.
    }
    game = hanalearn.HanabiEnv(params, -1, False)  # max_len  # verbose
    game.reset()

    player_options = [[], []]
    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            h[k] = v.cuda().unsqueeze(0) # add batch dim

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

            player_options[i].append((
                {k: hid[k].cpu() for k in hid.keys()},
                {k: obs[k].cpu() for k in obs.keys()},
                {k: new_hid[k].cpu() for k in new_hid.keys()},
                game.get_move(action.item()),
            ))
            new_hids.append(new_hid)

        hids = new_hids
        cur_player = game.get_current_player()
        move = game.get_move(actions[-1][cur_player])
        game.step(move)

    history = game.move_history()
    moves = []
    for move in history:
        moves.append(move.move)

    deck_card_history = game.deck_card_history()
    original_deck = [d for d in reversed(deck_card_history)]
    return GameState(params=params,
                     player_options=player_options,
                     game_moves=moves,
                     game_deck=original_deck,
                     life=game.get_life(),
                     info=game.get_info(),
                     score=game.get_score())


def replay_game(p1_model, p2_model, game_state, recolor=True, divergence_point=1, num_cards=5):
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

    def recolor_state():
        color_im = [l for l in range(num_cards)]
        color = [l for l in range(num_cards)]
        while color == color_im:
            random.shuffle(color)
        remapper = {i: color[i] for i in range(num_cards)}

        remapped_deck = []
        for card in game_state.game_deck:
            remapped_deck.append(hanalearn.HanabiCardValue(remapper[card.color()], card.rank()))

        remapped_actions = []
        for action in game_state.game_moves:
            remapped_actions.append(hanalearn.HanabiMove(
                action.move_type(),
                action.card_index(),
                action.target_offset(),
                -1 if action.color() == -1 else remapper[action.color()],
                action.rank()))
        return remapper, remapped_deck, remapped_actions

    if recolor:
        remapper, deck, game_moves = recolor_state()
    else:
        remapper = {i:i for i in range(0, num_cards)}
        deck = game_state.game_deck
        game_moves = game_state.game_moves

    # Load the deck with recolored cards.
    game = hanalearn.HanabiEnv(game_state.params, -1, False)  # max_len  # verbose
    game.reset_with_deck_no_chance(deck)

    # Make sure that we still deal cards in the right order.
    deck_deal = []
    for card in reversed(deck):
        deck_deal.append(hanalearn.HanabiMove(
            hanalearn.MoveType.Deal,
            -1,
            -1,
            card.color(),
            card.rank()))

    # This is the "force"
    game_moves = game_moves[:(num_cards*2)+divergence_point-1]

    # Seed the initial h0.
    player_options = [[], []]
    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            h[k] = v.cuda().unsqueeze(0) # add batch dim

    seed_diverged = False
    seeded = False
    while not game.terminated():
        def advance():
            nonlocal player_options
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
                player_options[i].append((hid, obs, new_hid, game.get_move(action.item())))
                new_hids.append(new_hid)
            return actions, new_hids

        def seed():
            nonlocal seed_diverged
            nonlocal game
            nonlocal hids
            nonlocal deck_deal
            for i, move in enumerate(game_moves):
                if game.is_chance():
                    # See applyMove() in rlcc/utils.cc for why we don't try to
                    # advance the hidden state at all.
                    #
                    # Similarly, the OBL bot in bot/hanabi_client.py doesn't either.
                    deck_deal = deck_deal[1:]
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

        if not seeded:
            seed()
            seeded = True

        if game.is_chance():
            # We need to deal a card, so deal in the deck order.
            assert len(deck_deal) > 0
            move = deck_deal[0]
            deck_deal = deck_deal[1:]
            # See applyMove() in rlcc/utils.cc for why we don't try to
            # advance the hidden state at all.
            game.apply_move(move)
        else:
            actions, new_hids = advance()
            cur_player = game.get_current_player()
            move = game.get_move(actions[-1][cur_player])

            game.apply_move(move)
            hids = new_hids

    history = game.move_history()
    moves = []
    for move in history:
        moves.append(move.move)

    assert game.terminated()
    return {
        "seed_diverged": seed_diverged,
        "recolor_life_minus_base": game.get_life() - game_state.life,
        "recolor_info_minus_base": game.get_info() - game_state.info,
        "recolor_score_minus_base": game.get_score() - game_state.score,
        "base_life": game_state.life,
        "base_info": game_state.info,
        "base_score": game_state.score,
        "played_moves": moves,
        "played_player_options": player_options,
        "remapper": remapper,
    }


def run_simulation(p1_model, p2_model, output_dir, num_game, repeat_main, num_subgame, fake_percents, num_cards):
    for gnum in tqdm(range(num_game), leave=False):
        gs = run_game(p1_model, p2_model, num_cards=num_cards)
        output = Path(output_dir) / f"game_{gnum}"
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(f"{output}/base", "wb") as f:
            pickle.dump(gs, f)

        sim_results = []
        for _ in tqdm(range(0, repeat_main), leave=False):
            sim_results.append(replay_game(p1_model, p2_model, gs, recolor=False, divergence_point=int(len(gs.game_moves) * 0.5), num_cards=num_cards))
        with open(f"{output}/base_replay", "wb") as f:
            pickle.dump(sim_results, f)

        for fake_percent in fake_percents.split(","):
            fake_spot = int(len(gs.game_moves) * float(fake_percent))
            if fake_spot < 1:
                continue

            sim_results = []
            for _ in tqdm(range(0, num_subgame), leave=False):
                sim_results.append(replay_game(p1_model, p2_model, gs, recolor=True, divergence_point=fake_spot, num_cards=num_cards))
            with open(f"{output}/intervention_{float(fake_percent)*100}_replay", "wb") as f:
                pickle.dump(sim_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cards", type=int, default=3)
    # Number of games to run.
    parser.add_argument("--num-game", type=int, default=50)
    parser.add_argument("--repeat-main", type=int, default=1000)
    parser.add_argument("--num-recolor", type=int, default=1000)
    # Point of each game to fake out. (# results is <= number of games if game prematurely ends)
    parser.add_argument("--fake-percents", type=str, default=None)
    parser.add_argument("--model1", type=str, default=None)
    parser.add_argument("--model2", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    #parser.add_argument("--glob-pattern", type=str, required=True)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        run_simulation(args.model1, args.model2, args.output,
                       num_game=args.num_game,
                       repeat_main=args.repeat_main,
                       num_subgame=args.num_recolor,
                       fake_percents=args.fake_percents,
                       num_cards=args.num_cards)
