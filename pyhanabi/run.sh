#!/bin/bash

set -ex

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_a_a_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_a_a_gm1000/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_a_e_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_a_e_gm1000/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_b_d_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl5_b_d_gm1000/"

# this hoards memory
#python tools/action_matrix.py --num_player 2 --num_game 10000 --num_thread 10

# go-go-go
#python tools/symmetry_analysis.py --fake-point-percent 0.5 --num-game 1000 --glob-pattern icml_OBL1 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.5 --num-game 1000 --glob-pattern icml_OBL2 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.5 --num-game 1000 --glob-pattern icml_OBL3 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.5 --num-game 1000 --glob-pattern icml_OBL4 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.5 --num-game 1000 --glob-pattern icml_OBL5 &
#pids[0]=$!
#
## wait for all pids
#for pid in ${pids[*]}; do
#    wait $pid
#done

# go-go-go
#python tools/symmetry_analysis.py --fake-point-percent 0.75 --num-game 1000 --glob-pattern icml_OBL1 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.75 --num-game 1000 --glob-pattern icml_OBL2 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.75 --num-game 1000 --glob-pattern icml_OBL3 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.75 --num-game 1000 --glob-pattern icml_OBL4 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.75 --num-game 1000 --glob-pattern icml_OBL5 &
#pids[0]=$!
#
## wait for all pids
#for pid in ${pids[*]}; do
#    wait $pid
#done

# go-go-go
#python tools/symmetry_analysis.py --fake-point-percent 0.25 --num-game 1000 --glob-pattern icml_OBL1 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.25 --num-game 1000 --glob-pattern icml_OBL2 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.25 --num-game 1000 --glob-pattern icml_OBL3 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.25 --num-game 1000 --glob-pattern icml_OBL4 &
#pids[0]=$!
#python tools/symmetry_analysis.py --fake-point-percent 0.25 --num-game 1000 --glob-pattern icml_OBL5 &
#pids[0]=$!
#
## wait for all pids
#for pid in ${pids[*]}; do
#    wait $pid
#done
