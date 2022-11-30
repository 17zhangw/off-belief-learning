#!/bin/bash

set -ex

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_a_a_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_a_a_gm1000/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_a_e_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_a_e_gm1000/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_b_d_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_b_d_gm1000/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=100 --num-recolor=100 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA1_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_b_bzad_gm100/"

python tools/symmetry_analysis.py --num-game=50 --repeat-main=1000 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA1_BELIEF_d/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/exps/xp_obl2_b_bzad_gm1000/"
