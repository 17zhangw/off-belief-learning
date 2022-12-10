#!/bin/bash

set -ex

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl1_self/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl1_xp/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl2_self/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl2_xp/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL3/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL3/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl3_self/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL3/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL3/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl3_xp/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl4_self/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl4_xp/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl5_self/"

python tools/symmetry_analysis.py --num_cards=5 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/models/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_e/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl5_xp/"
