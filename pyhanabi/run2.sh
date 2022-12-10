#!/bin/bash

set -ex

python tools/symmetry_analysis.py --num_cards=3 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/pyhanabi/exps/obl1/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/pyhanabi/exps/obl1/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl1_long_self/"

python tools/symmetry_analysis.py --num_cards=3 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/pyhanabi/exps/obl1/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/pyhanabi/exps/obl1/model4.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl1_long_xp/"

python tools/symmetry_analysis.py --num_cards=3 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/pyhanabi/exps/obl2/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/pyhanabi/exps/obl2/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl2_long_self/"

python tools/symmetry_analysis.py --num_cards=3 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/pyhanabi/exps/obl2/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/pyhanabi/exps/obl2/model4.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl2_long_xp/"

python tools/symmetry_analysis.py --num_cards=3 --num-game=1 --repeat-main=50 --num-recolor=1000 --fake-percents="0.25,0.50,0.75" \
                                  --model1="/home/wz2/off-belief-learning/pyhanabi/exps/obl3/model0.pthw" \
                                  --model2="/home/wz2/off-belief-learning/pyhanabi/exps/obl3/model0.pthw" \
                                  --output="/home/wz2/off-belief-learning/pyhanabi/symmetry/obl3_long_self/"
