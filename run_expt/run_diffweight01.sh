#!/bin/sh
# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# different weights experiment, with training on 0, 1 weights and testing on 2 weights

# no context
python3 ../omnipush_ANP.py --job_name weight-0,1-anp --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-baseline --context_type nocontext --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,1_split.json

# indicator context
python3 ../omnipush_cazsl.py --job_name weight-0,1-concat --context_type indicator --concat --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-cazsl-noreg --context_type indicator --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-cazsl-l2reg --context_type indicator --reg_type l2 --lambda1 0.01 --lambda2 10 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-cazsl-neuralreg --context_type indicator --reg_type neural --lambda1 0.01 --lambda2 1 --split ../0,1_split.json

# visual context
python3 ../omnipush_cazsl.py --job_name weight-0,1-image-concat --context_type visual --concat --lambda1 0 --lambda2 0 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-image-cazsl-noreg --context_type visual --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-image-cazsl-l2reg --context_type visual --reg_type l2 --lambda1 0.01 --lambda2 0.01 --split ../0,1_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,1-image-cazsl-neuralreg --context_type visual --reg_type neural --lambda1 0.01 --lambda2 1 --split ../0,1_split.json
