#!/bin/sh
# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# different weights experiment, with training on 0, 2 weights and testing on 1 weights

# no context
python3 ../omnipush_ANP.py --job_name weight-0,2-anp --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-baseline --context_type nocontext --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,2_split.json

# indicator context
python3 ../omnipush_cazsl.py --job_name weight-0,2-concat --context_type indicator --concat --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-cazsl-noreg --context_type indicator --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-cazsl-l2reg --context_type indicator --reg_type l2 --lambda1 0.01 --lambda2 10 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-cazsl-neuralreg --context_type indicator --reg_type neural --lambda1 0.01 --lambda2 1 --split ../0,2_split.json

# visual context
python3 ../omnipush_cazsl.py --job_name weight-0,2-image-concat --context_type visual --concat --lambda1 0 --lambda2 0 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-image-cazsl-noreg --context_type visual --reg_type noreg --lambda1 0 --lambda2 0 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-image-cazsl-l2reg --context_type visual --reg_type l2 --lambda1 0.01 --lambda2 0.01 --split ../0,2_split.json
python3 ../omnipush_cazsl.py --job_name weight-0,2-image-cazsl-neuralreg --context_type visual --reg_type neural --lambda1 0.01 --lambda2 1 --split ../0,2_split.json
