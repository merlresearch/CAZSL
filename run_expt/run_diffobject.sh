#!/bin/sh

# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# different object experiment

# no context
python3 ../omnipush_ANP.py --job_name omnipush-anp
python3 ../omnipush_cazsl.py --job_name omnipush-baseline --context_type nocontext --reg_type noreg --lambda1 0 --lambda2 0

# indicator context
python3 ../omnipush_cazsl.py --job_name omnipush-concat --context_type indicator --concat --reg_type noreg --lambda1 0 --lambda2 0
python3 ../omnipush_cazsl.py --job_name omnipush-cazsl-noreg --context_type indicator --reg_type noreg --lambda1 0 --lambda2 0
python3 ../omnipush_cazsl.py --job_name omnipush-cazsl-l2reg --context_type indicator --reg_type l2 --lambda1 0.01 --lambda2 10
python3 ../omnipush_cazsl.py --job_name omnipush-cazsl-neuralreg --context_type indicator --reg_type neural --lambda1 0.01 --lambda2 1

# visual context
python3 ../omnipush_cazsl.py --job_name omnipush-image-concat --context_type visual --concat --lambda1 0 --lambda2 0
python3 ../omnipush_cazsl.py --job_name omnipush-image-cazsl-noreg --context_type visual --reg_type noreg --lambda1 0 --lambda2 0
python3 ../omnipush_cazsl.py --job_name omnipush-image-cazsl-l2reg --context_type visual --reg_type l2 --lambda1 0.01 --lambda2 0.01
python3 ../omnipush_cazsl.py --job_name omnipush-image-cazsl-neuralreg --context_type visual --reg_type neural --lambda1 0.01 --lambda2 1
