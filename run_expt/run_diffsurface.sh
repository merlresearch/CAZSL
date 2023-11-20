#!/bin/sh
# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# different surface experiment

# no context
python3 ../omnipush_ANP.py --job_name plywood-anp --test_data ../data/plywood --split ../plywood_split.json
python3 ../omnipush_cazsl.py --job_name plywood-baseline --context_type nocontext --reg_type noreg --lambda1 0 --lambda2 0 --test_data ../data/plywood --split ../plywood_split.json

# indicator context
python3 ../omnipush_cazsl.py --job_name plywood-concat --context_type indicator --concat --reg_type noreg --lambda1 0 --lambda2 0 --test_data ../data/plywood --split ../plywood_split.json
python3 ../omnipush_cazsl.py --job_name plywood-cazsl-noreg --context_type indicator --reg_type noreg --lambda1 0 --lambda2 0 --test_data ../data/plywood --split ../plywood_split.json
python3 ../omnipush_cazsl.py --job_name plywood-cazsl-l2reg --context_type indicator --reg_type l2 --lambda1 0.01 --lambda2 10 --test_data ../data/plywood --split ../plywood_split.json
python3 ../omnipush_cazsl.py --job_name plywood-cazsl-neuralreg --context_type indicator --reg_type neural --lambda1 0.01 --lambda2 1 --test_data ../data/plywood --split ../plywood_split.json
