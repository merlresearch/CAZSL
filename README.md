<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CAZSL: Zero-Shot Regression for Pushing Models by Generalizing Through Context

## Features

Implements the CAZSL methods and experiments on the Omnipush Dataset in **CAZSL: Zero-Shot Regression for Pushing Models by Generalizing Through Contexts** by Wenyu Zhang, Skyler Seto, and Devesh Jha.

## Installation

***This version of CAZSL is implemented with Python 3 and PyTorch 1.3***

## Usage

## Data and preprocessing

Steps to download and preprocess data for experiments on the Omnipush Dataset are delineated. See http://web.mit.edu/mcube/omnipush-dataset/index.html (License: `CC-BY-4.0`) for a description of the dataset.

### *Different objects* and *Different weights*

Save the following data files in the folder /data/0,1,2_weights/.

- ftp://omnipush.mit.edu/omnipush/meta-learning_data/0,1,2_weights/

### *Different surfaces*

Save the following data files in the folder /data/plywood/.

- ftp://omnipush.mit.edu/omnipush/meta-learning_data/plywood/

### Visual context

To use visual context, save the following data files in the folder /data/top-down_view.

- ftp://omnipush.mit.edu/omnipush/top-down_view/0_weight/
- ftp://omnipush.mit.edu/omnipush/top-down_view/1_weight/
- ftp://omnipush.mit.edu/omnipush/top-down_view/2_weight/

Resize the images to default 32-by-32 by running the `preprocess.py` file. The resized images will be saved in the folder /data/top-down_view32.

```unix
python3 preprocess.py
```

# Testing

## Running CAZSL models and baselines

### Fitting a single model

To fit a model with a specified list of arguments, run `omnipush_cazsl.py`. It can also be configured to run baseline models for easy comparison.

The key arguments to be specified based on the experiment and the chosen model are:

- job name for saving model outputs (job_name)
- data folder (data)
- test data folder, if different from data folder (test_data)
- JSON file containing train-test sample split (split)
- type of context used (context_type: nocontext, indicator, visual)
- image folder for visual context (image_dir)
- concatenate context to input of base network, instead of using CAZSL masking and regularization (add --concat)
- type of CAZSL context regularization (reg_type: noreg, l2, neural)
- CAZSL context regularization hyperparameters (&lambda;1, &lambda;2)

See `omnipush_cazsl.py` for other arguments that can be specified. The Attentive Neural Process baseline is in `omnipush_ANP.py`, with similar arguments that can be specified.

### Running an experiment

To replicate an experiment in the paper **CAZSL: Zero-Shot Regression for Pushing Models by Generalizing Through Contexts**, run the appropriate bash file in the /run_expt/ folder and obtain results by running the `read_tensorboard_log.py` file. For example for the *Different objects* experiment, job names for all methods will be prefixed by *omnipush* or *omnipush-image* depending on whether context is indicator or visual, and results can be obtained by:

```unix
./run_diffobject.sh
python3 read_tensorboard_log.py --expt omnipush
python3 read_tensorboard_log.py --expt omnipush-image
```

See `read_tensorboard_log.py` for a list of arguments to obtain results for other experiments. The arguments follow those in `omnipush_cazsl.py`.

## Citation

If you use the software, please cite the following  ([TR2020-140](TR_URL)):

```bibTeX
@inproceedings{zhangcazsl,
author = {Zhang, Wenyu and Seto, Skyler and Jha, Devesh K.},
title = {CAZSL: Zero-Shot Regression for Pushing Models by Generalizing Through Context},
booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
year = {2020},
pages = {7131-7138},
publisher = {IEEE},
doi = {10.1109/IROS45743.2020.9340858}
}
```

## Related Links

https://merl.com/publications/TR2020-140

## Contact

jha@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
