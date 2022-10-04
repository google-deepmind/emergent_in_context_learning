# Emergent In-Context Learning in Transformers

This is the codebase associated with the following paper:

**Data Distributional Properties Drive Emergent In-Context Learning in
Transformers** ([arXiv](https://arxiv.org/abs/2205.05055))<br/>
_Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya
Singh, Pierre H. Richemond, Jay McClelland, Felix Hill_

The experiments involve training and evaluating a transformer on sequences of
[Omniglot](https://github.com/brendenlake/omniglot) image-label pairs, to elicit
and measure (few-shot) in-context learning vs in-weights learning. See Sec 2 of
the paper for an overview of the experimental design.


## Installation

To install the necessary requirements:

```shell
python3 -m venv eicl_venv
source eicl_venv/bin/activate
pip install --upgrade pip
pip install -r ./emergent_in_context_learning/requirements.txt
```

## Usage

### Default configs

Default experiment configurations are provided in `configs/`, and can be used
in `$PATH_TO_CONFIG` in the launch commands below.

*   `images_all_exemplars.py`: Each character class consists of 20 image
    examples (the original Omniglot problem).
*   `images_augmented.py`: We augment the total number of classes to 8x the
    original number, by applying transformations to each image class: flip left
    or right + rotate 0, 90, 180, or 270 degrees.
*   `images_identical.py`: Each character class consists only of a single image
    (the 1st of the 20 examples provided in the original Omniglot dataset)
*   `symbolic.py`: (relatively untested; not used in the paper)

Config files can be edited or forked as desired.

### Varieties of data sequences + Configurations for each

Omniglot sequences are generated in `datasets/data_generators.py`.

The image classes are divided into training and holdout. Training classes can be
"common" or "rare". The training classes can be uniformly or Zipf-distributed
(jointly over both common and rare classes). Related configurations are set in
`config.data.generator_config`.

There are few different types of data sequences:

*   `bursty` : These are the canonical bursty (and non-bursty) sequences used in
    training in the paper
*   `no_support_common`, `no_support_rare`, `non_support_zipfian` : These
    sequences enforce that the query class does not appear anywhere in the
    context, and are the sequences used for evaluating in-weights learning in
    the paper. They can consist entirely of common classes, rare classes, or be
    Zipf-distributed over all training classes.
*   `fewshot_common`, `fewshot_rare`, `fewshot_zipfian`, `fewshot_holdout` :
    These sequence are standard k-shot n-way fewshot sequences, and are used for
    evaluating in-context learning in the paper. They can exist of holdout
    classes, common classes, rare classes, or be Zipf-distributed over all
    training classes.
*   `mixed`: A mix of standard fewshot and iid randomly generated sequences.

Sequence types are specified in `config.data.train_seqs` and in
`config.eval_modes` (with an additional `eval_` prefix). You may specify a list
of eval modes, to evaluate the same learner on multiple sequence types.

See `experiment/experiment.py: _get_ds_seqs` and `datasets/data_generators.py:
SeqGenerator` for more details on settings, which are specified in
`config.data.seq_config`.


### Launch commands

These commands should be executed from the directory that you cloned the
repository into.

To run training:

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG --jaxline_mode train --logtostderr
# (save checkpoints using Ctrl+C)
```

To evaluate a trained model, override `config.restore_path` with the
subdirectory of `config.checkpoint_dir` containing the relevant checkpoint
(`$CKPT_DIR` below).

To evaluate on in-context learning (on holdout classes):

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG
 --logtostderr --config.one_off_evaluate --config.restore_path $CKPT_DIR
 --jaxline_mode eval_fewshot_holdout
```

To evaluate on in-weights learning (on trained classes):

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG
 --logtostderr --config.one_off_evaluate --config.restore_path $CKPT_DIR
 --jaxline_mode eval_no_support_zipfian
```


## Citing this work

If you use this work, please cite the following paper
```
@misc{chan_data_2022,
  title = {Data Distributional Properties Drive Emergent In-Context Learning in Transformers},
  author = {Chan, Stephanie C. Y. and Santoro, Adam and Lampinen, Andrew K. and Wang, Jane X. and Singh, Aaditya and Richemond, Pierre H. and McClelland, Jay and Hill, Felix},
  journal = {Neural Information Processing Systems},
  year = {2022},
}
```

We would also like to thank the following colleagues for their contributions to
the implementation of the transformer model:
Igor Babuschkin, Junyoung Chung, David Choi, Tamara Norman, Sebastian Borgeaud,
Jack Rae, David Saxton, Yujia Li, Phil Blunsom, Maribeth Rauh, Roman Ring,
Nate Kushman, Vinicius Zambaldi, Tom Hennigan


## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
