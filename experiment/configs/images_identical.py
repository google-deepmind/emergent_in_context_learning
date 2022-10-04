# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config for transformer experiment."""

from jaxline import base_config
from ml_collections import config_dict


ZIPF_EXPONENT = 0.


def get_config(debug=False):
  """Return config object for training."""

  def m(default_value, debug_value):
    """Helper function to return the default or debug value based debug."""
    return debug_value if debug else default_value

  config = base_config.get_base_config()

  # Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              data=dict(
                  train_seqs='bursty',
                  example_type='omniglot',  # 'omniglot' or 'symbolic'
                  generator_config=dict(
                      n_rare_classes=1603,  # 1623 - 20
                      n_common_classes=10,
                      n_holdout_classes=10,
                      zipf_exponent=ZIPF_EXPONENT,
                      use_zipf_for_common_rare=False,
                      noise_scale=0.,
                      preserve_ordering_every_n=None,
                  ),
                  omniglot_config=dict(
                      omniglot_split='all',  # 1623 total classes
                      exemplars='single',  # 'single' / 'separated' / 'all'
                      augment_images=False,  # multiply total classes x 8
                  ),
                  symbolic_config=dict(dataset_size=1000,),
                  seq_config=dict(
                      seq_len=9,  # NB: can get overridden for some seq types
                      fs_shots=4,
                      bursty_shots=3,
                      ways=2,
                      p_bursty=0.9,
                      p_bursty_common=0.,
                      p_bursty_zipfian=1.,
                      p_fewshot=0.1,
                      non_bursty_type='zipfian',
                      labeling_common='ordered',
                      labeling_rare='ordered',
                      randomly_generate_rare=False,
                      grouped=False,
                  ),
              ),
              preproc=dict(downsample=False,),
              optimizer=dict(
                  name='adam',
                  kwargs={},
                  # Set up the learning rate schedule.
                  max_lr=3e-4,
                  warmup_steps=4000,
                  clip_level=0.25,
              ),
              training=dict(
                  batch_size=4 * 8,
                  learning_rate=1e-4,
                  w_interim_predictions=0.,
              ),
              embedding=dict(
                  num_classes=None,  # is set later, depending on data config
                  emb_dim=64,
                  example_encoding='resnet',  # 'resnet'/'linear'/'embedding'
                  flatten_superpixels=False,  # to flatten resnet outputs
                  example_dropout_prob=0.0,
                  concatenate_labels=False,
                  use_positional_encodings=True,
                  positional_dropout_prob=0.0,
              ),
              seq_model='transformer',  # 'transformer'/'lstm'/'vanilla_rnn'
              transformer=dict(
                  num_classes=None,  # is set later, depending on data config
                  num_layers=m(12, 2),
                  num_heads=m(8, 2),
                  dropout_prob=0.0,
              ),
              rnn=dict(
                  num_classes=None,  # is set later, depending on data config
                  num_layers=m(12, 2),
                  hidden_size=64,
                  dropout_prob=0.0,
              ),
              evaluation=dict(batch_size=1,),
          ),))

  # Training loop config.
  config.training_steps = int(5e5)
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 300
  config.train_checkpoint_all_hosts = False
  config.checkpoint_dir = '/tmp/jaxline/transformer_omniglot/'
  config.eval_specific_checkpoint_dir = ''
  config.restore_path = ''

  # Evaluation modes.
  if ZIPF_EXPONENT:
    config.eval_modes = ('eval_no_support_zipfian', 'eval_fewshot_rare',
                         'eval_fewshot_holdout', 'eval_fewshot_common')
  else:  # uniform
    config.eval_modes = ('eval_no_support_zipfian', 'eval_fewshot_zipfian',
                         'eval_fewshot_holdout')

  return config
