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

"""RNN module."""

import haiku as hk

from emergent_in_context_learning.modules import transformer_core


class RNN(hk.Module):
  """RNN tower."""

  def __init__(self,
               input_embedder,
               model_type='lstm',
               num_classes=1623,
               num_layers=8,
               hidden_size=512,
               dropout_prob=0.1,
               dense_init_scale=1.0,
               name=None):
    """Initialize the RNN tower.

    Args:
      input_embedder: InputEmbedder object.
      model_type: 'vanilla_rnn' or 'lstm'
      num_classes: Total number of output classes.
      num_layers: Number of RNN layers
      hidden_size: Size of RNN hidden layer.
      dropout_prob: Dropout probability.
      dense_init_scale: Scale for dense layer initialization.
      name: Optional name for the module.
    """
    super(RNN, self).__init__(name=name)
    self._input_embedder = input_embedder
    self._model_type = model_type
    self._num_classes = num_classes
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._dropout_prob = dropout_prob
    self._dense_init_scale = dense_init_scale

  def __call__(self, examples, labels, mask=None, is_training=True):
    """Call to the Transformer tower.

    Args:
      examples: input sequence of shape
        [batch_size, seq_len, height, width, channels]
      labels: input sequence of shape [batch_size, seq_len]
      mask: optional input mask of shape [batch_size, seq_len].
      is_training: if is currently training.

    Returns:
      outputs: output of the transformer tower
        of shape [batch_size, seq_len, channels].
    """
    # Encode the examples and labels.
    hh = self._input_embedder(examples, labels, is_training)

    if mask is not None:
      raise NotImplementedError  # not implemented properly below
      # see gelato.x.models.transformer.TransformerBlock

    if self._model_type == 'vanilla_rnn':
      rnn_module = hk.VanillaRNN
    elif self._model_type == 'lstm':
      rnn_module = hk.LSTM
    else:
      raise ValueError('Invalid self._model_type: %s' % self._model_type)

    rnn_stack = [rnn_module(self._hidden_size) for _ in range(self._num_layers)]
    rnn_stack = hk.DeepRNN(rnn_stack)
    state = rnn_stack.initial_state(batch_size=hh.shape[0])
    hh, _ = hk.static_unroll(rnn_stack, hh, state, time_major=False)  # (B,S,E)

    return transformer_core.conv1(
        hh, self._num_classes, init_scale=self._dense_init_scale)
