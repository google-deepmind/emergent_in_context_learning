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

"""Transformer module."""

import haiku as hk

from emergent_in_context_learning.modules import transformer_core


class Transformer(hk.Module):
  """Transformer tower."""

  def __init__(self,
               input_embedder,
               num_classes=1623,
               num_layers=8,
               num_heads=8,
               dropout_prob=0.1,
               self_att_init_scale=1.0,
               dense_init_scale=1.0,
               name=None):
    """Initialize the Transformer tower.

    Args:
      input_embedder: InputEmbedder object.
      num_classes: Total number of output classes.
      num_layers: Number of transformer blocks.
      num_heads: Number of transformer heads.
      dropout_prob: Dropout probability.
      self_att_init_scale: Scale for self-attention initialization.
      dense_init_scale: Scale for dense layer initialization.
      name: Optional name for the module.
    """
    super(Transformer, self).__init__(name=name)
    self._input_embedder = input_embedder
    self._num_classes = num_classes
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._self_att_init_scale = self_att_init_scale
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
      attention_mask = mask[:, None, None, :]
    else:
      attention_mask = None

    for _ in range(self._num_layers):
      if mask is not None:
        hh *= mask[:, :, None]
      hh = transformer_core.TransformerBlock(
          causal=True,
          widening_factor=4,
          num_heads=self._num_heads,
          self_att_init_scale=self._self_att_init_scale,
          dense_init_scale=self._dense_init_scale,
          dropout_prob=self._dropout_prob)(
              hh, mask=attention_mask, is_training=is_training)
    hh = transformer_core.layer_norm(hh)
    if mask is not None:
      hh *= mask[:, :, None]  # (B,S,E)
    return transformer_core.conv1(
        hh, self._num_classes, init_scale=self._dense_init_scale)
