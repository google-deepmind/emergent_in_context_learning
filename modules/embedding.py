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

"""Input embedding module."""

import functools

import haiku as hk
from haiku import initializers as init
import jax.numpy as jnp

from emergent_in_context_learning.modules import resnet


def _create_positional_encodings(inputs, max_time=30.0):
  """Generates positional encodings for the input.

  Args:
    inputs: A tensor of shape [batch_size, seq_len, emb_size].
    max_time: (default 10000) Constant used to scale position by in the
      encodings.

  Returns:
    pos_emb: as defined above, of size [1, seq_len, emb_size].
  """

  _, seq_len, embedding_size = inputs.shape

  if embedding_size % 2 == 1:
    raise ValueError(
        'Embedding sizes must be even if using positional encodings.')

  # Generate a sequence of positions and frequencies.
  pos = jnp.arange(seq_len, dtype=jnp.float32)
  freqs = jnp.arange(0, embedding_size, 2, dtype=jnp.float32)
  inverse_freqs = 1.0 / (max_time**(freqs / embedding_size))

  # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
  pos_emb = jnp.einsum('i,j->ij', pos, inverse_freqs)

  # Concat sines and cosines and return.
  pos_emb = jnp.concatenate([jnp.sin(pos_emb), jnp.cos(pos_emb)], -1)

  return pos_emb


class InputEmbedder(hk.Module):
  """Input embedder."""

  def __init__(self,
               num_classes=1623,
               emb_dim=64,
               example_encoding='resnet',
               flatten_superpixels=False,
               example_dropout_prob=0.0,
               concatenate_labels=False,
               use_positional_encodings=True,
               positional_dropout_prob=0.1,
               name=None):
    """Initialize the input embedder.

    Args:
      num_classes: Total number of output classes.
      emb_dim: Dimensionality of example and label embeddings.
      example_encoding: How to encode example inputs.
        'resnet': simple resnet encoding
        'linear': flatten and pass through a linear layer
        'embedding': pass through an embedding layer
      flatten_superpixels: Whether to flatten the output of the resnet (instead
        of taking a mean over superpixels).
      example_dropout_prob: Dropout probability on example embeddings. Note that
        these are applied at both train and test.
      concatenate_labels: Whether to concatenate example and label embeddings
        into one token for each (example, label) pair, rather than being fed to
        the transformer as two separate tokens.
      use_positional_encodings: Whether to use positional encoding.
      positional_dropout_prob: Positional dropout probability.
      name: Optional name for the module.
    """
    super(InputEmbedder, self).__init__(name=name)
    self._num_classes = num_classes
    self._emb_dim = emb_dim
    self._example_encoding = example_encoding
    self._flatten_superpixels = flatten_superpixels
    self._example_dropout_prob = example_dropout_prob
    self._concatenate_labels = concatenate_labels
    self._use_positional_encodings = use_positional_encodings
    self._positional_dropout_prob = positional_dropout_prob

  def __call__(self, examples, labels, is_training=True):
    """Call to the input embedder.

    Args:
      examples: input sequence of shape
        [batch_size, seq_len, height, width, channels]
      labels: input sequence of shape [batch_size, seq_len]
      is_training: if is currently training.

    Returns:
      outputs: output of the transformer tower
        of shape [batch_size, seq_len, channels].
    """
    # Encode the example inputs into shape (B, SS, E)
    if self._example_encoding == 'resnet':
      if self._flatten_superpixels:
        resnet_emb_dim = int(self._emb_dim / 16)
      else:
        resnet_emb_dim = self._emb_dim
      example_encoding = resnet.SimpleResNet(
          blocks_per_group=(2, 2, 2, 2),
          channels_per_group=(16, 32, 32, resnet_emb_dim),
          flatten_superpixels=self._flatten_superpixels,
      )
      example_encoding = hk.BatchApply(
          functools.partial(example_encoding, is_training=is_training))
      h_example = example_encoding(examples)
    elif self._example_encoding == 'linear':
      h_example = hk.Flatten(preserve_dims=2)(examples)
      h_example = hk.Linear(self._emb_dim)(h_example)
    elif self._example_encoding == 'embedding':
      h_example = hk.Embed(self._num_classes, self._emb_dim)(examples)
    else:
      raise ValueError('Invalid example_encoding: %s' % self._example_encoding)

    # Add dropout to example embeddings.
    # Note that this is not restricted to training, because the purpose is to
    # add noise to the examples, not for regularization.
    if self._example_dropout_prob:
      h_example = hk.dropout(hk.next_rng_key(), self._example_dropout_prob,
                             h_example)

    # Embed the labels.
    n_emb_classes = self._num_classes
    labels_to_embed = labels
    if self._concatenate_labels:
      # Dummy label for final position, where we don't want the label
      # information to be available.
      n_emb_classes += 1
      labels_to_embed = labels_to_embed.at[:, -1].set(n_emb_classes - 1)
    embs = hk.get_parameter(
        'embs', [n_emb_classes, self._emb_dim],
        init=init.TruncatedNormal(stddev=0.02))
    h_label = embs[labels_to_embed]  # (B, SS, E)

    if self._concatenate_labels:
      # Concatenate example and label embeddings
      hh = jnp.concatenate((h_example, h_label), axis=2)  # (B,SS,E*2)
    else:
      # Interleave example and label embeddings
      hh = jnp.empty(
          (h_example.shape[0], h_example.shape[1] * 2 - 1, h_example.shape[2]),
          dtype=h_example.dtype)
      hh = hh.at[:, 0::2].set(h_example)
      hh = hh.at[:, 1::2].set(h_label[:, :-1])
      # hh is (B,S,E) where S=SS*2-1

    # Create positional encodings.
    if self._use_positional_encodings:
      positional_encodings = _create_positional_encodings(hh)
      if is_training:
        positional_encodings = hk.dropout(hk.next_rng_key(),
                                          self._positional_dropout_prob,
                                          positional_encodings)
      # Add on the positional encoding.
      hh += positional_encodings

    return hh
