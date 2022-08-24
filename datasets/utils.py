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

"""Dataset utilities."""

import tensorflow.compat.v2 as tf


def prepare_seqs_for_transformer(ds,
                                 use_constant_labels=False,
                                 interleave_targets=True,
                                 downsample=False):
  """Convert example and label sequences for use by the transformer.

  Args:
    ds: A tf.data.Dataset where each example contains
      'example': a batch of examples with shape
                 (batch_size, seq_len, height, width, channels)
      'label': a batch of labels with shape
               (batch_size, seq_len)
    use_constant_labels: Whether to use target labels of all ones, instead of
      the true labels.
    interleave_targets: Whether to create targets consisting of alternating
      [label, 0, label, 0, ...] sequences, or just [label, label, ...]
    downsample: Whether to downsample images.

  Returns:
    A new tf.data.Dataset where each example contains
      'examples': a batch of examples
          for images: (batch_size, seq_len, height, width, channels) tf.float32
          for integers: (batch_size, seq_len) tf.int32
      'labels': a batch of labels (batch_size, seq_len) tf.int32
      'target': a batch of labels (batch_size, final_seq_len) tf.int32
                where final_seq_len = (seq_len*2 - 1) if interleave_targets is
                True, otherwise final_seq_len = seq_len
  """

  def _convert_dict(example):
    # (dims: B:batch, SS:original seqlen, H:height, W:width, C:channels)
    is_image = (len(example['example'].shape) == 5)

    # Cast the examples into the correct shape and tf datatype.
    if is_image:
      examples = tf.cast(example['example'], tf.float32)  # (B,SS,H,W,C)
      if downsample:
        examples = tf.map_fn(lambda batch: tf.image.resize(batch, [28, 28]),
                             examples)
    else:
      examples = tf.cast(example['example'], tf.int32)  # (B, SS)

    # Cast the labels into the correct tf datatype.
    if use_constant_labels:
      labels = tf.ones_like(example['label'], tf.int32)
    else:
      labels = tf.cast(example['label'], tf.int32)  # (B,SS)
    seq_len = labels.shape[-1]

    # Create the target sequence.
    if interleave_targets:
      # Alternating labels with zeros, e.g. [label, 0, label, 0, ...].
      zeros = tf.zeros_like(labels)
      target = tf.stack((labels[..., None], zeros[..., None]), axis=-1)
      target = tf.reshape(target, [-1, seq_len * 2])[:, :-1]  # (B,SS*2-1)
    else:
      # Just use the original sequence of labels, e.g. [label, label, ...]
      target = labels  # (B,SS)

    ret_dict = {'examples': examples,
                'labels': labels,
                'target': target}
    return tf.data.Dataset.from_tensors(ret_dict)

  return ds.flat_map(_convert_dict)
