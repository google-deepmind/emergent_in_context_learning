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

"""Transformer experiment for Omniglot Sequences datasets."""

import collections
import datetime
import functools
import math
import os
import signal
import threading

from absl import app
from absl import flags
from absl import logging
import dill
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils
import numpy as np
import optax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from emergent_in_context_learning.datasets import data_generators
from emergent_in_context_learning.datasets import utils as dataset_utils
from emergent_in_context_learning.modules import losses
from emergent_in_context_learning.modules.embedding import InputEmbedder
from emergent_in_context_learning.modules.rnn import RNN
from emergent_in_context_learning.modules.transformer import Transformer

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS


class Experiment(experiment.AbstractExperiment):
  """Omniglot sequences transformer experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Determine what kinds of sequences we'll train/eval on.
    if self.mode == 'train':
      self.seq_type = self.config.data.train_seqs
    else:
      self.seq_type = self.mode.replace('eval_', '')

    # Determine kinds of data the sequences will be composed of.
    self.example_type = self.config.data.example_type
    if self.example_type == 'omniglot':
      dataset_for_sampling = data_generators.OmniglotDatasetForSampling(
          **self.config.data.omniglot_config)
    elif self.example_type == 'symbolic':
      dataset_for_sampling = data_generators.SymbolicDatasetForSampling(
          **self.config.data.symbolic_config)
    else:
      raise ValueError('Invalid value for self.example_type: %s' %
                       self.example_type)
    self.data_generator_factory = data_generators.SeqGenerator(
        dataset_for_sampling,
        **self.config.data.generator_config,
    )

    sub_configs = self._get_sub_configs()
    self.embed_config, self.seq_config, self.model_config = sub_configs

    self.forward = hk.transform_with_state(self._forward_fn)

    if self.mode == 'train':
      init_batch = next(self._build_train_input())
      init_examples = init_batch['examples']  # (D,B,SS,H,W,C) for images
      # (D,B,SS) for symbols
      init_labels = init_batch['labels']  # (D,B,SS)
      p_init = jax.pmap(functools.partial(self.forward.init, is_training=True))
      init_mask = None
      init_rng = utils.bcast_local_devices(self.init_rng)
      self._params, self._state = p_init(init_rng, init_examples, init_labels,
                                         init_mask)
      self._train_input = utils.py_prefetch(self._build_train_input)
      self._train_input = utils.double_buffer_on_gpu(self._train_input)
      self._opt_init, _ = self.optimizer(self.config.training.learning_rate)
      self._opt_state = jax.pmap(self._opt_init)(self._params)
      self._update_func = jax.pmap(self._update_func, axis_name='i')
    else:
      # Needed for checkpoint restore.
      self._params = None
      self._state = None
      self._opt_state = None
      # JIT the evaluation function for the single-device case.
      # (In the training case above, pmap compiles the function to XLA so jit is
      # not needed.)
      self._eval_batch = jax.jit(self._eval_batch)

  def _get_sub_configs(self):
    """Get embed_config, seq_config, and model_config."""

    # Initialize embed config.
    embed_config = self.config.embedding

    # Get sequence config.
    seq_config = self.config.data.seq_config
    if ('fewshot' in self.seq_type) or (self.seq_type == 'mixed'):
      seq_config.seq_len = seq_config.fs_shots * seq_config.ways + 1

    # Initialize model config.
    if self.config.seq_model == 'transformer':
      model_config = self.config.transformer
    elif self.config.seq_model in ['lstm', 'vanilla_rnn']:
      model_config = self.config.rnn
    else:
      raise ValueError('Invalid value for config.seq_model: %s' %
                       self.config.seq_model)

    # Set num_classes, based on the data config.
    if 'ordered_polysemy' in seq_config.labeling_rare:
      polysemy_factor = int(
          seq_config.labeling_rare.split('ordered_polysemy')[1])
      num_classes = (
          polysemy_factor * self.data_generator_factory.n_rare_classes +
          self.data_generator_factory.n_common_classes)
    else:
      num_classes = self.data_generator_factory.n_classes
    embed_config.num_classes = num_classes
    model_config.num_classes = num_classes

    return embed_config, seq_config, model_config

  def _forward_fn(self, examples, labels, mask, is_training):
    embedder = InputEmbedder(**self.embed_config)
    seq_model = self.config.seq_model
    if seq_model == 'transformer':
      model = Transformer(embedder, **self.model_config)
    elif seq_model in ['lstm', 'vanilla_rnn']:
      model = RNN(embedder, seq_model, **self.model_config)
    else:
      raise ValueError('Invalid config.seq_model: %s' % seq_model)
    return model(examples, labels, mask, is_training=is_training)

  def optimizer(self, learning_rate):
    optimizer = getattr(optax, self.config.optimizer.name)
    return optimizer(learning_rate, **self.config.optimizer.kwargs)

  def _linear_warmup_and_sqrt_decay(self, global_step):
    """Linear warmup and then an inverse square root decay of learning rate."""
    max_lr = self.config.optimizer['max_lr']
    warmup_steps = int(self.config.optimizer['warmup_steps'])
    linear_ratio = max_lr / warmup_steps
    decay_ratio = jnp.power(warmup_steps * 1.0, 0.5) * max_lr
    return jnp.min(jnp.array([
        linear_ratio * global_step, decay_ratio * jnp.power(global_step, -0.5)
    ]))

  def _compute_loss_and_accuracy(self, logits, labels):
    """Computes cross entropy loss and accuracy for given logits and labels.

    The loss and accuracy are also computed separately for the interim
    predictions, i.e. for the support examples, (loss_interim, accuracy_interim)
    and for the final query predictions (loss_query, accuracy_query).

    Args:
      logits: A tensor of shape [batch_size, seq_len, n_classes].
      labels: A tensor of shape [batch_size, seq_len]

    Returns:
      A dict with entries {'scalar_name': scalar_value} where the scalar metrics
      are aggregated across batches.
    """
    # Compute softmax cross entropy.
    labels_one_hot = hk.one_hot(labels, self.model_config.num_classes)
    losses_all = losses.softmax_cross_entropy(
        logits, labels_one_hot, reduction=None)

    # Compute support and query masks.
    w_interim = self.config.training.w_interim_predictions
    n_interim = int((labels.shape[-1] - 1)/ 2)
    interim_mask = jnp.full_like(losses_all, False).at[:, :-1:2].set(True)
    query_mask = jnp.full_like(losses_all, False).at[:, -1].set(True)

    # Compute weighted loss mask.
    if w_interim:
      # Loss weighting on both interim and query predictions.
      # e.g. a seq with 2 support examples: weights are [w/2, 0, w/2, 0, (1-w)]
      if self.embed_config.concatenate_labels:
        raise NotImplementedError  # below assumes interleaved examples & labels
      loss_weightings = jnp.full_like(losses_all, 0.)
      loss_weightings += interim_mask * w_interim / n_interim
      loss_weightings += query_mask * (1 - w_interim)
    else:
      # Loss computed only for query predictions.
      # e.g. for a seq w/ 2 support examples, weights are [0, 0, 0, 0, 1]
      loss_weightings = query_mask

    def _apply_masks(values):
      values_query = jnp.sum(query_mask * values) / jnp.sum(query_mask)
      if w_interim:
        values_interim = jnp.sum(interim_mask * values) / jnp.sum(interim_mask)
      else:
        values_interim = 0.
      return values_query, values_interim

    # Compute loss numbers.
    losses_weighted = losses_all * loss_weightings
    loss = jnp.sum(losses_weighted) / jnp.sum(loss_weightings)
    loss_query, loss_interim = _apply_masks(losses_weighted)

    # Compute accuracy numbers.
    predicted_labels = jnp.argmax(logits, axis=-1)
    if ('eval' in self.mode and 'no_support' in self.seq_type and
        'polysemy' in self.config.data.seq_config.labeling_rare):
      labeling_rare = self.config.data.seq_config.labeling_rare
      assert self.config.data.train_seqs == 'bursty'
      assert 'ordered_polysemy' in labeling_rare
      polysemy_factor = int(labeling_rare.split('ordered_polysemy')[1])
      if self.seq_type in ['no_support_rare', 'no_support_zipfian']:
        # Compare predictions with all possible polysemous labels.
        labels_start_vals = labels // polysemy_factor * polysemy_factor
        correct = jnp.zeros_like(labels).astype(jnp.float32)
        for i in range(polysemy_factor):
          correct += jnp.equal(predicted_labels, labels_start_vals + i)
      elif self.seq_type == 'no_support_common':
        # Labels should be shifted to account for extra 'rare' labels.
        n_rare_classes = self.data_generator_factory.n_rare_classes
        common_start_idx = n_rare_classes * polysemy_factor
        labels += common_start_idx - n_rare_classes
        correct = jnp.equal(predicted_labels, labels).astype(jnp.float32)
      else:
        raise NotImplementedError
    else:
      # Regular accuracy computation.
      correct = jnp.equal(predicted_labels, labels).astype(jnp.float32)
    accuracy_query, accuracy_interim = _apply_masks(correct)

    # Determine the common and rare labels.
    if self.config.data.train_seqs != 'bursty':
      # Below assumes training on bursty seqs
      raise NotImplementedError
    labeling_common = self.seq_config.labeling_common
    labeling_rare = self.seq_config.labeling_rare
    n_rare_classes = self.data_generator_factory.n_rare_classes
    n_holdout_classes = self.data_generator_factory.n_holdout_classes
    n_classes = self.data_generator_factory.n_classes
    if 'polysemy' in labeling_rare:
      polysemy_factor = int(labeling_rare.split('ordered_polysemy')[1])
    # Common classes.
    if labeling_common == 'ordered':
      if 'polysemy' in labeling_rare:
        common_start_idx = n_rare_classes * polysemy_factor
      else:
        common_start_idx = n_rare_classes
      common_labels = range(common_start_idx, n_classes - n_holdout_classes)
    elif labeling_common == 'original':
      common_labels = self.data_generator_factory.common_classes
    else:
      raise NotImplementedError
    # Rare classes.
    if 'polysemy' in labeling_rare:
      rare_labels = range(n_rare_classes * polysemy_factor)
    elif labeling_rare in ['unfixed', 'ordered']:
      rare_labels = range(n_rare_classes)
    elif labeling_common == 'original':
      rare_labels = self.data_generator_factory.rare_classes
    else:
      raise NotImplementedError

    # Compute closed-class accuracy, for certain sequence types.
    # (only consider logits for the relevant classes)
    if ('bursty' in self.seq_type or 'fewshot' in self.seq_type or
        'no_support' in self.seq_type):
      if 'bursty' in self.seq_type:
        valid_labels = range(self.seq_config.ways)
      if 'fewshot' in self.seq_type:
        valid_labels = range(self.seq_config.ways)
      elif self.seq_type == 'no_support_common':
        valid_labels = common_labels
      elif self.seq_type == 'no_support_rare':
        valid_labels = rare_labels
      elif self.seq_type == 'no_support_zipfian':
        valid_labels = list(common_labels) + list(rare_labels)
      valid_labels = jnp.array(valid_labels)
      logits_closed = jnp.full_like(logits, -jnp.inf)
      logits_closed = (
          logits_closed.at[:, :, valid_labels].set(logits[:, :, valid_labels]))
      predicted_labels_closed = jnp.argmax(logits_closed, axis=-1)
      correct_closed = jnp.equal(predicted_labels_closed, labels)
      accuracy_closed, _ = _apply_masks(correct_closed.astype(jnp.float32))
    else:
      accuracy_closed = 0.

    # Compute whether query predictions were from common or rare classes.
    from_common_all = jnp.isin(predicted_labels, jnp.array(common_labels))
    from_rare_all = jnp.isin(predicted_labels, jnp.array(rare_labels))
    from_common, _ = _apply_masks(from_common_all)  # average for query only
    from_rare, _ = _apply_masks(from_rare_all)  # average for query only

    # Compute whether query predictions were from the fewshot classes.
    fewshot_ways = self.seq_config.ways
    from_fewshot_all = jnp.isin(predicted_labels, jnp.arange(fewshot_ways))
    from_fewshot, _ = _apply_masks(from_fewshot_all)  # for query only

    # Compute whether query predictions were from labels in the support.
    # (Use reshaping trick to take advantage of Numpy's outer operations.)
    support_labels = labels[:, :-2:2]
    batch_size, seq_len = predicted_labels.shape
    support_len = support_labels.shape[1]
    predicted_labels_reshaped = predicted_labels.reshape(batch_size, seq_len, 1)
    support_labels_reshaped = support_labels.reshape(batch_size, 1, support_len)
    from_support_all = (predicted_labels_reshaped == support_labels_reshaped)
    from_support_all = from_support_all.sum(-1).astype(bool)
    from_support, _ = _apply_masks(from_support_all)  # avg for query only
    from_support_common, _ = _apply_masks(from_support_all * from_common_all)
    from_support_rare, _ = _apply_masks(from_support_all * from_rare_all)
    from_support_fewshot, _ = _apply_masks(from_support_all * from_fewshot_all)

    return {
        'loss': loss,
        'loss_query': loss_query,
        'loss_interim': loss_interim,
        'accuracy_query': accuracy_query,
        'accuracy_interim': accuracy_interim,
        'accuracy_closed': accuracy_closed,
        'from_common': from_common,
        'from_rare': from_rare,
        'from_fewshot': from_fewshot,
        'from_support': from_support,
        'from_support_common': from_support_common,
        'from_support_rare': from_support_rare,
        'from_support_fewshot': from_support_fewshot,
    }

  def _get_ds_seqs(self):
    """Build a TF dataset of sequences for desired sequence type."""

    # Get sequence generator and corresponding config arguments.
    cfg = self.seq_config
    if self.seq_type == 'bursty':
      seq_generator = self.data_generator_factory.get_bursty_seq
      generator_args = (cfg.seq_len, cfg.bursty_shots, cfg.ways, cfg.p_bursty,
                        cfg.p_bursty_common, cfg.p_bursty_zipfian,
                        cfg.non_bursty_type, cfg.labeling_common,
                        cfg.labeling_rare, cfg.randomly_generate_rare,
                        cfg.grouped)
    elif self.seq_type == 'no_support_common':
      seq_generator = self.data_generator_factory.get_no_support_seq
      all_unique = False
      generator_args = ('common', cfg.seq_len, all_unique, cfg.labeling_common,
                        cfg.randomly_generate_rare)
    elif self.seq_type == 'no_support_rare':
      seq_generator = self.data_generator_factory.get_no_support_seq
      all_unique = False
      generator_args = ('rare', cfg.seq_len, all_unique, cfg.labeling_common,
                        cfg.randomly_generate_rare)
    elif self.seq_type == 'no_support_zipfian':
      seq_generator = self.data_generator_factory.get_no_support_seq
      all_unique = False
      generator_args = ('zipfian', cfg.seq_len, all_unique, cfg.labeling_common,
                        cfg.randomly_generate_rare)
    elif self.seq_type == 'fewshot_rare':
      seq_generator = self.data_generator_factory.get_fewshot_seq
      generator_args = ('rare', cfg.fs_shots, cfg.ways, 'unfixed',
                        cfg.randomly_generate_rare, cfg.grouped)
    elif self.seq_type == 'fewshot_common':
      seq_generator = self.data_generator_factory.get_fewshot_seq
      generator_args = ('common', cfg.fs_shots, cfg.ways, 'unfixed', False,
                        cfg.grouped)
    elif self.seq_type == 'fewshot_zipfian':
      seq_generator = self.data_generator_factory.get_fewshot_seq
      generator_args = ('zipfian', cfg.fs_shots, cfg.ways, 'unfixed',
                        cfg.randomly_generate_rare, cfg.grouped)
    elif self.seq_type == 'fewshot_holdout':
      seq_generator = self.data_generator_factory.get_fewshot_seq
      generator_args = ('holdout', cfg.fs_shots, cfg.ways, 'unfixed',
                        cfg.randomly_generate_rare, cfg.grouped)
    elif self.seq_type == 'mixed':
      seq_generator = self.data_generator_factory.get_mixed_seq
      generator_args = (cfg.fs_shots, cfg.ways, cfg.p_fewshot)
    else:
      raise ValueError('Invalid seq_type: %s' % self.seq_type)

    # Set the correct example shape and dtype.
    if self.example_type == 'omniglot':
      example_shape = (cfg.seq_len, 105, 105, 1)
      example_dtype = tf.dtypes.float32
    elif self.example_type == 'symbolic':
      example_shape = (cfg.seq_len,)
      example_dtype = tf.dtypes.int32
    else:
      raise ValueError('Invalid self.example_type: %s' % self.example_type)

    # Build the TF dataset from the generator.
    ds_seqs = tf.data.Dataset.from_generator(
        seq_generator,
        args=generator_args,
        output_signature={
            'example':
                tf.TensorSpec(
                    shape=example_shape, dtype=example_dtype),
            'label':
                tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.dtypes.int32),
            'is_rare':
                tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.dtypes.int32)
        })

    return ds_seqs

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, writer, **unused_args):
    """See base class."""

    batch = next(self._train_input)
    (self._params, self._state, self._opt_state, scalars, logits, labels) = (
        self._update_func(self._params, self._state, self._opt_state,
                          global_step, batch, rng))

    # Log logits, labels, example for last prediction in the first sequence.
    logits_to_log = logits[0][0][-1]
    scalars = utils.get_first(scalars)
    scalars.update({
        'prediction': np.argmax(logits_to_log),
        'label': labels[0][0][-1]
    })
    if self.example_type == 'symbolic':
      scalars.update({'example': batch['examples'][0][0][-1]})
    return scalars

  def _build_train_input(self):
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    # Build TF dataset of sequences for desired sequence type.
    ds_seqs = self._get_ds_seqs()

    # Batch and prepare data for transformer.
    shuffle_buffer_size = 100
    ds = ds_seqs.batch(per_device_batch_size)
    ds = dataset_utils.prepare_seqs_for_transformer(
        ds,
        use_constant_labels=False,
        interleave_targets=(not self.embed_config.concatenate_labels),
        downsample=self.config.preproc.downsample,
        )
    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(jax.local_device_count())
    return iter(tfds.as_numpy(ds))

  def _loss_fn(self, params, state, batch, rng):
    attention_mask = None
    logits, state = self.forward.apply(
        params,
        state,
        rng=rng,
        examples=batch['examples'],
        labels=batch['labels'],
        mask=attention_mask,
        is_training=True)

    labels = batch['target']

    loss_acc_scalars = self._compute_loss_and_accuracy(logits, labels)
    loss = loss_acc_scalars['loss']

    return loss, (state, logits, labels, loss_acc_scalars)

  def _update_func(self, params, state, opt_state, global_step, batch, rng):
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    grads, (state, logits, labels,
            loss_acc_scalars) = grad_loss_fn(params, state, batch, rng)
    grads = jax.lax.pmean(grads, axis_name='i')

    # Compute and apply updates via our optimizer.
    learning_rate = self._linear_warmup_and_sqrt_decay(global_step)
    _, opt_update = self.optimizer(learning_rate)
    updates, opt_state = opt_update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = jax.lax.pmean(loss_acc_scalars, axis_name='i')

    return params, state, opt_state, scalars, logits, labels

  def _vector_to_square(self, vector):
    """Convert 1-D array into a square-ish 2-D array."""
    n = len(vector)
    height = math.ceil(np.sqrt(n))
    width = math.ceil(n / height)
    vector_padded = jnp.concatenate((vector, jnp.zeros(height * width - n)))
    square = np.reshape(vector_padded, (height, -1))
    return square

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, writer, **unused_kwargs):
    """See base class."""

    global_step = np.array(utils.get_first(global_step))
    loss_acc_scalars, other_scalars, _ = self._eval_epoch(
        utils.get_first(rng))
    scalars = {**loss_acc_scalars, **other_scalars}
    scalars = {k: np.array(v) for k, v in scalars.items()}
    logging.info('[Step %d] eval_loss=%.2f, eval_accuracy=%.2f', global_step,
                 scalars['loss'], scalars['accuracy_query'])
    for k, v in scalars.items():
      logging.info('%s: %d', k, v)
    return scalars

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""

    # Build TF dataset of sequences for desired sequence type.
    ds_seqs = self._get_ds_seqs()

    # Batch and prepare data for transformer.
    ds = ds_seqs.batch(self.config.evaluation.batch_size)
    ds = dataset_utils.prepare_seqs_for_transformer(
        ds,
        use_constant_labels=False,
        interleave_targets=(not self.embed_config.concatenate_labels),
        downsample=self.config.preproc.downsample,
        )
    return iter(tfds.as_numpy(ds))

  def _eval_batch(self, params, state, batch, rng):
    """Evaluates a batch."""
    logits, _ = self.forward.apply(
        params, state, examples=batch['examples'], labels=batch['labels'],
        mask=None, rng=rng, is_training=False)  # [B, T, K]
    labels = batch['target']  # [B, T]

    loss_acc_scalars = self._compute_loss_and_accuracy(logits, labels)

    # Also return the last example, and the corresponding prediction and label.
    logits_to_log = logits[0][-1]
    logits_image = self._vector_to_square(logits_to_log)
    last_example = batch['examples'][0][-1]
    non_scalars = {
        'logits_image': logits_image,
    }
    last_prediction = np.argmax(logits_to_log)
    last_label = labels[0][-1]
    other_scalars = {
        'last_prediction': last_prediction,
        'last_label': last_label
    }
    if self.example_type == 'omniglot':
      non_scalars['last_example'] = last_example
    else:
      other_scalars['last_example'] = last_example

    return loss_acc_scalars, other_scalars, non_scalars

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    loss_acc_scalar_totals = collections.defaultdict(float)
    total_num_sequences = 0.

    # Checkpoints broadcast for each local device.
    params = utils.get_first(self._params)
    state = utils.get_first(self._state)

    n_batches_to_eval = 10000
    for i, batch in enumerate(self._build_eval_input()):
      # Make sure that the input has batch_dim=1
      assert batch['examples'].shape[0] == 1
      assert batch['labels'].shape[0] == 1

      loss_acc_scalars_batch, other_scalars, non_scalars = self._eval_batch(
          params, state, batch, rng)
      for k, v in loss_acc_scalars_batch.items():
        loss_acc_scalar_totals[k] += v
      total_num_sequences += batch['examples'].shape[0]

      if i > n_batches_to_eval:
        break

    loss_acc_scalars = {}
    for k, v in loss_acc_scalar_totals.items():
      loss_acc_scalars[k] = v / total_num_sequences

    return loss_acc_scalars, other_scalars, non_scalars


def _restore_state_to_in_memory_checkpointer(restore_path):
  """Initializes experiment state from a checkpoint."""

  # Load pretrained experiment state.
  python_state_path = os.path.join(restore_path, 'checkpoint.dill')
  with open(python_state_path, 'rb') as f:
    pretrained_state = dill.load(f)
  logging.info('Restored checkpoint from %s', python_state_path)

  # Assign state to a dummy experiment instance for the in-memory checkpointer,
  # broadcasting to devices.
  dummy_experiment = Experiment(
      mode='train', init_rng=0, config=FLAGS.config.experiment_kwargs.config)
  for attribute, key in Experiment.CHECKPOINT_ATTRS.items():
    setattr(dummy_experiment, attribute,
            utils.bcast_local_devices(pretrained_state[key]))

  jaxline_state = dict(
      global_step=pretrained_state['global_step'],
      experiment_module=dummy_experiment)
  snapshot = utils.SnapshotNT(0, jaxline_state)

  # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
  utils.GLOBAL_CHECKPOINT_DICT['latest'] = utils.CheckpointNT(
      threading.local(), [snapshot])


def _get_step_date_label(global_step):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


def _setup_signals(save_model_fn):
  """Sets up a signal for model saving."""
  # Save a model on Ctrl+C.
  def sigint_handler(unused_sig, unused_frame):
    # Ideally, rather than saving immediately, we would then "wait" for a good
    # time to save. In practice this reads from an in-memory checkpoint that
    # only saves every 30 seconds or so, so chances of race conditions are very
    # small.
    save_model_fn()
    logging.info(r'Use `Ctrl+\` to save and exit.')

  # Exit on `Ctrl+\`, saving a model.
  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(unused_sig, unused_frame):
    # Restore previous handler early, just in case something goes wrong in the
    # next lines, so it is possible to press again and exit.
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    save_model_fn()
    logging.info(r'Exiting on `Ctrl+\`')

    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info('Saving model.')
  for (checkpoint_name,
       checkpoint) in utils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    pickle_nest = checkpoint.history[-1].pickle_nest
    global_step = pickle_nest['global_step']

    state_dict = {'global_step': global_step}
    for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
      state_dict[key] = utils.get_first(
          getattr(pickle_nest['experiment_module'], attribute))
    save_dir = os.path.join(
        save_path, checkpoint_name, _get_step_date_label(global_step))
    python_state_path = os.path.join(save_dir, 'checkpoint.dill')
    os.makedirs(save_dir, exist_ok=True)
    with open(python_state_path, 'wb') as f:
      dill.dump(state_dict, f)
    logging.info(
        'Saved "%s" checkpoint to %s', checkpoint_name, python_state_path)


def main(argv, experiment_class):

  # Maybe restore a model.
  restore_path = FLAGS.config.restore_path
  if restore_path:
    _restore_state_to_in_memory_checkpointer(restore_path)

  # Maybe save a model.
  save_dir = os.path.join(FLAGS.config.checkpoint_dir, 'models')
  if FLAGS.config.one_off_evaluate:
    save_model_fn = lambda: None  # No need to save checkpoint in this case.
  else:
    save_model_fn = functools.partial(
        _save_state_from_in_memory_checkpointer, save_dir, experiment_class)
  _setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

  try:
    platform.main(experiment_class, argv)
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.
  platform.main(experiment_class, argv)


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(lambda argv: main(argv, Experiment))
