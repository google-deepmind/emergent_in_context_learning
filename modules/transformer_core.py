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

"""Transformers."""

import math
from typing import Callable, Optional

import chex
import haiku as hk
from haiku import initializers as init
import jax
import jax.numpy as jnp


def conv1(
    x: chex.Array,
    num_units: int,
    init_scale: float = 1.,
    with_bias: bool = True,
) -> chex.Array:
  """Faster than actual 1D Conv on TPUs."""
  return hk.Linear(
      num_units,
      with_bias=with_bias,
      w_init=init.VarianceScaling(init_scale))(x)


def layer_norm(x: chex.Array) -> chex.Array:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def get_pos_start(timesteps, batch_size):
  # Need to find the right slice of positional embeddings for incremental
  # sampling.
  pos_start = hk.get_state(
      'cache_progress_idx', [batch_size], dtype=jnp.int32, init=jnp.zeros)
  hk.set_state('cache_progress_idx', pos_start + timesteps)
  return pos_start


def tiled_dropout(
    rng: chex.PRNGKey,
    rate: float,
    x: chex.Array,
    num_tile_dims: int,
) -> chex.Array:
  """Dropout with shared mask for the last `num_tile_dims` dimensions.

  Setting num_tile_dims to `0` recovers traditional dropout.

  Args:
    rng: A JAX random key.
    rate: Probability that each element of `x` is discarded. Must be a scalar in
      the range `[0, 1)`.
    x: input tensor to be dropped out.
    num_tile_dims: number of trailing dimensions to share the mask over.

  Returns:
    x: tensor with dropout and rescaling applied.
  """
  if rate < 0 or rate >= 1:
    raise ValueError('rate must be in [0, 1).')

  if rate == 0.0:
    return x

  keep_rate = 1.0 - rate
  keep = jax.random.bernoulli(rng, keep_rate, shape=x.shape[-num_tile_dims:])
  return keep * x / keep_rate


def attend(
    q: chex.Array,
    k: chex.Array,
    v: chex.Array,
    mask: Optional[chex.Array] = None,
    attend_fn: Optional[Callable[[chex.Array, chex.Array], chex.Array]] = None,
    dropout_prob: float = 0.,
    dropout_tile_dims: int = 0,
) -> chex.Array:
  """Computes multi-head attention using the given query, key and value.

  Args:
    q: Query with shape [batch, q_timesteps, num_heads, head_dim].
    k: Key with shape [batch, timesteps, num_heads, head_dim].
    v: Value with shape [batch, timesteps, num_heads, head_dim].
    mask: Attention mask to apply [batch, 1, timesteps, timesteps].
    attend_fn: An optionally defined attend function. The default attend_fn is
      is jnp.einsum('bthd,bThd->bhtT', q, k).
    dropout_prob: dropout probability on the attention weights.
    dropout_tile_dims: number of trailing dims to share dropout mask. Setting to
        zero falls back to the usual dropout.

  Returns:
    Output of the attention with shape [batch, timesteps, hiddens]
  """
  batch, q_time, num_heads, head_dim = q.shape
  hiddens = num_heads * head_dim

  _, kv_time, _, _ = k.shape
  expected_kv_shape = tuple([batch, kv_time, num_heads, head_dim])

  if k.shape != expected_kv_shape:
    raise ValueError(
        f'Expected key shape {expected_kv_shape} but got shape {k.shape}')
  if v.shape != expected_kv_shape:
    raise ValueError(
        f'Expected value shape {expected_kv_shape} but got shape {v.shape}')

  if attend_fn is not None:
    attention = attend_fn(q, k)
  else:
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)

  scale = 1. / math.sqrt(head_dim)
  attention *= scale
  if mask is not None:
    attention = attention * mask - 1e10 * (1 - mask)
  normalized = jax.nn.softmax(attention)
  if dropout_prob > 0:
    normalized = tiled_dropout(
        hk.next_rng_key(), dropout_prob, normalized, dropout_tile_dims)
  summed = jnp.einsum('bhtT,bThd->bthd', normalized, v)
  return jnp.reshape(summed, [batch, q_time, hiddens])


def get_reset_attention_mask(should_reset: chex.Array) -> chex.Array:
  """Maps a reset token vector into an attention mask that consists of blocks.

  A sequence of should reset tokens such as:
    [0, 1, 0, 1, 0, 0]
  transforms into an attention mask such as:
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0],
     [0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1]]
  Args:
    should_reset: Reset tokens with shape [batch, timesteps].

  Returns:
    attention_mask: Attention mask with shape [batch, timesteps, timesteps].
  """
  should_reset = jnp.cumsum(should_reset, axis=-1)
  attention_mask = should_reset[:, :, None] == should_reset[:, None, :]
  return attention_mask.astype(jnp.float32)


def relative_shift(x: chex.Array) -> chex.Array:
  x_shape = list(x.shape)
  x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
  x = jnp.reshape(
      x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])[:, :, 1:, :]
  x = jnp.reshape(x, x_shape)
  return x


class SinusoidalPositionEmbedding(hk.Module):
  """Position encoding, using mixture of sinusoidal signals."""

  def __init__(
      self,
      dim: int,
      max_timescale: float = 1e4,
      cache_steps: int = 0,
      reverse_order: bool = False,
      clamp_len: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initialize a SinusoidalPositionEmbedding.

    Args:
      dim: Embedding dimension.
      max_timescale: Max x such that sin(t/x) appears in the signals. (Thus this
        should be some factor like 2*pi larger than the max input length.)
      cache_steps: The length of the memory.
      reverse_order: If set to True, position index is reversed.
      clamp_len: position beyond clamp_len will be reset to clamp_len, default
        to not clamping.
      name: Optional name for this Haiku module.
    """
    super().__init__(name=name)
    self._dim = dim
    self._max_timescale = max_timescale
    self._cache_steps = cache_steps
    self._reverse_order = reverse_order
    self._clamp_len = clamp_len

  def __call__(self, timesteps: int, batch_size: int) -> chex.Array:
    # _dim must be even and num_timescales-1 must be > 0
    assert self._dim >= 4
    assert self._dim % 2 == 0
    num_timescales = self._dim // 2
    min_timescale = 1.0
    full_length = timesteps + self._cache_steps

    if self._reverse_order:
      positions = jnp.arange(full_length - 1, -1, -1)
      positions = jnp.repeat(positions[None, :], batch_size, axis=0)
    else:
      if self._cache_steps > 0:
        positions = (get_pos_start(timesteps, batch_size)[:, None]
                     + jnp.arange(timesteps)[None, :])
      else:
        positions = jnp.arange(0, full_length)
        positions = jnp.repeat(positions[None, :], batch_size, axis=0)

    if self._clamp_len is not None:
      positions = jnp.minimum(positions, self._clamp_len)

    log_timescale_increment = (
        math.log(float(self._max_timescale) / float(min_timescale)) /
        (num_timescales - 1))
    inv_timescales = min_timescale * jnp.exp(
        jnp.arange(num_timescales) * -log_timescale_increment)
    scaled_time = positions[:, :, None] * inv_timescales[None, None, :]
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2)


class RelativePositionEmbedding(hk.Module):
  """Position encoding, using relative positions than absolute positions."""

  def __init__(
      self,
      dim: int,
      max_timescale: float = 1e4,
      clamp_len: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initialize a RelativePositionEmbedding.

    Args:
      dim: Embedding dimension.
      max_timescale: Max x such that sin(t/x) appears in the signals. (Thus this
        should be some factor like 2*pi larger than the max input length.)
      clamp_len: position beyond clamp_len will be reset to clamp_len, default
        to not clamping.
      name: Optional name for this Haiku module.
    """
    super().__init__(name=name)
    self._dim = dim
    self._sinusoidal_pos_emb = SinusoidalPositionEmbedding(
        dim=dim,
        max_timescale=max_timescale,
        reverse_order=True,
        clamp_len=clamp_len,
        name=name)

  def __call__(self, q: chex.Array, k: chex.Array) -> chex.Array:
    # Use key instead of query to obtain the length.
    batch_size, key_length, num_heads, head_dim = list(k.shape)
    # Content based addressing and global content bias
    r_w_bias = hk.get_parameter(
        'r_w_bias', [1, 1, num_heads, head_dim], init=init.VarianceScaling())
    content_score = jnp.einsum('bthd,bThd->bhtT', q + r_w_bias, k)

    # Relative position encoding
    rel_pos_emb = conv1(
        self._sinusoidal_pos_emb(key_length, batch_size), self._dim)
    rel_pos_emb = jnp.reshape(rel_pos_emb, [
        batch_size, key_length, num_heads, head_dim])

    # Content dependent positional bias and global positional bias
    r_r_bias = hk.get_parameter(
        'r_r_bias', [1, 1, num_heads, head_dim], init=init.VarianceScaling())
    rel_pos_score = jnp.einsum('bthd,bThd->bhtT', q + r_r_bias, rel_pos_emb)
    rel_pos_score = relative_shift(rel_pos_score)
    assert content_score.shape == rel_pos_score.shape
    return content_score + rel_pos_score


class Attention(hk.Module):
  """Attention."""

  def __init__(
      self,
      num_heads: int = 8,
      init_scale: float = 1.,
      hiddens_per_head: Optional[int] = None,
      with_final_bias: bool = True,
      final_init_scale_multiplier: float = 1.,
      relative_pos_emb: bool = False,
      relative_pos_clamp_len: Optional[int] = None,
      dropout_prob: float = 0.,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._init_scale = init_scale
    self._hiddens_per_head = hiddens_per_head
    self._with_final_bias = with_final_bias
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._relative_pos_emb = relative_pos_emb
    self._relative_pos_clamp_len = relative_pos_clamp_len
    self._dropout_prob = dropout_prob

  def __call__(
      self,
      x: chex.Array,
      y: chex.Array,
      mask: Optional[chex.Array] = None,
      should_reset: Optional[chex.Array] = None,
      cache_steps: int = 0,
      dropout_tile_dims: int = 0,
  ) -> chex.Array:
    hiddens_in = x.shape[-1]
    steps = x.shape[1]
    batch_size = x.shape[0]

    if self._hiddens_per_head is not None:
      qkv_hiddens = self._hiddens_per_head * self._num_heads
    else:
      qkv_hiddens = hiddens_in
    q = conv1(x, qkv_hiddens, init_scale=self._init_scale)
    k = conv1(y, qkv_hiddens, init_scale=self._init_scale)
    v = conv1(y, qkv_hiddens, init_scale=self._init_scale)

    # Reshape hiddens into multi-head attention here so the cache memory
    # layout is better.
    batch, q_time, _ = q.shape
    _, kv_time, _ = k.shape
    head_dim = qkv_hiddens // self._num_heads
    q = jnp.reshape(q, [batch, q_time, self._num_heads, head_dim])
    k = jnp.reshape(k, [batch, kv_time, self._num_heads, head_dim])
    v = jnp.reshape(v, [batch, kv_time, self._num_heads, head_dim])

    def update_cache(key, value, cache_steps=None, axis=1):
      """Update the state in hk.state."""
      cache_shape = list(value.shape)
      value_steps = cache_shape[axis]
      if cache_steps is not None:
        cache_shape[axis] += cache_steps
      cache = hk.get_state(
          key, shape=cache_shape, dtype=value.dtype, init=jnp.zeros)

      # Overwrite at index 0, then rotate timesteps left so what was just
      # inserted is first.
      value = jax.lax.dynamic_update_slice(
          cache, value, jnp.zeros(len(cache_shape), dtype=jnp.int32))
      value = jnp.roll(value, -value_steps, axis)
      hk.set_state(key, value)
      return value

    # Logic for using and updating cached activations (needed by transformer-xl
    # and incremental sampling).
    if cache_steps > 0:
      # Tells us how much of the cache should be used.
      cache_progress_idx = hk.get_state(
          'cache_progress_idx', [batch_size], dtype=jnp.int32, init=jnp.zeros)
      hk.set_state('cache_progress_idx', cache_progress_idx + steps)
      k = update_cache('k', k, cache_steps=cache_steps)
      v = update_cache('v', v, cache_steps=cache_steps)
      if mask is None:
        mask = jnp.ones((batch_size, 1, steps, steps))
      cache_mask = (jnp.arange(cache_steps - 1, -1, -1)[None, None, None, :]
                    < cache_progress_idx[:, None, None, None])
      cache_mask = jnp.broadcast_to(
          cache_mask,
          (batch_size, 1, steps, cache_steps)
          )
      mask = jnp.concatenate([cache_mask, mask], axis=-1)
    if should_reset is not None:
      if cache_steps > 0:
        should_reset = update_cache('should_reset', should_reset,
                                    cache_steps=cache_steps)
      reset_mask = get_reset_attention_mask(should_reset)[:, None, :, :]
      mask *= reset_mask[:, :, cache_steps:, :]

    if self._relative_pos_emb:
      attend_fn = RelativePositionEmbedding(
          dim=qkv_hiddens, clamp_len=self._relative_pos_clamp_len)
    else:
      attend_fn = None
    result = attend(
        q, k, v, mask=mask, attend_fn=attend_fn,
        dropout_prob=self._dropout_prob, dropout_tile_dims=dropout_tile_dims)
    return conv1(
        result,
        hiddens_in,
        with_bias=self._with_final_bias,
        init_scale=self._final_init_scale)


class SelfAttention(Attention):
  """SelfAttention."""

  def __call__(
      self,
      x: chex.Array,
      y: chex.Array = None,  # ignored.
      mask: Optional[chex.Array] = None,
      should_reset: Optional[chex.Array] = None,
      cache_steps: int = 0,
      dropout_tile_dims: int = 0,
  ) -> chex.Array:
    return super().__call__(
        x=x, y=x, mask=mask, should_reset=should_reset, cache_steps=cache_steps,
        dropout_tile_dims=dropout_tile_dims)


class CausalSelfAttention(SelfAttention):
  """CausalSelfAttention."""

  def __call__(
      self,
      x: chex.Array,
      y: chex.Array = None,  # ignored.
      mask: Optional[chex.Array] = None,
      should_reset: Optional[chex.Array] = None,
      cache_steps: int = 0,
  ) -> chex.Array:
    timesteps = x.shape[1]
    batch_size = x.shape[0]
    t = jnp.arange(timesteps, dtype=jnp.int32)
    causal_mask = (t[:, None] >= t[None, :])[None, None, :, :]
    causal_mask = causal_mask.astype(x.dtype)
    if mask is None:
      mask = jnp.broadcast_to(
          causal_mask,
          (batch_size, 1, timesteps, timesteps)
          )
    else:
      mask *= causal_mask
    return super().__call__(
        x=x, mask=mask, should_reset=should_reset, cache_steps=cache_steps)


class SelfAttentionBlock(hk.Module):
  """SelfAttentionBlock."""

  def __init__(
      self,
      causal: bool = False,
      num_heads: int = 8,
      dropout_prob: float = 0.1,
      dropout_attn_prob: float = 0.,
      init_scale: float = 1.,
      relative_pos_emb: bool = False,
      relative_pos_clamp_len: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._causal = causal
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._init_scale = init_scale
    self._relative_pos_emb = relative_pos_emb
    self._relative_pos_clamp_len = relative_pos_clamp_len

  def __call__(
      self,
      x: chex.Array,
      mask: Optional[chex.Array] = None,
      should_reset: Optional[chex.Array] = None,
      cache_steps: int = 0,
  ) -> chex.Array:
    if self._causal:
      x = CausalSelfAttention(
          num_heads=self._num_heads,
          init_scale=self._init_scale,
          relative_pos_emb=self._relative_pos_emb,
          relative_pos_clamp_len=self._relative_pos_clamp_len,
          dropout_prob=self._dropout_attn_prob)(
              x=x,
              mask=mask,
              should_reset=should_reset,
              cache_steps=cache_steps)
    else:
      x = SelfAttention(
          num_heads=self._num_heads,
          init_scale=self._init_scale,
          dropout_prob=self._dropout_attn_prob,
          relative_pos_emb=self._relative_pos_emb)(x=x, mask=mask)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, x)


class CondAttentionBlock(hk.Module):
  """CondAttentionBlock."""

  def __init__(
      self,
      num_heads: int = 8,
      dropout_prob: float = 0.1,
      dropout_attn_prob: float = 0.,
      init_scale: float = 1.,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._init_scale = init_scale

  def __call__(self, x, cond, mask=None):
    x = Attention(num_heads=self._num_heads,
                  init_scale=self._init_scale,
                  dropout_prob=self._dropout_attn_prob)(x, cond, mask)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, x)


class DenseBlock(hk.Module):
  """Dense block."""

  def __init__(
      self,
      widening_factor: int = 4,
      dropout_prob: float = 0.1,
      init_scale: float = 1.,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._init_scale = init_scale

  def __call__(self, x: chex.Array) -> chex.Array:
    hiddens = x.shape[-1]
    x = conv1(x, num_units=self._widening_factor * hiddens,
              init_scale=self._init_scale)
    x = jax.nn.gelu(x)
    x = conv1(x, num_units=hiddens, init_scale=self._init_scale)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, x)


class TransformerBlock(hk.Module):
  """TransformerBlock."""

  def __init__(
      self,
      causal: bool = True,
      widening_factor: int = 4,
      dropout_prob: float = 0.1,
      dropout_attn_prob: float = 0.,
      num_heads: int = 8,
      self_att_init_scale: float = 1.,
      cond_att_init_scale: float = 1.,
      dense_init_scale: float = 1.,
      relative_pos_emb: bool = False,
      relative_pos_clamp_len: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._causal = causal
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._num_heads = num_heads
    self._self_att_init_scale = self_att_init_scale
    self._cond_att_init_scale = cond_att_init_scale
    self._dense_init_scale = dense_init_scale
    self._relative_pos_emb = relative_pos_emb
    self._relative_pos_clamp_len = relative_pos_clamp_len

  def __call__(
      self,
      x: chex.Array,
      cond: Optional[chex.Array] = None,
      mask: Optional[chex.Array] = None,
      cond_mask: Optional[chex.Array] = None,
      is_training: bool = True,
      should_reset: Optional[chex.Array] = None,
      cache_steps: int = 0,
  ) -> chex.Array:
    dropout_prob = self._dropout_prob if is_training else 0.0
    dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0
    x += SelfAttentionBlock(
        causal=self._causal,
        num_heads=self._num_heads,
        dropout_prob=dropout_prob,
        dropout_attn_prob=dropout_attn_prob,
        init_scale=self._self_att_init_scale,
        relative_pos_emb=self._relative_pos_emb,
        relative_pos_clamp_len=self._relative_pos_clamp_len)(
            layer_norm(x),
            mask=mask,
            should_reset=should_reset,
            cache_steps=cache_steps)
    if cond is not None:
      x += CondAttentionBlock(
          num_heads=self._num_heads,
          dropout_prob=dropout_prob,
          dropout_attn_prob=dropout_attn_prob,
          init_scale=self._cond_att_init_scale)(
              layer_norm(x), layer_norm(cond), mask=cond_mask)
    x += DenseBlock(
        widening_factor=self._widening_factor,
        dropout_prob=dropout_prob,
        init_scale=self._dense_init_scale)(
            layer_norm(x))
    return x
