from typing import Dict
import jax.numpy as jnp
import haiku as hk

ArrayNest = Dict[str, jnp.ndarray]

class CPC:
  class Encoder(hk.Module):
    """Takes a dict of 3d arrays indexed by batch_size, time, num_events."""

    def __call__(self, inputs: ArrayNest, is_training: bool) -> jnp.ndarray:
      """{'duration': (TensorShape([16, 64, 64]), tf.int64), 
      'event_type': (TensorShape([16, 64, 64]), tf.int64), 
      'start_time': (TensorShape([16, 64, 64]), tf.int64), 
      'start_time_features': (TensorShape([16, 64, 64, 9]), tf.int64), 
      'time': (TensorShape([16, 64]), tf.int64), 
      'time_features': (TensorShape([16, 64, 1, 9]), tf.int64)}
      """
      # inputs['duration']
      # TODO: un-hardcode vocab_size and embed_dim using init
      vocab_size = 5103
      embed_dim = 256
      event_embedding = hk.Embed(vocab_size, embed_dim)(inputs["event_type"])  # [bs, t, events, dim]
      year_embedding = hk.Embed(100, embed_dim)(inputs["time_features"][..., 0]-1970)  # [bs, t, 1, dim]
      month_embedding = hk.Embed(12, embed_dim)(inputs["time_features"][..., 1]-1)  # [bs, t, 1, dim]
      day_embedding = hk.Embed(31, embed_dim)(inputs["time_features"][..., 2]-1)  # [bs, t, 1, dim]
      hour_embedding = hk.Embed(24, embed_dim)(inputs["time_features"][..., 3])  # [bs, t, 1, dim]
      min_embedding = hk.Embed(60, embed_dim)(inputs["time_features"][..., 4])  # [bs, t, 1, dim]
      week_embedding = hk.Embed(7, embed_dim)(inputs["time_features"][..., 6])  # [bs, t, 1, dim]
      # TODO: look up numpy broadcasting
      # TODO: problem: adding time information, this might make prediction too easy
      # embedding = event_embedding + year_embedding + month_embedding + day_embedding + hour_embedding + min_embedding + week_embedding
      embedding = event_embedding
      batch_size = embedding.shape[0]
      seq_len = embedding.shape[1]
      num_events = embedding.shape[2]
      embedding_btd = jnp.reshape(
          embedding, [batch_size * seq_len, num_events, embed_dim])
      mask = jnp.greater(inputs["event_type"], 0)
      event_mask = jnp.reshape(mask, [batch_size * seq_len, num_events])
      event_mask = jnp.logical_and(event_mask[:, None, :, None], event_mask[:, None, None, :])
      time_mask = mask.any(axis=2)
      time_mask = jnp.logical_and(time_mask[:, None, :, None], time_mask[:, None, None, :])
      transformer = Transformer(num_heads=4, 
                                num_layers=6,
                                causal=False,  
                                dropout_rate=0.1)
      # TODO: add temporal down sampling
      output_embedding = transformer(embedding_btd, 
                                     mask=event_mask, 
                                     is_training=is_training)
      output_embedding = jnp.reshape(
          output_embedding, [batch_size, seq_len, num_events, embed_dim])
      output_embedding = output_embedding.mean(axis=2)  # [bs, t, dim]
      return output_embedding, time_mask

  class CausalTemporalEncoder(hk.Module):
    def __call__(self, z_t: jnp.ndarray, mask: jnp.ndarray, is_training: bool
                 ) -> jnp.ndarray:
      """z_t: [bs, t, dim]"""
      transformer = Transformer(num_heads=4, num_layers=6, causal=True, dropout_rate=0.1)
      c_t = transformer(z_t, mask, is_training=is_training)
      return c_t

  class Model(hk.Module):
    def __call__(self, inputs: ArrayNest, is_training: bool) -> jnp.ndarray:
      z_t, mask = CPC.Encoder()(inputs, is_training)  # [bs, t, dim]
      c_t = CPC.CausalTemporalEncoder()(z_t, mask, is_training) # [bs, t, dim]
      # c0, c1, c2, ..., c63
      # z0, z1, z2, ..., z63
      num_steps_to_ignore = 1
      num_steps_to_predict = 10
      embed_dim = z_t.shape[-1]
      batch_size = z_t.shape[0]
      seq_len = z_t.shape[1]
      loss = 0
      mask = mask.any(axis=1)
      for k in range(num_steps_to_ignore, num_steps_to_predict):
        # example: for k = 5
        # predictors c0, c1, c2, ..., c58
        # targets    z5, z6, z7, ..., z63
        # TODO: Check masking
        targets = z_t[:, k:]
        targets_mask = mask[:, k:].any(axis=2)
        predictors = c_t[:, :-k]
        predictors_mask = mask[:, :, :-k].any(axis=1)
        target_predictor_mask = jnp.logical_and(predictors_mask, targets_mask)
        # target_predictor_mask = jnp.logical_and(
        #     target_predictor_mask[:, None, :, None], 
        #     target_predictor_mask[:, None, None, :])
        # print("before step_loss, target predictor mask shape", 
        #       target_predictor_mask.shape)
        target_predictor_mask = target_predictor_mask.reshape([-1])
        predictions = hk.Linear(embed_dim)(predictors)
        flat_targets = jnp.reshape(targets, [-1, embed_dim])
        flat_predictions = jnp.reshape(predictions, [-1, embed_dim])
        logits = jnp.dot(flat_predictions, flat_targets.T)
        log_probs = jax.nn.log_softmax(logits)
        num_items = batch_size*(seq_len-k)
        labels = jnp.arange(num_items)[...,None]
        # print("before step_loss, log_probs", log_probs.shape)
        # print("before step_loss, labels shape", labels.shape)
        # print("after step_loss, target predictor mask shape", 
        #       target_predictor_mask.shape)
        step_loss = jnp.take_along_axis(-log_probs, labels, axis=-1)[..., 0]
        # print("after step_loss, step_loss shape", step_loss.shape)
        loss += jnp.sum(step_loss * target_predictor_mask) / jnp.sum(target_predictor_mask)
      return loss
