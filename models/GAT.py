import tensorflow as tf
from utils import glorot

conv1d = tf.layers.conv1d
        
def attn_head(seq, hidden_size, bias_mat, activation, in_drop=0.0, coef_drop=0.0):
  if in_drop != 0.0:
     seq = tf.nn.dropout(seq, 1.0 - in_drop)

  #the first layer seq_shape:(1,n+m, n+m)
  #other layer seq_shape:(1,n+m, hidden_size)
  seq_fts = seq

  latent_factor_size = hidden_size

  w_2 = glorot([2*seq_fts.shape[2].value,latent_factor_size])

  #f_1_shape: (1, n+m, 1), where n+m = seq_len - kernel_size(1) + 1
  f_1 = tf.layers.conv1d(seq_fts, 1, 1)
  f_2 = tf.layers.conv1d(seq_fts, 1, 1)

  #logits_shape: (1, n+m, n+m)
  logits = f_1 + tf.transpose(f_2, [0, 2, 1])

  #coefs: weighted matrix
  coefs = tf.nn.softmax(tf.nn.leaky_relu(logits[0]) + bias_mat[0])

  if coef_drop != 0.0:
     coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
  if in_drop != 0.0:
     seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

  neigh_embs = tf.matmul(coefs, seq_fts[0])

  #weighted sum
  final_embs = activation(tf.matmul(tf.concat([seq_fts[0],neigh_embs],axis=-1),w_2))

  return final_embs, coefs