import tensorflow as tf
from models import GAT
from utils import glorot
from utils import masked_accuracy

# Batch Normalization
def bn_layer(x, is_training=True, moving_decay=0.9, eps=1e-5):
    shape = x.shape

    param_shape = shape[-1]

    # learnable parameters，y=gamma*x+beta
    gamma = tf.Variable(tf.ones(param_shape))
    beta = tf.Variable(tf.zeros(param_shape))

    # calculate mean and var
    axes = list(range(len(shape) - 1))
    batch_mean, batch_var = tf.nn.moments(x, axes)

    ema = tf.train.ExponentialMovingAverage(moving_decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    # update the mean and var when training
    # use the latest mean and var when validating
    mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

class NetWork():
    def training(loss, lr):
        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        train_op = opt.minimize(loss)
        return train_op

    def combine_feat(U, V, disease_text_feat, mirna_text_feat, dense0, dense1, is_training):
        """
            n: number_of_disease
            m: number_of_biomarker
            U_shape:(n,hidden_dim)
            V_shape:(m,hidden_dim)
            disease_text_feat_shape:(n,768)
            biomarker_text_feat_shape:(m,768)
        """
        with tf.compat.v1.variable_scope("deco"):
            w_in_1 = glorot([U.shape[1].value, dense1])
            w_in_2 = glorot([disease_text_feat.shape[1].value, dense0])
            w_in_3 = glorot([dense0, dense1])
            w_in_4 = glorot([V.shape[1].value, dense1])
            w_in_5 = glorot([mirna_text_feat.shape[1].value, dense0])
            w_in_6 = glorot([dense0, dense1])

            w_in_7 = glorot([dense1, dense1])
            w_in_8 = glorot([dense1, dense1])

        dense2 = tf.matmul(U, w_in_1)
        dense3 = tf.matmul(disease_text_feat, w_in_2)
        dense4 = tf.matmul(V, w_in_4)
        dense5 = tf.matmul(mirna_text_feat, w_in_5)

        dense2 = tf.nn.elu(bn_layer(dense2, is_training=is_training))  # disease graph features
        dense3 = tf.nn.elu(bn_layer(dense3, is_training=is_training))  # disease text features
        dense4 = tf.nn.elu(bn_layer(dense4, is_training=is_training))  # biomarker graph features
        dense5 = tf.nn.elu(bn_layer(dense5, is_training=is_training))  # biomarker text features

        dense3 = tf.matmul(dense3, w_in_3)
        dense5 = tf.matmul(dense5, w_in_6)
        dense3 = tf.nn.elu(bn_layer(dense3, is_training=is_training))
        dense5 = tf.nn.elu(bn_layer(dense5, is_training=is_training))

        dense2 = tf.matmul(dense2, w_in_7)
        dense3 = tf.matmul(dense3, w_in_8)

        logits1 = tf.matmul(dense2, tf.transpose(dense4))  # (n,m)
        logits2 = tf.matmul(dense3, tf.transpose(dense5))  # (n,m)
        logits = logits1 + logits2

        logits = tf.reshape(logits, [-1, 1])

        return tf.nn.sigmoid(logits)

    """
        GAT encoder: use GAT to generate graph features
        U, V represent diseases and biomarkers graph features
    """
    def encoder(inputs, n, attn_drop, ffd_drop,
                bias_mat, hid_units, n_layers, activation=tf.nn.elu):
        print(inputs.dtype)
        for _ in range(n_layers):
            attn_temp, coefs = GAT.attn_head(inputs, bias_mat=bias_mat, hidden_size=hid_units[0], activation=activation, in_drop=ffd_drop, coef_drop=attn_drop)
            inputs = attn_temp[tf.newaxis]
        h_1 = attn_temp
        U = h_1[0:n, :]
        V = h_1[n:, :]
        return U, V

    def loss_sum(scores, lbl_in, msk_in, neg_msk):

        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)

        return loss_basic