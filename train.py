import time
import numpy as np
import tensorflow as tf
from models import NetWork
from utils import adj_to_bias
from utils import load_data
from utils import construct_labels_with_scores
from utils import load_text_feat
from sklearn import metrics

def train(args, train_arr, test_arr, dataset, times, fold):
    # training params
    batch_size = 1
    nb_epochs = args.epoch
    lr = args.lr
    hid_units = [args.hid_units]
    dense0 = args.dense0
    dense1 = args.dense1
    n_layers = args.layers
    attention_drop = args.attention_drop
    feedforward_drop = args.feedforward_drop

    model = NetWork

    print('----- Archi. hyperparams -----')
    print(' dataset: ' + str(args.dataset))
    print(' epoch: ' + str(nb_epochs))
    print(' lr: ' + str(lr))
    print(' Graph units: ' + str(hid_units))
    print(' dense0: ' + str(dense0))
    print(' dense1: ' + str(dense1))
    print(' layers num: ' + str(n_layers))
    print(' attention_drop:' + str(attention_drop))
    print(' feedforward_drop:' + str(feedforward_drop))

    """
        labels_shape:(relation_num,3) represents adj.txt
        y_train.shape:(n*m,1), if there is relation between disease and biomarker, y_train=1, otherwise 0.01
        y_test.shape:(n*m,1)
        train_mask: its data type is bool
        features.shape:(n+m,n+m), feature matrix
        interaction.shape:(n+m,n+m)
        n,m denote the number of diseases and biomarkers, respectively
    """

    interaction, features, y_train, y_test, train_mask, test_mask, labels = load_data(train_arr, test_arr, dataset)
    dis_text_feat, bio_text_feat = load_text_feat(train_arr, dataset)


    nb_nodes = features.shape[0]  # n+m
    ft_size = features.shape[1]  # n+m
    print('nb_nodes:######', nb_nodes)

    features = features[np.newaxis]

    interaction = interaction[np.newaxis]
    biases = adj_to_bias(interaction, [nb_nodes], nhood=1)

    n = np.max(labels[:, 0])
    m = np.max(labels[:, 1])
    n = n.astype(np.int32)
    m = m.astype(np.int32)
    entry_size = n * m

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            # define tf input graph
            feature_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            disease_text_feat = tf.compat.v1.placeholder(dtype=tf.float32, shape=(n, 768))  # 768 is the dimension of scibert output
            biomarker_text_feat = tf.compat.v1.placeholder(dtype=tf.float32, shape=(m, 768)) # 768 is the dimension of scibert output
            bias_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
            msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
            neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
            attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
            is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())

        """
            define output graph
            U_shape:(n,layers*hidden_size)
            V_shape:(m,layers*hidden_size)
            U, V represent diseases graph features and biomarker graph features, respectively
        """

        U, V = model.encoder(feature_in, n, attn_drop, ffd_drop, bias_mat=bias_in, hid_units=hid_units, n_layers=n_layers)

        # combine graph features and text features
        scores = model.combine_feat(U, V, disease_text_feat, biomarker_text_feat, dense0, dense1, is_training=is_train)

        loss = model.loss_sum(scores, lbl_in, msk_in, neg_msk)

        train_op = model.training(loss, lr)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run(init_op)

            test_negative_samples = np.loadtxt(f'data/{dataset}/data_dir/{times}/{fold}/test_label_neg.txt')
            label_neg = np.loadtxt(f'data/{dataset}/data_dir/{times}/{fold}/train_label_neg.txt')

            neg_mask = np.zeros((n, m))
            test_neg_mask = np.zeros((n, m))

            for i in range(len(label_neg)):
                a = int(label_neg[i, 0])
                b = int(label_neg[i, 1])
                neg_mask[a, b] = 1

            for i in range(len(test_negative_samples)):
                a = int(test_negative_samples[i, 0])
                b = int(test_negative_samples[i, 1])
                test_neg_mask[a, b] = 1

            neg_mask = np.reshape(neg_mask, [-1, 1])
            test_neg_mask = np.reshape(test_neg_mask, [-1, 1])

            best_test_auc = 0.0
            for epoch in range(nb_epochs):
                t = time.time()
                tr_loss = 0.0
                tr_step = 0
                tr_size = features.shape[0]

                # tr_size=1
                while tr_step * batch_size < tr_size:
                    _, train_out_come, loss_value_tr = sess.run([train_op, scores, loss],
                                                                feed_dict={
                                                                    feature_in: features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                    disease_text_feat: dis_text_feat,
                                                                    biomarker_text_feat: bio_text_feat,
                                                                    bias_in: biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                    lbl_in: y_train,
                                                                    msk_in: train_mask,
                                                                    neg_msk: neg_mask,
                                                                    is_train: True,
                                                                    attn_drop: attention_drop,
                                                                    ffd_drop: feedforward_drop})
                    tr_loss += loss_value_tr
                    tr_step += 1

                ts_size = features.shape[0]
                ts_step = 0
                ts_loss = 0.0

                while ts_step * batch_size < ts_size:
                    test_out_come, loss_value_ts = sess.run([scores, loss],
                                                            feed_dict={
                                                                feature_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                disease_text_feat: dis_text_feat,
                                                                biomarker_text_feat: bio_text_feat,
                                                                bias_in: biases[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                lbl_in: y_test,
                                                                msk_in: test_mask,
                                                                neg_msk: test_neg_mask,
                                                                is_train: False,
                                                                attn_drop: 0.0, ffd_drop: 0.0})
                    ts_loss += loss_value_ts
                    ts_step += 1

                train_out_come = train_out_come.reshape((n, m))
                train_labels, train_score = construct_labels_with_scores(train_out_come, labels, train_arr, label_neg)

                test_out_come = test_out_come.reshape((n, m))
                test_labels, test_score = construct_labels_with_scores(test_out_come, labels, test_arr, test_negative_samples)

                train_auc = metrics.roc_auc_score(train_labels, train_score)
                test_auc = metrics.roc_auc_score(test_labels, test_score)

                if best_test_auc < test_auc:
                    best_test_auc = test_auc
                    best_test_labels = test_labels
                    best_test_score = test_score

                if (epoch+1) % 10 == 0 or epoch == 0:
                    print('Epoch: %04d | train_loss = %.5f, train_auc = %.5f, test_loss = %.5f, test_auc = %.5f, time = %.5f' % ((epoch+1), tr_loss / tr_step, train_auc, ts_loss / ts_step, test_auc, time.time() - t))
            return best_test_labels, best_test_score