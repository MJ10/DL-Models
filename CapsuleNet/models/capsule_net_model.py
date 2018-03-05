from base.base_model import BaseModel
import tensorflow as tf
from utils.ops import squash, safe_norm


class CapsuleNet(BaseModel):
    def __init__(self, config):
        super(CapsuleNet, self).__init__(config)
        # Primary Capsules
        self.caps1_n_maps = 32
        self.caps1_n_caps = self.caps1_n_maps * 6 * 6
        self.caps1_n_dims = 8

        self.conv1_params = {
            "filters": 256,
            "kernel_size": 9,
            "strides": 1,
            "padding": "valid",
            "activation": tf.nn.relu,
        }

        self.conv2_params = {
            "filters": self.caps1_n_maps * self.caps1_n_dims,  # 256 convolutional filters
            "kernel_size": 9,
            "strides": 2,
            "padding": "valid",
            "activation": tf.nn.relu
        }

        self.caps2_n_caps = 10
        self.caps2_n_dims = 16

        self.init_sigma = 0.01

        self.alpha = 0.0005

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        batch_size = tf.shape(self.X)[0]

        # Input Layer
        conv1 = tf.layers.conv2d(self.X, name='conv1', **self.conv1_params)
        conv2 = tf.layers.conv2d(conv1, name='conv2', **self.conv2_params)

        # Primary Capsules Layer
        caps1_raw = tf.reshape(conv2, shape=[-1, self.caps1_n_caps, self.caps1_n_dims],
                               name='caps1_raw')

        self.caps1_output = squash(caps1_raw, name="caps1_output")

        # Digit Capsules Layer
        W_init = tf.random_normal(shape=(1,
                                         self.caps1_n_caps,
                                         self.caps2_n_caps,
                                         self.caps2_n_dims,
                                         self.caps1_n_dims),
                                  stddev=self.init_sigma, dtype=tf.float32,
                                  name='W_init')
        W = tf.Variable(W_init, name='W')
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name='W_tiled')

        caps1_output_ex = tf.expand_dims(self.caps1_output, -1, name='caps1_output_ex')
        caps1_output_tile = tf.expand_dims(caps1_output_ex, 2, name='caps1_output_tile')
        self.caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.caps2_n_caps, 1, 1],
                                          name='caps1_output_tiled')

        self.caps2_predicted = tf.matmul(W_tiled, self.caps1_output_tiled, name='caps2_predicted')

        # Routing By Agreement
        # Round 1
        raw_weights = tf.zeros([batch_size, self.caps1_n_caps, self.caps2_n_caps, 1, 1],
                               dtype=tf.float32, name='raw_weights')
        routing_weights = tf.nn.softmax(raw_weights, dim=2, name='routing_weights')
        weighted_pred = tf.multiply(routing_weights, self.caps2_predicted, name='weighted_pred')
        weighted_sum = tf.reduce_sum(weighted_pred, axis=1, keep_dims=True,
                                     name='weighted_sum')
        caps2_output_round1 = squash(weighted_sum, axis=-2, name='caps2_output_round1')

        caps2_output_round1_tiled = tf.tile(caps2_output_round1,
                                            [1, self.caps1_n_caps, 1, 1, 1],
                                            name='caps2_output_round1_tiled')
        agreement = tf.matmul(self.caps2_predicted, caps2_output_round1_tiled,
                              transpose_a=True,
                              name='agreement')
        # Round 2
        raw_weights_round2 = tf.add(raw_weights, agreement, name='raw_weights_round2')

        routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2,
                                               name="routing_weights_round2")
        weighted_predictions_round2 = tf.multiply(routing_weights_round2, self.caps2_predicted,
                                                  name="weighted_predictions_round2")
        weighted_sum_round2 = tf.reduce_sum(weighted_predictions_round2,
                                            axis=1, keep_dims=True,
                                            name="weighted_sum_round2")
        caps2_output_round2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round2")

        caps2_output_round2_tiled = tf.tile(caps2_output_round2,
                                            [1, self.caps1_n_caps, 1, 1, 1],
                                            name='caps2_output_round2_tiled')
        agreement_round2 = tf.matmul(self.caps2_predicted, caps2_output_round2_tiled,
                                     transpose_a=True,
                                     name='agreement_round2')

        # Round 3
        raw_weights_round3 = tf.add(raw_weights_round2, agreement_round2,
                                    name='raw_weights_round3')
        routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2,
                                               name='routing_weights_round3')

        weighted_predictions_round3 = tf.multiply(routing_weights_round3, self.caps2_predicted,
                                                  name="weighted_predictions_round3")
        weighted_sum_round3 = tf.reduce_sum(weighted_predictions_round3,
                                            axis=1, keep_dims=True,
                                            name="weighted_sum_round3")
        caps2_output_round3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round3")

        self.caps2_output = caps2_output_round3

        # Estimate Probabilities
        y_prob = safe_norm(self.caps2_output, axis=-2, name='y_prob')
        y_prob_argmax = tf.argmax(y_prob, axis=2, name='y_prob_argmax')
        self.y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2], name='y_pred')

        # Margin Loss
        mplus = 0.9
        mminus = 0.1
        lambda_ = 0.5

        T = tf.one_hot(self.y, depth=self.caps2_n_caps, name='T')
        caps2_output_norm = safe_norm(self.caps2_output, axis=-2, keep_dims=True,
                                      name='caps2_output_norm')

        self.present_err = tf.reshape(tf.square(tf.maximum(0., mplus - caps2_output_norm)),
                                      shape=(-1, 10), name='present_err')
        self.absent_err = tf.reshape(tf.square(tf.maximum(0., caps2_output_norm - mminus)),
                                     shape=(-1, 10), name='absent_err')
        ls = T * self.present_err + lambda_ * (1. - T) * self.absent_err
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(ls, axis=1), name='margin_loss')

        # Reconstruction
        mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                       name='mask_with_labels')
        recon_target = tf.cond(mask_with_labels, lambda: self.y, lambda: self.y_pred,
                               name='recon_target')
        self.recon_mask = tf.reshape(tf.one_hot(recon_target, depth=self.caps2_n_caps),
                                     shape=[-1, 1, self.caps2_n_caps, 1, 1],
                                     name='recon_mask')
        self.decoder_in = tf.reshape(self.caps2_output * self.recon_mask,
                                     shape=[-1, self.caps2_n_caps * self.caps2_n_dims],
                                     name='decoder_in')
        n_hidden = 512
        n_hidden2 = 1024
        n_out = 28 * 28

        with tf.name_scope('decoder'):
            hidden1 = tf.layers.dense(self.decoder_in, n_hidden, activation=tf.nn.relu,
                                      name='hidden1')
            hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                                      name='hidden2')
            self.decoder_output = tf.layers.dense(hidden2, n_out, activation=tf.nn.sigmoid,
                                                  name='decoder_output')

        X_flat = tf.reshape(self.X, [-1, n_out], name='X_flat')
        self.recon_loss = tf.reduce_mean(tf.square(X_flat - self.decoder_output), name='recon_loss')

        self.loss = self.margin_loss + self.alpha * self.recon_loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_pred), dtype=tf.float32), name='accuracy')
        self.optimizer = tf.train.AdamOptimizer()
        self.training_step = self.optimizer.minimize(self.loss, name='training_step')
        self.init = tf.global_variables_initializer()

    def init_saver(self):
        self.saver = tf.train.Saver()
