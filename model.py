import tensorflow as tf

BATCH_SIZE, EPOCHS, ROUTING_ITERATIONS = 128, 100, 3
LAMBDA = 0.5
M_PLUS = 0.9
M_MINUS = 0.1
EPSILON = 1e-15
RECONSTRUCTION_LOSS = 5e-4 * 784
IMAGE_HEIGHT = IMAGE_WIDTH = 28
CONV_KERNEL_SIZE = 3
CONV_FILTERS = 256
CONV_STRIDE = 1
CAPSULE_STRIDE = 2
CAPSULE_KERNEL_SIZE = 9
CAPSULE_DIM = 8
CAPSULE_FILTERS = 32
CAPSULE_OUT_DIM = 16
CLASSES = 10
RECON_NET_1, RECON_NET_2, RECON_NET_3 = 512, 1024, IMAGE_HEIGHT * IMAGE_WIDTH


def squash(x, axis):
    sq_norm = tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)
    scalar_factor = sq_norm / (1 + sq_norm) / tf.sqrt(sq_norm + EPSILON)
    return tf.multiply(scalar_factor, x)


def placeholder(type, shape, name):
    return tf.placeholder(type, shape=shape, name=name)


def conv_layer(ip, filters, kernel_size, strides, name):
    return tf.layers.conv2d(ip, filters, kernel_size, strides, padding='VALID',
                            use_bias=True, bias_initializer=tf.zeros_initializer,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=tf.nn.relu, name=name)


def main():
    X = placeholder(tf.float32, (None, IMAGE_WIDTH, IMAGE_HEIGHT, 1), 'X')
    y = placeholder(tf.float32, (None, CLASSES), 'y')

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(X, CONV_FILTERS, CONV_KERNEL_SIZE, CONV_STRIDE, 'CONV1')

    with tf.variable_scope('prime_caps'):
        prime_caps = conv_layer(conv1,CAPSULE_DIM * CAPSULE_FILTERS, CAPSULE_KERNEL_SIZE,
                                CAPSULE_STRIDE, 'prime_caps')

        prime_caps = tf.reshape(prime_caps, shape=(BATCH_SIZE,
                                                   6 * 6 * CAPSULE_FILTERS, 1,
                                                   CAPSULE_DIM, 1))
        prime_caps = tf.tile(prime_caps, [1, 1, CLASSES, 1, 1])

        prime_caps = squash(prime_caps, axis=3)

    W = tf.get_variable('W', dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.1),
                        shape=(1, 6 * 6 * CAPSULE_FILTERS,
                               CLASSES, CAPSULE_DIM, CAPSULE_OUT_DIM))

    W = tf.tile(W, [BATCH_SIZE, 1, 1, 1, 1], 'tiledW')

    with tf.variable_scope('capsule_to_digits'):
        u = tf.matmul(W, prime_caps, transpose_a=True)
        u = tf.reshape(tf.squeeze(u), (-1, 6 * 6 * CAPSULE_FILTERS,
                                       CAPSULE_OUT_DIM, CLASSES), name='u')

        b = tf.zeros((BATCH_SIZE, 6 * 6 * CAPSULE_FILTERS, CLASSES), dtype=tf.float32, name='b')

        for routing_iter in range(ROUTING_ITERATIONS):
            with tf.variable_scope('route_'+str(routing_iter)):
                c = tf.nn.softmax(b, dim=2)
                c = tf.reshape(c, shape=(BATCH_SIZE, 6 * 6 * CAPSULE_FILTERS, 1, CLASSES))

                s = tf.reduce_sum(u * c, axis=1, keep_dims=False)

                v = squash(s, axis=1)

                if routing_iter < ROUTING_ITERATIONS - 1:
                    v_routed = tf.reshape(v, shape=(-1, 1, CAPSULE_OUT_DIM, CLASSES))

                    uv = tf.reduce_sum(u * v_routed, axis=2, name='uv')

                    b += uv

