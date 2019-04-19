import tensorflow as tf
from tensorpack import *


class OctConv2D(tf.keras.layers.Layer):
    def __init__(self, scope, kernel_size, num_outputs, alpha_out, activation=BNReLU):
        super(OctConv2D, self).__init__()

        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.alpha_out = alpha_out

        self.up_sampler = tf.keras.layers.UpSampling2D()
        self.down_sampler = tf.keras.layers.AveragePooling2D(padding='same')
        self.activation = activation
        self.scope = scope

    def build(self, input_shape):
        num_in_channel_low = int(input_shape[0][-1])
        num_in_channel_high = int(input_shape[1][-1])

        num_out_channel_low = int(self.alpha_out * self.num_outputs)
        num_out_channel_high = self.num_outputs - num_out_channel_low

        with tf.variable_scope(self.scope):
            self.filter_ll = tf.get_variable('w_ll', shape=[self.kernel_size, self.kernel_size, num_in_channel_low,
                                                            num_out_channel_low])
            self.filter_hl = tf.get_variable('w_hl', shape=[self.kernel_size, self.kernel_size, num_in_channel_high,
                                                            num_out_channel_low])

            self.filter_hh = tf.get_variable('w_hh', shape=[self.kernel_size, self.kernel_size, num_in_channel_high,
                                                            num_out_channel_high])
            self.filter_lh = tf.get_variable('w_lh', shape=[self.kernel_size, self.kernel_size, num_in_channel_low,
                                                            num_out_channel_high])
            # add bias if no batch normalization
            if self.activation != BNReLU and self.activation != BatchNorm and self.activation != tf.nn.batch_normalization:
                self.bias_l = tf.get_variable('b_l', shape=[num_out_channel_low])
                self.bias_h = tf.get_variable('b_h', shape=[num_out_channel_high])

    def call(self, inputs, **kwargs):
        x_low = inputs[0]
        x_high = inputs[1]

        with tf.variable_scope(self.scope):
            with tf.variable_scope('low'):
                ll = tf.nn.conv2d(x_low, self.filter_ll, (1, 1, 1, 1), 'SAME') if int(x_low.shape[-1]) != 0 and int(self.filter_ll.shape[-1]) != 0 else 0
                hl = tf.nn.conv2d((self.down_sampler(x_high)), self.filter_hl, (1, 1, 1, 1), 'SAME') if int(x_high.shape[-1]) != 0 and int(self.filter_hl.shape[-1]) != 0 else 0
                y_low = ll + hl

                if y_low != 0:
                    if self.activation != BNReLU and self.activation != BatchNorm and self.activation != tf.nn.batch_normalization:
                        tf.nn.bias_add(y_low, self.bias_l)

                    if self.activation is not None:
                        y_low = self.activation(y_low)

            with tf.variable_scope('high'):
                lh = tf.nn.conv2d(self.up_sampler(x_low), self.filter_lh, (1, 1, 1, 1), 'SAME') if int(x_low.shape[-1]) != 0 and int(self.filter_lh.shape[-1]) != 0 else 0
                hh = tf.nn.conv2d(x_high, self.filter_hh, (1, 1, 1, 1), 'SAME') if int(x_high.shape[-1]) != 0 and int(self.filter_hh.shape[-1]) != 0 else 0

                if lh != 0 and hh != 0 and lh.shape[1] > hh.shape[1]:
                    lh = lh[:, :hh.shape[1], :hh.shape[2], :]
                y_high = lh + hh

                if y_high != 0:
                    if self.activation != BNReLU and self.activation != BatchNorm and self.activation != tf.nn.batch_normalization:
                        tf.nn.bias_add(y_high, self.bias_h)

                    if self.activation is not None:
                        y_high = self.activation(y_high)
        return [y_low, y_high]

