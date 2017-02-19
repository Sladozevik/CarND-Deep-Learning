# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:45:55 2017

@author: aslado
"""

# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)
cross_entropy = -tf.reduce_sum(tf.mul(one_hot, tf.log(softmax)))

# TODO: Print cross entropy from session
with tf.Session() as session:
    print(session.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))