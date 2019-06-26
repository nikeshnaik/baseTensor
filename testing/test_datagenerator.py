import unittest
import tensorflow as tf
from dataPrep.data_generator import Datagenerator
import numpy as np

def test_init():
    obj = Datagenerator('./config/config.json')
    with tf.Session() as sess:
        assert 64 == np.shape(sess.run(obj.get_next)[0])[0]
        assert 2 == len(sess.run(obj.get_next))
