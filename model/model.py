import tensorflow as tf
import os
# tf.enable_eager_execution()

class Datagenerator():
    def __init__(self,path):
        tf.compat.v1.set_random_seed(123)
        self.data = data_main(path)
    
    def read_data(path=''):
        col_names = [each for each in range(1,785)]
        if path and os.path.exists(path):
            features = tf.data.experimental.CsvDataset(path,record_defaults=[tf.int32]*784,select_cols=col_names,compression_type='GZIP',header=True,field_delim=",")
            label = tf.data.experimental.CsvDataset(path,record_defaults=[tf.int32]*1,select_cols=[0],compression_type='GZIP',header=True,field_delim=",")
    #
        else:
            raise FileNotFoundError

        return (features,label)

    def preprocessing(features,label):

    # TODO: Make it Faster as it takes time of 2 secs
    def normalize(*args):
        return tuple(each/255 for each in args)

    features.shuffle(20000).repeat(3).batch(64).prefetch(30)
    features= features.map(normalize)
    label.shuffle(20000).repeat(3).batch(64).prefetch(30)
    return (features,label)

    def data_main(self,path):
        features,label = read_data(path)
        features,label = preprocessing(features,label)

        fiterator = features.make_one_shot_iterator()
        literator = label.make_one_shot_iterator()

        fnext = fiterator.get_next()
        lnext = literator.get_next()

        data_input = {'features':features,'labels':label,'fiterator':fnext,'literator':lnext}
        # data_input = (features,label),(fiterator,literator)
        return data_input

if __name__ == '__main__':
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer)
        features,label = read_data(path='../data/mnist_test_mod.csv')
        features,label = preprocessing(features,label)
        fiterator = features.make_one_shot_iterator()
        literator = label.make_one_shot_iterator()
        fnext = fiterator.get_next()
        lnext = literator.get_next()
        featureone = sess.run(fnext)
        labelone = sess.run(lnext)
        print("Label:-->{}\n\nFeatures--->{}".format(labelone,featureone))
        featureone = sess.run(fnext)
        labelone = sess.run(lnext)
        print("Label:-->{}\n\nFeatures--->{}".format(labelone,sum(featureone)))
        featureone = sess.run(fnext)
        labelone = sess.run(lnext)
        print("Label:-->{}\n\nFeatures--->{}".format(labelone,sum(featureone)))
