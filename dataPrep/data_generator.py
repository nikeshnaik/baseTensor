import tensorflow as tf
import os
from utils.Utils import open_config

class Datagenerator():

    def __init__(self,config):
        path,type,seed,no_epochs = open_config(config,['data_path','type','seed','no_epochs'])
        tf.compat.v1.set_random_seed(seed)
        data = self.datagen(path,no_epochs)
        for key,val in data.items():
            setattr(self,key,val)

    def read_data(self,path=''):

        col_names = [*range(1,785)]
        if path and os.path.exists(path):
            features = tf.data.experimental.CsvDataset(path,record_defaults=[tf.int32]*784,select_cols=col_names,compression_type='GZIP',header=True,field_delim=",")
            label = tf.data.experimental.CsvDataset(path,record_defaults=[tf.int32]*1,select_cols=[0],compression_type='GZIP',header=True,field_delim=",")
        else:
            raise FileNotFoundError

        return (features,label)

    def preprocessing(self,features,label,no_epochs):

        def normalize(*args):
            return tf.stack(args)/255

        features= features.map(normalize)
        dataset = tf.data.Dataset.zip((features,label))
        dataset = dataset.shuffle(2000)
        dataset = dataset.repeat(no_epochs)
        dataset = dataset.batch(64)
        dataset = dataset.prefetch(30)
        return dataset

    def datagen(self,path,no_epochs):

        features,label = self.read_data(path)
        dataset = self.preprocessing(features,label,no_epochs)
        iterator = dataset.make_one_shot_iterator().get_next()
        data_input = {'get_next':iterator}
        return data_input

if __name__ == '__main__':
    import time
    start = time.time()
    config = './config/config.json'
    print("Start Timer")
    d = Datagenerator(config)
    with tf.Session() as sess:
        print(type(sess.run(d.get_next)))
        print(len(sess.run(d.get_next)))
        # print(np.shape(sess.run(d.get_nextbatch)[0])[0])
        # print("One Example -->{}".format(a))
    print("Total Time--> {}".format(time.time()-start))
