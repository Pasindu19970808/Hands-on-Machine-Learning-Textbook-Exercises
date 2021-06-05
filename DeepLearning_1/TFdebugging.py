import tensorflow as tf
import numpy as np
from tensorflow import keras

class customPrecision(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def update_state(self,y_true, y_pred,sample_weight = None):
        try: 
            for _class in self.unique_classes:
                self.total_prediction[_class] = self.total_prediction[_class] + tf.where(ypred == _class).shape[0]
                self.total_truepositives[_class] = self.total_truepositives[_class] + len([i for i in tf.where(ypred == _class) if i in tf.where(ytrue == _class)])
            return [self.total_prediction,self.total_truepositives]
        except:
            self.unique,_,_ = tf.unique_with_counts(ytrue)
            #obtains the unique classes to a list that we can loop through
            self.unique_classes = self.unique.numpy()
            self.total_truepositives = {_class:0 for _class in self.unique_classes}
            self.total_prediction = {_class:0 for _class in self.unique_classes}
            for _class in self.unique_classes:
                self.total_prediction[_class] = self.total_prediction[_class] + tf.where(ypred == _class).shape[0]
                self.total_truepositives[_class] = self.total_truepositives[_class] + len([i for i in tf.where(ypred == _class) if i in tf.where(ytrue == _class)])
            return [self.total_prediction,self.total_truepositives]
    def result(self):
        return self.total/self.count
    def get_config(self):
        base_config = self.get_config()
        return {**base_config, "threshold":self.threshold}

ytrue = tf.Variable([1,0,1,1,0,1,1,0,1])
ypred = tf.Variable([1,1,0,0,1,1,1,0,0])
prec_test = customPrecision()
res = prec_test.update_state(ytrue,ypred)