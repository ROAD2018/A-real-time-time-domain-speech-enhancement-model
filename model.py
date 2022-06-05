#from msilib.schema import Shortcut
import os, fnmatch
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D,GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
from random import shuffle, seed
import numpy as np
import tf2onnx


class RTrnn():
    def __init__(self):
        self.model=[]
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 8
        self.blockLen = 512
        self.block_shift = 128
        self.numUnit = 128
        self.numlayers = 3
        self.featuresize = 257
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)

    def rnnet(self,nninputs,stateful):
        r=tf.keras.layers.Convolution1D(self.featuresize,1)(nninputs)
        shortcut0=r
        r=tf.keras.layers.Dense(self.numUnit)(r)
        shortcut1=r

        r=tf.keras.layers.GRU(self.numUnit,return_sequences=True,stateful=stateful)(r)
        r=tf.keras.layers.Add()([shortcut1,r])
        shortcut2=r
        r=tf.keras.layers.GRU(self.numUnit,return_sequences=True,stateful=stateful)(r)
        r=tf.keras.layers.Add()([shortcut2,r])
        r=tf.keras.layers.LayerNormalization()(r)
        r=tf.keras.layers.GRU(self.numUnit,return_sequences=True,stateful=stateful)(r)
        r=tf.keras.layers.Dense(self.featuresize,activation='sigmoid')(r)
        r=tf.keras.layers.Multiply()([shortcut0,r])
        nnoutput=tf.keras.layers.Convolution1D(self.blockLen,1)(r)
        return nnoutput
    
    def staternnet(self,nninputs,in_states):
        states_h=[]
        #states_c=[]
        r=tf.keras.layers.Convolution1D(self.featuresize,1)(nninputs)
        shortcut0=r
        r=tf.keras.layers.Dense(self.numUnit)(r)
        shortcut=r
        for idx in range(self.numlayers):
            #in_state=[in_states[:,idx,:, 0], in_states[:,idx,:, 1]]
            in_state=in_states[:,idx,:,0]
            r,h_state=GRU(self.numUnit,return_sequences=True,return_state=True,unroll=True)(r,initial_state=in_state)
            states_h.append(h_state)
            #states_c.append(c_state)
            if idx==0:
                r=tf.keras.layers.Add()([shortcut,r])
                shortcut=r
            if idx==1:
                r=tf.keras.layers.Add()([shortcut,r])
                r=tf.keras.layers.LayerNormalization()(r)
        out_states_h = tf.reshape(tf.stack(states_h), 
                                  [1,self.numlayers,self.numUnit])
        out_states=tf.stack([out_states_h],axis=-1)
        r=tf.keras.layers.Dense(self.featuresize,activation='sigmoid')(r)
        r=tf.keras.layers.Multiply()([shortcut0,r])
        nnoutput=tf.keras.layers.Convolution1D(self.blockLen,1)(r)
        return nnoutput,out_states
    
   # def staternn(self,nninputs,in_states):


    def overlapAddLayer(self, x):
        return tf.signal.overlap_and_add(x, self.block_shift)

    def framegenerate(self,x):
        return tf.signal.frame(x,self.blockLen,self.block_shift)
    
    def build_model(self):
        time_dat = Input(batch_shape=(None, None))
        frames=Lambda(self.framegenerate)(time_dat)
        nnoutput=self.rnnet(frames,False)
        estimated_sig = Lambda(self.overlapAddLayer)(nnoutput)
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        print(self.model.summary())

    
    def build_state_model(self):
        time_dat = Input(batch_shape=(1, self.blockLen))
        frame = tf.expand_dims(time_dat, axis=1)
        nnoutput=self.rnnet(frame,True)
        estimated_sig = nnoutput
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        print(self.model.summary())
    
    def create_h5_model(self,weight_file,target_name):
        self.build_state_model()
        self.model.load_weights(weight_file)
        self.model.save(target_name+'.h5')
        print('save h5 model')


    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):
        self.build_state_model()
        self.model.load_weights(weights_file)
        timeinput=Input(batch_shape=(1,1,(self.blockLen)))
        statesin=Input(batch_shape=(1,self.numlayers,self.numUnit,1))
        nnoutput,statesout=self.staternnet(timeinput,statesin)
        rnnmodel=Model(inputs=[timeinput,statesin],outputs=[nnoutput,statesout])
        weights=self.model.get_weights()
        rnnmodel.set_weights(weights)
        converter=tf.lite.TFLiteConverter.from_keras_model(rnnmodel)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '.tflite', 'wb') as f:
            f.write(tflite_model)
        print('tflite convert')

    def create_onnx_model(self,weights_file,target_name):
        self.build_state_model()
        self.model.load_weights(weights_file)
        timeinput=Input(batch_shape=(1,1,self.blockLen))
        statesin=Input(batch_shape=(1,self.numlayers,self.numUnit,1))
        nnoutput,statesout=self.staternnet(timeinput,statesin)
        rnnmodel=Model(inputs=[timeinput,statesin],outputs=[nnoutput,statesout])
        weights=self.model.get_weights()
        rnnmodel.set_weights(weights)
        tempfile=target_name+'.onnx'
        tf2onnx.convert.from_keras(rnnmodel,output_path=tempfile)
        print('onnx convert')

    
    def compile_model(self):
        optimizerAdam = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=3.0)
        self.model.compile(loss='mae', optimizer=optimizerAdam)
   


