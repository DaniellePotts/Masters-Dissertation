import time
from anyrl.rollouts import replay
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Conv2D
from keras.optimizers import SGD, Adam
from keras import initializers
from keras import regularizers
import keras.backend as K

import numpy as np

#load in and build machine learning model
def load_model(n_action, model_path):
	model = build_model(n_action)
	model.load_weights(model_path)

	return model

#run the model and get a prediction
def get_model_prediction(model, n_action, curr_obs):
   #set empty arrays for feeding into model - used for appeasing model shape 
    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, n_action))

    #parse the current observation
    temp_curr_obs = np.array(curr_obs)
    temp_curr_obs = temp_curr_obs.tolist()['pov'].reshape(1,temp_curr_obs.tolist()['pov'].shape[0], temp_curr_obs.tolist()['pov'].shape[1], temp_curr_obs.tolist()['pov'].shape[2])
    
    #run the model
    q, _, _ = model.predict([temp_curr_obs, temp_curr_obs,empty_by_one, empty_exp_action_by_one,empty_action_len_by_one])
    return np.argmax(q)

#build the ML model
def build_model(action_len, img_rows=64, img_cols=64, img_channels=3, dueling=True, clip_value=1.0,
                learning_rate=1e-4, nstep_reg=1.0, slmc_reg=1.0, l2_reg=10e-5):
  input_img = Input(shape=(img_rows, img_cols, img_channels), name='input_img', dtype='float')
  scale_img = Lambda(lambda x: x/255.)(input_img)
  layer_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                     activation='relu', input_shape=(img_rows, img_cols, img_channels),
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(scale_img)#(input_img)
  layer_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                    kernel_initializer=initializers.glorot_normal(seed=31),
                    kernel_regularizer=regularizers.l2(l2_reg),
                    bias_regularizer=regularizers.l2(l2_reg))(layer_1)
  layer_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                    kernel_initializer=initializers.glorot_normal(seed=31),
                    kernel_regularizer=regularizers.l2(l2_reg),
                    bias_regularizer=regularizers.l2(l2_reg))(layer_2)
  x = Flatten()(layer_3)
  x = Dense(256, activation='relu',
              kernel_initializer=initializers.glorot_normal(seed=31),
              kernel_regularizer=regularizers.l2(l2_reg),
              bias_regularizer=regularizers.l2(l2_reg))(x)
  
  if not dueling:
    cnn_output = Dense(action_len,
                           kernel_initializer=initializers.glorot_normal(seed=31),
                           kernel_regularizer=regularizers.l2(l2_reg),
                           bias_regularizer=regularizers.l2(l2_reg), name='cnn_output')(x)
  else:
    dueling_values = Dense(1,
                               kernel_initializer=initializers.glorot_normal(seed=31),
                               kernel_regularizer=regularizers.l2(l2_reg),
                               bias_regularizer=regularizers.l2(l2_reg), name='dueling_values')(x)
    dueling_actions = Dense(action_len,
                            kernel_initializer=initializers.glorot_normal(seed=31),
                            kernel_regularizer=regularizers.l2(l2_reg),
                            bias_regularizer=regularizers.l2(l2_reg), name='dq_actions')(x)
    
    def dueling_operator(duel_input):
        duel_v = duel_input[0]
        duel_a = duel_input[1]
        return duel_v + (duel_a - K.mean(duel_a, axis=1, keepdims=True))
    
    cnn_output = Lambda(dueling_operator, name='cnn_output')([dueling_values, dueling_actions])
    
  cnn_model = Model(input_img, cnn_output)

  input_img_dq = Input(shape=(img_rows, img_cols, img_channels), name='input_img_dq', dtype='float32')
  input_img_nstep = Input(shape=(img_rows, img_cols, img_channels), name='input_img_nstep', dtype='float32')
  dq_output = cnn_model(input_img_dq)
  nstep_output = cnn_model(input_img_nstep)

  input_is_expert = Input(shape=(1,), name='input_is_expert')
  input_expert_action = Input(shape=(2,), name='input_expert_action', dtype='int32')
  input_expert_margin = Input(shape=(action_len,), name='input_expert_margin')
  
  def slmc_operator(slmc_input):
    is_exp = slmc_input[0]
    sa_values = slmc_input[1]
    exp_act = K.cast(slmc_input[2], dtype='int32')
    exp_margin = slmc_input[3]

    exp_val = tf.gather_nd(sa_values, exp_act)

    max_margin = K.max(sa_values + exp_margin, axis=1)
    max_margin_2 = max_margin - exp_val
    max_margin_3 = K.reshape(max_margin_2,K.shape(is_exp))
    max_margin_4 = tf.multiply(is_exp,max_margin_3)
    return max_margin_4
  
  slmc_output = Lambda(slmc_operator, name='slmc_output')([input_is_expert, dq_output,
                                                             input_expert_action, input_expert_margin])
  
  model = Model(inputs=[input_img_dq, input_img_nstep, input_is_expert, input_expert_action, input_expert_margin],
                outputs=[dq_output, nstep_output, slmc_output])
  
  if clip_value is not None:
    adam = Adam(lr=learning_rate, clipvalue=clip_value)
  else:
    adam = Adam(lr=learning_rate)
  

  model.compile(optimizer=adam,
                loss=['mse','mse','mae'],
                loss_weights=[1., nstep_reg, slmc_reg])
  
  return model

