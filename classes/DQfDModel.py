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

class DQfDModel:
	def build_model(self, action_len, img_rows=64, img_cols=64, img_channels=3, dueling=False, clip_value=1.0,
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
			print('duelling')
		
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
	def inner_train_function(self, train_model, target_model, exp_buffer, replay_buffer,
                         states_batch, action_batch, reward_batch,
                         next_states_batch, done_batch, nstep_rew_batch, nstep_next_batch,
                         is_expert_input, expert_action_batch, expert_margin,
                         action_len, exp_minibatch, minibatch,
                         batch_size=32, gamma=0.99, nstep_gamma=0.99, exp_batch_size=8):
  
		empty_batch_by_one = np.zeros((batch_size, 1))
		empty_action_batch = np.zeros((batch_size, 2))
		empty_action_batch[:,0] = np.arange(batch_size)
		empty_batch_by_action_len = np.zeros((batch_size, action_len))
		ti_tuple = tuple([i for i in range(batch_size)])
		nstep_final_gamma = nstep_gamma ** 10

		q_values_next_target, nstep_q_values_next_target, _ = target_model.predict(
			[next_states_batch, nstep_next_batch,
			empty_batch_by_one, empty_action_batch,
			empty_batch_by_action_len]) #
		
		q_values_next_train, nstep_q_values_next_train, _ = train_model.predict(
			[next_states_batch, nstep_next_batch,
			empty_batch_by_one, empty_action_batch,
			empty_batch_by_action_len])
		
		action_max = np.argmax(q_values_next_train, axis=1)
		nstep_action_max = np.argmax(nstep_q_values_next_train, axis=1)

		dq_targets, nstep_targets, _ = train_model.predict([states_batch, states_batch, is_expert_input,
															expert_action_batch, expert_margin])
		
		dq_targets[ti_tuple, action_batch] = reward_batch + \
											(1 - done_batch) * gamma \
											* q_values_next_target[np.arange(batch_size), action_max]

		nstep_targets[ti_tuple, action_batch] = nstep_rew_batch + \
												(1 - done_batch) * nstep_final_gamma \
												* nstep_q_values_next_target[np.arange(batch_size), nstep_action_max]

		dq_pred, nstep_pred, slmc_pred = train_model.predict_on_batch([states_batch, states_batch,
																		is_expert_input, expert_action_batch, expert_margin])
		
		dq_loss = np.square(dq_pred[np.arange(batch_size),action_batch]-dq_targets[np.arange(batch_size),action_batch])
		nstep_loss = np.square(nstep_pred[np.arange(batch_size), action_batch] - nstep_targets[np.arange(batch_size), action_batch])

		loss = train_model.train_on_batch([states_batch, states_batch, is_expert_input, expert_action_batch, expert_margin],
											[dq_targets, nstep_targets, empty_batch_by_one])
		
		dq_loss_weighted = np.reshape(dq_loss, (batch_size, 1))/np.sum(dq_loss)*loss[1] * batch_size
		nstep_loss_weighted = np.reshape(nstep_loss, (batch_size, 1))/np.sum(nstep_loss)*loss[2]*batch_size

		sample_losses = dq_loss_weighted + nstep_loss_weighted + np.abs(slmc_pred)

		if replay_buffer is not None:
			exp_buffer.update_weights(exp_minibatch, sample_losses[:exp_batch_size])
			rep_buffer.update_weights(minibatch, sample_losses[-(batch_size-exp_batch_size):])
		else:
			exp_buffer.update_weights(exp_minibatch, sample_losses)


		return np.array(loss)

	def train_expert_network(train_model, target_model, replay_buffer, action_len, batch_size=32,
                         train_steps=10000, update_every=10000,gamma=0.99, nstep_gamma=0.99,exp_margin_constant=0.8):
		time_int = int(time.time())
		loss = np.zeros((4,))

		for current_step in range(train_steps):
			print(str(current_step) + "/" + str(train_steps))
			relay_mini_batch = replay_buffer.sample(batch_size)
			replay_zip_batch = []

			for i in relay_mini_batch:
			replay_zip_batch.append(i['sample'])#
			
			exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch, \
			exp_done_batch, exp_reward_batch, exp_nstep_next_batch = map(np.array, zip(*replay_zip_batch))

			
			is_expert_input = np.ones((batch_size, 1))

			input_exp_action = np.zeros((batch_size, 2))
			input_exp_action[:,0] = np.arange(batch_size)
			input_exp_action[:,1] = exp_action_batch
			
			exp_margin = np.ones((batch_size, action_len)) * exp_margin_constant
			exp_margin[np.arange(batch_size), exp_action_batch] = 0.
		
			loss += inner_train_function(train_model, target_model, replay_buffer, None, 
										exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch,
										exp_done_batch, exp_reward_batch, exp_nstep_next_batch, is_expert_input, 
										input_exp_action, exp_margin, action_len, relay_mini_batch, None, #
										batch_size, gamma, nstep_gamma)
			
		#save the model every n steps
			if current_step % update_every == 0 and current_step >= update_every:
			print("Saving expert training weights at step {}. Loss is {}".format(current_step, loss))
			
			zString = "expert_model_{}_{}.h5".format(time_int, current_step)
			train_model.save_weights(zString, overwrite=True)
			# updating fixed Q network weights
			target_model.load_weights(zString)
			
		print("Saving expert final weights. Loss is {}".format(loss))
		zString = "expert_model_{}_{}.h5".format(time_int, current_step)
		train_model.save_weights(zString, overwrite=True)

		return train_model, replay_buffer	