import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import live_player
import live_server
import static_a3c as a3c
import load

S_INFO = 8
S_LEN = 12
A_DIM = 6	
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 1

TRAIN_SEQ_LEN = 200
MODEL_SAVE_INTERVAL = 100

# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 12000.0]	# 5 actions

RANDOM_SEED = 13
RAND_RANGE = 1000
MS_IN_S = 1000.0
KB_IN_MB = 1000.0	# in ms
SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
# Initial buffer length on server side
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound

STARTING_EPOCH = 0
NN_MODEL = None
# STARTING_EPOCH = 70000
# NN_MODEL = './results/nn_model_s_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_ep_' + str(STARTING_EPOCH) + '.ckpt'
TERMINAL_EPOCH = 20000

DEFAULT_ACTION = 0			# lowest bitrate
ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
REBUF_PENALTY = 10.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0 * CHUNK_SEG_RATIO 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 2.0			# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1

LATENCY_MAX = BUFFER_MAX/MS_IN_S
LATENCY_BIN = 0.1
BUFFER_MAX = USER_LATENCY_TOL
BUFFER_BIN = 0.1

# bitrate number is 6, no bin

DATA_DIR = '../bw_traces/'
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TRACE_NAME = '../bw_traces/BKLYN_1.txt'

# TRAIN_TRACES = './traces/bandwidth/'


def ReLU(x):
	return x * (x > 0)

def main():
	# create result directory
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)
	# Initial server and player
	cooked_time, cooked_bw = load.load_single_trace(TRACE_NAME)

	player = live_player.Live_Player(time_traces=cooked_time, throughput_traces=cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=agent_id)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH)
		
	initial = 1
	action_num = DEFAULT_ACTION
	last_bit_rate = action_num
	bit_rate = action_num
	# last_bit_rate = DEFAULT_ACTION%len(BITRATE)
	# bit_rate = DEFAULT_ACTION%len(BITRATE)
	# playing_speed = NORMAL_PLAYING

	action_vec = np.zeros(A_DIM)
	action_vec[action_num] = 1


	r_table_pre = [[[[float("-inf")] for _ in range(A_DIM)] for _ in range(LATENCY_MAX/LATENCY_BIN)] for _ in range(BUFFER_MAX/BUFFER_BIN)]
	r_table_curr = [[[[float("-inf")] for _ in range(A_DIM)] for _ in range(LATENCY_MAX/LATENCY_BIN)] for _ in range(BUFFER_MAX/BUFFER_BIN)]

	action_reward = 0.0		# Total reward is for all chunks within on segment
	take_action = 1
	latency = 0.0
	missing_count = 0

	while True:
		# get download chunk info
		# Have to modify here, as we can get several chunks at the same time
		# And the recording are jointly calculated. 
		# assert len(server.chunks) >= 1

		# Here, should get server next_delivery. Might be several chunks or a whole segment
		# download_chunk_info = server.chunks[0]		
		# download_chunk_size = download_chunk_info[2]
		# download_chunk_idx = download_chunk_info[1]
		# download_seg_idx = download_chunk_info[0]
		download_chunk_info = server.get_next_delivery()
		download_seg_idx = download_chunk_info[0]
		download_chunk_idx = download_chunk_info[1]
		download_chunk_end_idx = download_chunk_info[2]
		download_chunk_size = download_chunk_info[3]
		chunk_number = download_chunk_end_idx - download_chunk_idx + 1
		server_wait_time = 0.0
		sync = 0
		missing_count = 0
		real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(bit_rate, download_chunk_size, 
																	download_seg_idx, download_chunk_idx, take_action, chunk_number)
		take_action = 0
		past_time = download_duration
		buffer_length = player.buffer
		# print(player.playing_time)
		# print(past_time, len(server.chunks), server.next_delivery)
		server_time = server.update(past_time)
		if not time_out:
			# server.chunks.pop(0)
			server.clean_next_delivery()
			sync = player.check_resync(server_time)
		else:
			assert player.state == 0
			assert np.round(player.buffer, 3) == 0.0
			# Pay attention here, how time out influence next reward, the smoothness
			# Bit_rate will recalculated later, this is for reward calculation
			bit_rate = 0
			sync = 1
		if sync:
			# To sync player, enter start up phase, buffer becomes zero
			sync_time, missing_count = server.sync_encoding_buffer()
			player.sync_playing(sync_time)
			buffer_length = player.buffer

		latency = server.time - player.playing_time
		player_state = player.state

		log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
		log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
		last_bit_rate = bit_rate
		# print(log_bit_rate, log_last_bit_rate)
		reward = ACTION_REWARD * log_bit_rate * chunk_number \
				- REBUF_PENALTY * freezing / MS_IN_S \
				- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
				- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number \
				- MISSING_PENALTY * missing_count
				# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
		# print(reward)
		action_reward += reward

		# chech whether need to wait, using number of available segs
		if len(server.chunks) == 0:
			server_wait_time = server.wait()
			assert server_wait_time > 0.0
			assert server_wait_time < CHUNK_DURATION
			player.wait(server_wait_time)
			buffer_length = player.buffer
		server.generate_next_delivery()
		# print(bit_rate, download_duration, server_wait_time, player.buffer, \
		# 	server.time, player.playing_time, freezing, reward, action_reward)

		# Establish state for next iteration
		state = np.roll(state, -1, axis=1)
		state[0, -1] = real_chunk_size / KB_IN_MB 		# chunk size
		state[1, -1] = download_duration / MS_IN_S		# downloading time
		state[2, -1] = buffer_length / MS_IN_S			# buffer length
		state[3, -1] = chunk_number						# number of chunk sent
		state[4, -1] = BITRATE[bit_rate] / BITRATE[0]	# video bitrate
		# state[4, -1] = latency / MS_IN_S				# accu latency from start up
		state[5, -1] = sync 							# whether there is resync
		# state[6, -1] = player_state						# state of player
		state[6, -1] = server_wait_time / MS_IN_S		# time of waiting for server
		state[7, -1] = freezing / MS_IN_S				# current freezing time
		# generate next set of seg size
		# if add this, this will return to environment
		# next_chunk_size_info = server.chunks[0][2]	# not useful
		# state[7, :A_DIM] = next_chunk_size_info		# not useful
		# print(state)

		# Get next chunk/chunks information
		# Should not directly get the next one, but might be several chunks togethers
		# next_chunk_idx = server.chunks[0][1]
		next_chunk_idx = server.get_next_delivery()[1]
		if next_chunk_idx == 0 or sync:
			# print(action_reward)
			take_action = 1
			r_batch.append(action_reward)
			action_reward = 0.0
			# If sync, might go to medium of segment, and there is no estimated chunk size
			'''
			next_seg_size_info = []
			if sync and not next_chunk_idx == 0:
				next_seg_size_info = [2 * np.sum(x) / KB_IN_MB for x in server.chunks[0][2]] 
			else:
				next_seg_size_info = [x/KB_IN_MB for x in server.chunks[0][3]]

			state[8, :A_DIM] = next_seg_size_info
			'''
			action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
			action_cumsum = np.cumsum(action_prob)
			# print(action_prob)
			# Selection action
			action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

			bit_rate = action_num
			# if action_num >= len(BITRATE):
			# 	playing_speed = FAST_PLAYING
			# else:
			# 	playing_speed = NORMAL_PLAYING
			entropy_record.append(a3c.compute_entropy(action_prob[0]))
			log_file.write(str(buffer_length) + '\t' +
							str(freezing) + '\t' +
							str(time_out) + '\t' +
							# str(buffer_length) + '\t' +
							str(server_wait_time) + '\t' +
						    str(action_prob) + '\t' +
							str(reward) + '\n')
			log_file.flush()
		else:
			log_file.write(str(buffer_length) + '\t' +
							str(freezing) + '\t' +
							str(time_out) + '\t' +
							# str(buffer_length) + '\t' +
							str(server_wait_time) + '\t' +
							str(reward) + '\n')
			log_file.flush()

		if len(r_batch) >= TRAIN_SEQ_LEN :
			# print(r_batch)
			if len(s_batch) >= 1:
				if initial:
					exp_queue.put([s_batch[1:],  # ignore the first chuck
									a_batch[1:],  # since we don't have the
									r_batch[1:],  # control over it
									# terminal,
									{'entropy': entropy_record}])
					initial = 0
				else:
					exp_queue.put([s_batch[:],  # ignore the first chuck
									a_batch[:],  # since we don't have the
									r_batch[:],  # control over it
									# terminal,
									{'entropy': entropy_record}])

				actor_net_params, critic_net_params = net_params_queue.get()
				actor.set_network_params(actor_net_params)
				critic.set_network_params(critic_net_params)

				del s_batch[:]
				del a_batch[:]
				del r_batch[:]
				del entropy_record[:]
				take_action = 1
				log_file.write('\n')  # so that in the log we know where video ends

			else:
				print("length of s batch is too short: ", len(s_batch))
				
		# This is infinit seq
		if next_chunk_idx == 0 or sync:			
			s_batch.append(state)
			state = np.array(s_batch[-1], copy=True)
			action_vec = np.zeros(A_DIM)
			action_vec[action_num] = 1
			a_batch.append(action_vec)


if __name__ == '__main__':
	main()
