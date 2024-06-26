import os
import logging
import numpy as np
import live_player
import testing_server
import load
import math

IF_SUBOPTI = 0
IF_MULTIPLE = 0
if IF_MULTIPLE == 1:
	IF_SUBOPTI = 1
NEW = 1
# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

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
SERVER_START_UP_TH = 4000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
BUFFER_LENGTHS = [2000.0, 3000.0, 4000.0]
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound

DEFAULT_ACTION = 0			# lowest bitrate
TYPE = 2								# <============== Modified
TYPES = [2, 3, 4]
LH_STEP = 1								# <============== Modified
LH_STEPS = [1, 2, 3, 5, 10, -1]
if TYPE == 1:
	ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
	REBUF_PENALTY = 6.0		# for second
	SMOOTH_PENALTY = 1.0
	LONG_DELAY_PENALTY = 5.0 * CHUNK_SEG_RATIO 
	LONG_DELAY_PENALTY_BASE = 1.2	# for second
	MISSING_PENALTY = 6.0	* CHUNK_SEG_RATIO 		# not included

elif TYPE == 2:			# Sensitive to latency
	ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
	REBUF_PENALTY = 6.0		# for second
	SMOOTH_PENALTY = 1.0
	# LONG_DELAY_PENALTY_BASE = 1.2	# for second
	MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 		# not included
	LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
	CONST = 6.0
	X_RATIO = 1.0

elif TYPE == 3:			# Sensitive to bitrate
	ACTION_REWARD = 2.0 * CHUNK_SEG_RATIO	
	REBUF_PENALTY = 6.0		# for second
	SMOOTH_PENALTY = 1.0
	# LONG_DELAY_PENALTY_BASE = 1.2	# for second
	MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 			# not included
	LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
	CONST = 6.0
	X_RATIO = 1.0

elif TYPE == 4:			# Sensitive to bitrate
	ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
	REBUF_PENALTY = 6.0		# for second
	SMOOTH_PENALTY = 1.5
	# LONG_DELAY_PENALTY_BASE = 1.2	# for second
	MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 			# not included
	LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
	CONST = 6.0
	X_RATIO = 1.0

# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1

TEST_DURATION = 100				# Number of testing <===================== Change length here
TIMING_MAX = TEST_DURATION*SEG_DURATION/MS_IN_S + 10.0
TIMING_BIN = 0.05
BUFFER_MAX = USER_LATENCY_TOL/MS_IN_S
BUFFER_BIN = 0.05

RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others

# bitrate number is 6, no bin

# DATA_DIR = '../bw_traces/'
# TRACE_NAME = '70ms_loss0.5_m5.txt'
if NEW:
	DATA_DIR = '../new_traces/test_sim_traces/'
	TRACE_NAME = 'norway_bus_20'
else:
	DATA_DIR = '../bw_traces_test/cooked_test_traces/'
	# TRACE_NAME = '85+-29ms_loss0.5_0_2.txt'
	TRACE_NAME = '70+-24ms_loss1_2_1.txt'

# OPT_RESULT = './results/total_reward_and_seq_timing.txt'
if IF_SUBOPTI:
	# OPT_RESULT = './subopt_results/total_reward_and_seq_latency_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '_step_' + str(LH_STEP) + '.txt'
	SUMMARY_DIR = './sub_test_results'
	LOG_FILE = './sub_test_results/subupper'
	# UPPER_DIR = './results'
else:
	if NEW:
		# OPT_RESULT = './results/total_reward_and_seq_latency_'+ str(SERVER_START_UP_TH/MS_IN_S)+'.txt'
		OPT_RESULT = './results/paper_norway_bus_20_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_2.txt'
	else:
		OPT_RESULT = './results/paper70+-24ms_loss1_2_1.txt_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_2.txt'
	SUMMARY_DIR = './test_results'
	LOG_FILE = './test_results/upper'
# TRAIN_TRACES = './traces/bandwidth/'

def ReLU(x):
	return x * (x > 0)

def lat_penalty(x):
	return 1.0/(1+math.exp(CONST-X_RATIO*x)) - 1.0/(1+math.exp(CONST))

def record_tp(tp_trace, starting_time_idx, duration):
	tp_record = []
	offset = 0
	num_record = int(np.ceil(duration/SEG_DURATION))
	for i in range(num_record):
		if starting_time_idx + i + offset >= len(tp_trace):
			offset = -len(tp_trace)
		tp_record.append(tp_trace[starting_time_idx + i + offset])
	return tp_record

# def curves_show(sub_r):
# 	assert len(sub_r) == len(TYPES) * len(BUFFER_LENGTHS)
# 	for i in range(len(TYPES)):
# 		for j in range(len(BUFFER_LENGTHS)):
# 			curr_curve = sub_r[i*len(TYPES)+j]

def m_main():
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	all_rewards = []
	plot_rewards = []
	total_path = './sub_test_results/plot_all_data.txt'
	total_rate_opt_path_t2_f3 = './sub_test_results/t2_f3.txt'
	end_log_file = open(total_path, 'w')
	t2_f3_file = open(total_rate_opt_path_t2_f3, 'w')
	for t in TYPES:
		if t == 1:
			ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
			REBUF_PENALTY = 6.0		# for second
			SMOOTH_PENALTY = 1.0
			LONG_DELAY_PENALTY = 5.0 * CHUNK_SEG_RATIO 
			LONG_DELAY_PENALTY_BASE = 1.2	# for second
			MISSING_PENALTY = 6.0	* CHUNK_SEG_RATIO 		# not included

		elif t == 2:			# Sensitive to latency
			ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
			REBUF_PENALTY = 6.0		# for second
			SMOOTH_PENALTY = 1.0
			# LONG_DELAY_PENALTY_BASE = 1.2	# for second
			MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 		# not included
			LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
			CONST = 6.0
			X_RATIO = 1.0

		elif t == 3:			# Sensitive to bitrate
			ACTION_REWARD = 2.0 * CHUNK_SEG_RATIO	
			REBUF_PENALTY = 6.0		# for second
			SMOOTH_PENALTY = 1.0
			# LONG_DELAY_PENALTY_BASE = 1.2	# for second
			MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 			# not included
			LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
			CONST = 6.0
			X_RATIO = 1.0

		elif t == 4:			# Sensitive to bitrate
			ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
			REBUF_PENALTY = 6.0		# for second
			SMOOTH_PENALTY = 1.5
			# LONG_DELAY_PENALTY_BASE = 1.2	# for second
			MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 			# not included
			LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
			CONST = 6.0
			X_RATIO = 1.0
		t_rewards = []
		t_rewards.append('type' + str(t))
		for b_l in BUFFER_LENGTHS:
			bl_rewards = []
			curve_rewards = [t, b_l/MS_IN_S]
			bl_rewards.append('buffer' + str(b_l/MS_IN_S))

			for lh_l in LH_STEPS:
				if lh_l == -1:
					subopt_path = './results/total_reward_and_seq_latency_' + str(b_l/MS_IN_S) + '_type_' + str(t) + '.txt'
				else:
					subopt_path = './subopt_results/total_reward_and_seq_latency_' + str(b_l/MS_IN_S) + '_type_' + str(t) + '_step_' + str(lh_l) + '.txt'
				print(subopt_path)
				np.random.seed(RANDOM_SEED)

				cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)
				player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
											seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
											start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_FREEZING_TOL + b_l,
											randomSeed=RANDOM_SEED)
				server = testing_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
											start_up_th=b_l, randomSeed=RANDOM_SEED)


				initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
				print(initial_delay)
				log_path = LOG_FILE + '_buff' + str(b_l) + '_type_' + str(t) + '_step_' + str(lh_l) + '.txt'
				log_file = open(log_path, 'w')

				upper_actions = []
				with open(subopt_path, 'r') as f:
					for line in f:
						upper_actions.append(int(float(line.strip('\n'))))
				upper_actions.pop(0)
				assert len(upper_actions) == TEST_DURATION
				bit_rate = upper_actions[0]	# Trick here, in order to make first action no penalty
				init = 1
				starting_time = server.get_time()	# Server starting time
				starting_time_idx = player.get_time_idx()
				buffer_length = 0.0
				r_batch = []
				new_r_batch = []
				last_bit_rate = -1
				action_reward = 0.0				# Total reward is for all chunks within on segment
				rate_action_reward = 0.0
				action_freezing = 0.0
				for i in range(len(upper_actions)):
					print("Current index: ", i)
					if init: 
						if CHUNK_IN_SEG == 5:
							ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
						else:
							ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
						
						server.set_ratio(ratio)
						server.init_encoding()
						init = 0

					# last_bit_rate = bit_rate
					bit_rate = upper_actions[i]		# Get optimal actions
					take_action = 1
					action_wait = 0.0
					while True:  # serve video forever
						download_chunk_info = server.get_next_delivery()
						# print "chunk info is " + str(download_chunk_info)
						download_seg_idx = download_chunk_info[0]
						download_chunk_idx = download_chunk_info[1]
						download_chunk_end_idx = download_chunk_info[2]
						download_chunk_size = download_chunk_info[3][bit_rate]		# Might be several chunks
						chunk_number = download_chunk_end_idx - download_chunk_idx + 1
						if not download_seg_idx == i:
							i = download_seg_idx
						assert chunk_number == 1
						server_wait_time = 0.0
						sync = 0
						missing_count = 0
						real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																				download_seg_idx, download_chunk_idx, take_action, chunk_number)
						take_action = 0
						action_freezing += freezing
						buffer_length = player.get_buffer_length()

						server_time = server.update(download_duration)
						if not time_out:
							# server.chunks.pop(0)
							server.clean_next_delivery()
							sync = player.check_resync(server_time)
						else:
							assert player.get_state() == 0
							assert np.round(player.get_buffer_length(), 3) == 0.0
							# Pay attention here, how time out influence next reward, the smoothness
							# Bit_rate will recalculated later, this is for reward calculation
							bit_rate = 0
							sync = 1
						# Disable sync for current situation
						if sync:
							# break	# No resync here
							# To sync player, enter start up phase, buffer becomes zero
							sync_time, missing_count = server.sync_encoding_buffer()
							player.sync_playing(sync_time)
							buffer_length = player.get_buffer_length()

						latency = server.get_time() - player.get_playing_time()
						# print "latency is: ", latency/MS_IN_S
						player_state = player.get_state()

						log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
						if last_bit_rate == -1:
							log_last_bit_rate = log_bit_rate
						else:
							log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
						if TYPE == 1:
							reward = ACTION_REWARD * log_bit_rate * chunk_number \
								- REBUF_PENALTY * freezing / MS_IN_S \
								- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
								- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number \
								- MISSING_PENALTY * missing_count
						else:
							reward = ACTION_REWARD * log_bit_rate * chunk_number \
								- REBUF_PENALTY * freezing / MS_IN_S \
								- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
								- LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number \
								- MISSING_PENALTY * missing_count
								# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
						# print(reward)
						action_reward += reward
						# print reward
						# print LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number
						rate_action_reward += (reward + LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number )
						last_bit_rate = bit_rate	# Do no move this term. This is for chunk continuous calcualtion

						# chech whether need to wait, using number of available segs
						if server.check_chunks_empty():
							# print "Enter wait"
							server_wait_time = server.wait()
							action_wait += server_wait_time
							# print " Has to wait: ", server_wait_time
							assert server_wait_time > 0.0
							assert server_wait_time < CHUNK_DURATION
							# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
							player.wait(server_wait_time)
							# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
							buffer_length = player.get_buffer_length()

						# print "After wait, ", server.get_time() - (seg_idx + 1) * SEG_DURATION
						if CHUNK_IN_SEG == 5:
							ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
						else:
							ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
						server.set_ratio(ratio)
						server.generate_next_delivery()
						next_chunk_idx = server.get_next_delivery()[1]
						if next_chunk_idx == 0 or sync:
							# Record state and get reward
							# if sync:
							# 	# Process sync
							# 	assert 0 == 1
							# 	pass
							# else:
							take_action = 1
							# last_bit_rate = bit_rate
							# print(action_reward)
							r_batch.append(action_reward)
							new_r_batch.append(rate_action_reward)
							log_file.write(	str(server.get_time()) + '\t' +
										    str(BITRATE[bit_rate]) + '\t' +
											str(buffer_length) + '\t' +
											# str(action_freezing) + '\t' +
											str(freezing) + '\t' +
											str(time_out) + '\t' +
											# str(action_wait) + '\t' +
											str(server_wait_time) + '\t' +
										    str(sync) + '\t' +
										    str(latency) + '\t' +
										    str(player.get_state()) + '\t' +
										    str(int(bit_rate/len(BITRATE))) + '\t' +						    
											# str(action_reward) + '\n')
											str(reward) + '\n')
							log_file.flush()
							action_reward = 0.0
							action_freezing = 0.0
							rate_action_reward = 0.0
							break

						else:
							log_file.write(	str(server.get_time()) + '\t' +
										    str(BITRATE[bit_rate]) + '\t' +
											str(buffer_length) + '\t' +
											str(freezing) + '\t' +
											str(time_out) + '\t' +
											str(server_wait_time) + '\t' +
										    str(sync) + '\t' +
										    str(latency) + '\t' +
										    str(player.get_state()) + '\t' +
										    str(int(bit_rate/len(BITRATE))) + '\t' +						    
											str(reward) + '\n')
							log_file.flush()

				time_duration = server.get_time() - starting_time
				tp_record = record_tp(player.get_throughput_trace(), starting_time_idx, time_duration) 
				print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
				end_log_file.write('type: ' + str(t) + ' buffer: ' + str(b_l) + ' step: ' + str(lh_l) + ' rewards: ' + str(np.round(np.sum(r_batch), 3)))
				end_log_file.write('\n')
				if t == 2:
					t2_f3_file.write('type: ' + str(t) + ' buffer: ' + str(b_l) + ' step: ' + str(lh_l) \
						+ ' rewards: ' + str(np.round(np.sum(r_batch), 3)) + ' new_r: ' + str(np.round(np.sum(new_r_batch), 3))  )
					t2_f3_file.write('\n')
				log_file.write('\t'.join(str(tp) for tp in tp_record))
				log_file.write('\n' + str(starting_time))
				log_file.write('\n')
				log_file.close()

				bl_rewards.append(str(lh_l) +  ': ' + str(np.round(np.sum(r_batch), 3)))
				curve_rewards.append(np.round(np.sum(r_batch)))

			plot_rewards.append(curve_rewards)
			t_rewards.append(bl_rewards)
		all_rewards.append(t_rewards)
	print(all_rewards)
	end_log_file.close()
	# curves_show(plot_rewards)

def main():
	np.random.seed(RANDOM_SEED)

	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	if not NEW:
		cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)
	else:
		cooked_time, cooked_bw = load.new_load_single_trace(DATA_DIR + TRACE_NAME)

	player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)
	server = testing_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

	initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
	print(initial_delay)
	if IF_SUBOPTI:
		if NEW:
			log_path = LOG_FILE + '_bus_20_buff' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '_step_' + str(LH_STEP) + '.txt'
		else:
			log_path = LOG_FILE + '+-70ms_buff' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '_step_' + str(LH_STEP) + '.txt'
	else:
		if NEW:
			log_path = LOG_FILE + '_bus_20_buff' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '.txt'
			paper_log = LOG_FILE + '_bus_20_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '.txt'
		else:
			log_path = LOG_FILE + '+-70_buff' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '.txt'
			# paper_log = LOG_FILE + '_paper+-_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE)
			paper_log = LOG_FILE + '_paper+-70ms_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE)+ '.txt'
	log_file = open(log_path, 'w')
	all_testing_log = open(paper_log, 'w')
	upper_actions = []
	with open(OPT_RESULT, 'r') as f:
		for line in f:
			upper_actions.append(int(float(line.strip('\n'))))
	upper_actions.pop(0)
	# upper_actions = [0, 0, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4]
	bit_rate = upper_actions[0]	# Trick here, in order to make first action no penalty
	init = 1
	starting_time = server.get_time()	# Server starting time
	starting_time_idx = player.get_time_idx()
	buffer_length = 0.0
	r_batch = []
	f_batch = []
	a_batch = []
	c_batch = []
	l_batch = []
	last_bit_rate = -1
	action_reward = 0.0				# Total reward is for all chunks within on segment
	action_freezing = 0.0
	for i in range(len(upper_actions)):
		print("Current index: ", i)
		if init: 
			if CHUNK_IN_SEG == 5:
				ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
			else:
				ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
			
			server.set_ratio(ratio)
			server.init_encoding()
			init = 0

		# last_bit_rate = bit_rate
		bit_rate = upper_actions[i]		# Get optimal actions
		take_action = 1
		action_wait = 0.0
		while True:  # serve video forever
			download_chunk_info = server.get_next_delivery()
			# print "chunk info is " + str(download_chunk_info)
			download_seg_idx = download_chunk_info[0]
			download_chunk_idx = download_chunk_info[1]
			download_chunk_end_idx = download_chunk_info[2]
			download_chunk_size = download_chunk_info[3][bit_rate]		# Might be several chunks
			chunk_number = download_chunk_end_idx - download_chunk_idx + 1
			# assert download_seg_idx == i
			assert chunk_number == 1
			server_wait_time = 0.0
			sync = 0
			missing_count = 0
			real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																	download_seg_idx, download_chunk_idx, take_action, chunk_number)
			take_action = 0
			action_freezing += freezing
			buffer_length = player.get_buffer_length()
			server_time = server.update(download_duration)
			if not time_out:
				# server.chunks.pop(0)
				server.clean_next_delivery()
				sync = player.check_resync(server_time)
			else:
				assert player.get_state() == 0
				assert np.round(player.get_buffer_length(), 3) == 0.0
				# Pay attention here, how time out influence next reward, the smoothness
				# Bit_rate will recalculated later, this is for reward calculation
				bit_rate = 0
				sync = 1
			# Disable sync for current situation
			if sync:
				print("Sync happen")
				# break	# No resync here
				# To sync player, enter start up phase, buffer becomes zero
				sync_time, missing_count = server.sync_encoding_buffer()
				player.sync_playing(sync_time)
				buffer_length = player.get_buffer_length()

			latency = server.get_time() - player.get_playing_time()
			# print "latency is: ", latency/MS_IN_S
			player_state = player.get_state()

			log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
			if last_bit_rate == -1:
				log_last_bit_rate = log_bit_rate
			else:
				log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
			if TYPE == 1:
				reward = ACTION_REWARD * log_bit_rate * chunk_number \
					- REBUF_PENALTY * freezing / MS_IN_S \
					- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
					- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number \
					- MISSING_PENALTY * missing_count
			else:
				reward = ACTION_REWARD * log_bit_rate * chunk_number \
					- REBUF_PENALTY * freezing / MS_IN_S \
					- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
					- LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number \
					- MISSING_PENALTY * missing_count
					# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
			# print(reward)
			action_reward += reward
			last_bit_rate = bit_rate	# Do no move this term. This is for chunk continuous calcualtion

			# chech whether need to wait, using number of available segs
			if server.check_chunks_empty():
				# print "Enter wait"
				server_wait_time = server.wait()
				action_wait += server_wait_time
				# print " Has to wait: ", server_wait_time
				assert server_wait_time > 0.0
				assert server_wait_time < CHUNK_DURATION
				# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
				player.wait(server_wait_time)
				# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
				buffer_length = player.get_buffer_length()

			# print "After wait, ", server.get_time() - (seg_idx + 1) * SEG_DURATION
			if CHUNK_IN_SEG == 5:
				ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
			else:
				ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
			server.set_ratio(ratio)
			server.generate_next_delivery()
			next_chunk_idx = server.get_next_delivery()[1]
			if next_chunk_idx == 0 or sync:
				# Record state and get reward
				if sync:
					# Process sync
					print("Sync happen")
					break
				else:
					take_action = 1
					# last_bit_rate = bit_rate
					# print(action_reward)
					r_batch.append(action_reward)
					f_batch.append(action_freezing)
					a_batch.append(BITRATE[bit_rate])
					l_batch.append(latency)
					c_batch.append(np.abs(BITRATE[bit_rate] - BITRATE[last_bit_rate]))
					log_file.write(	str(server.get_time()) + '\t' +
								    str(BITRATE[bit_rate]) + '\t' +
									str(buffer_length) + '\t' +
									str(freezing) + '\t' +
									str(time_out) + '\t' +
									# str(action_wait) + '\t' +
									str(server_wait_time) + '\t' +
								    str(sync) + '\t' +
								    str(latency) + '\t' +
								    str(player.get_state()) + '\t' +
								    str(int(bit_rate/len(BITRATE))) + '\t' +						    
									# str(action_reward) + '\n')
									str(reward) + '\n')
					log_file.flush()
					print(action_freezing)
					action_reward = 0.0
					action_freezing = 0.0
					break
			else:
				log_file.write(	str(server.get_time()) + '\t' +
							    str(BITRATE[bit_rate]) + '\t' +
								str(buffer_length) + '\t' +
								str(freezing) + '\t' +
								str(time_out) + '\t' +
								str(server_wait_time) + '\t' +
							    str(sync) + '\t' +
							    str(latency) + '\t' +
							    str(player.get_state()) + '\t' +
							    str(int(bit_rate/len(BITRATE))) + '\t' +						    
								str(reward) + '\n')
				log_file.flush()

	# need to modify
	time_duration = server.get_time() - starting_time
	tp_record = record_tp(player.get_throughput_trace(), starting_time_idx, time_duration) 
	print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
	log_file.write('\t'.join(str(tp) for tp in tp_record))
	log_file.write('\n' + str(starting_time))
	log_file.write('\n')
	log_file.close()

	all_testing_log.write('norway_bus_20' + '\t')
	all_testing_log.write(str(np.sum(r_batch)) + '\t')
	all_testing_log.write(str(np.mean(a_batch)) + '\t')
	all_testing_log.write(str(np.sum(f_batch)) + '\t')
	all_testing_log.write(str(np.mean(c_batch)) + '\t')
	all_testing_log.write(str(np.mean(l_batch)) + '\t')
	all_testing_log.write('\n')
	all_testing_log.close()

if __name__ == '__main__':
	if not IF_MULTIPLE:
		main()
	else:
		m_main()


