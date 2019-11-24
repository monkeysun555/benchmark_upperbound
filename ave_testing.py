import os
import logging
import numpy as np
import live_player
import testing_server
import load
import math
import filenames as fns
import matplotlib as mpl 
import matplotlib.pyplot as plt 

BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
MAKEUP = 0.0
IF_SIMULATING = 0
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
BUFFER_LENGTHS = [2000.0, 3000.0, 4000.0]
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL	

DEFAULT_ACTION = 0			# lowest bitrate
TYPE = 2								# <============== Modified
TYPES = [2, 3, 4]
LH_STEP = 1								# <============== Modified
LH_STEPS = [1,2,3,5,10,-1]
NUM_TRACES = 8

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

TEST_DURATION = 100				# Number of testing <===================== Change length here
TIMING_MAX = TEST_DURATION*SEG_DURATION/MS_IN_S + 10.0
TIMING_BIN = 0.05
BUFFER_MAX = USER_LATENCY_TOL/MS_IN_S
BUFFER_BIN = 0.05

RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others

DATA_DIR = '../bw_traces/'

SUMMARY_DIR = './new_ave_total_rewards/'
LOG_FILE = SUMMARY_DIR + 'ave_upper'

SIMULATION_RESULTS = './new_ave_total_rewards/plot_all_data.txt'

def box_plt(buffer_curves):
	# Transform data
	# print buffer_curves
	buffer_boxes = []
	for i in range(len(buffer_curves)):
		# For each buffer curve
		boxes = []
		for j in range(len(LH_STEPS)):
			boxes.append([curve[j] for curve in buffer_curves[i]])
		buffer_boxes.append(boxes)


	p = plt.figure(figsize=(7,5.5))

	for box in buffer_boxes:
		plt.boxplot(box)

	plt.axis([0, 6 , 0, 1])

	p.show()
	raw_input()




def collect_results():
	matlab_plot_ratios = SUMMARY_DIR + 'matlab_plot.txt'
	matlab_log = open(matlab_plot_ratios, 'wb')
	results = []
	with open(SIMULATION_RESULTS, 'rb') as f:
		for line in f:
			parse = line.strip('\n')

			results.append([parse.split()[1], parse.split()[3], parse.split()[5], parse.split()[7], parse.split()[-1]])
	# print results
	final_curves = []
	for i in range(len(TYPES)):
		boxplot_curves = []
		for j in range(len(BUFFER_LENGTHS)):
			final_curve = []
			final_curve.append(str(TYPES[i]))
			final_curve.append(str(BUFFER_LENGTHS[j]))
			matlab_log.write(str(TYPES[i]) + '\t' )
			matlab_log.write(str(BUFFER_LENGTHS[j]) + '\t')
			file_curves = []
			for k in range(NUM_TRACES):
				file_curve = []
				optimal_reward_idx = (NUM_TRACES * (len(LH_STEPS) - 1) + k) +  j * len(LH_STEPS) * NUM_TRACES + i * len(BUFFER_LENGTHS) * len(LH_STEPS) * NUM_TRACES
				optimal_reward = float(results[optimal_reward_idx][-1])
				# print optimal_reward
				for l in range(len(LH_STEPS)):
					curr_idx = k + l * NUM_TRACES + j * len(LH_STEPS) * NUM_TRACES + i * len(BUFFER_LENGTHS) * len(LH_STEPS) * NUM_TRACES
					curr_reward = float(results[curr_idx][-1])
					if i == 0 and j == 2 :
						ratio = (curr_reward+MAKEUP)/(optimal_reward+MAKEUP)
					else:
						ratio = curr_reward/optimal_reward
					# print ratio
					file_curve.append(ratio)
				file_curves.append(file_curve)
			assert len(file_curves) == NUM_TRACES
			boxplot_curves.append(file_curves)
			ave_ratio = []
			for h in range(len(LH_STEPS)):
				ave_ratio.append(np.mean([curve[h] for curve in file_curves]))
				matlab_log.write(str(np.round(np.mean([curve[h] for curve in file_curves]),3)) + '\t')
			final_curve.extend([str(rs) for rs in ave_ratio])
			final_curves.append(final_curve)
			matlab_log.write('\n')
		if i == 0:
			p = box_plt(boxplot_curves)
	matlab_log.close()

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

def main():
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	all_rewards = []

	total_path = './new_ave_total_rewards/plot_all_data.txt'
	end_log_file = open(total_path, 'wb')

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
		t_rewards.append('type: ' + str(t))
		for b_l in BUFFER_LENGTHS:
			bl_rewards = []
			curve_rewards = [t, b_l/MS_IN_S]
			bl_rewards.append('buffer: ' + str(b_l/MS_IN_S))

			for lh_l in LH_STEPS:
				lh_rewards = []
				lh_rewards.append('step: ' + str(lh_l))
				if lh_l == -1:
					subopt_dir = './multi_results/type_' + str(t) + '/' 
				else:
					subopt_dir = './multi_subopt_results/type_' + str(t) + '/'
				print subopt_dir

				for file_idx in range(NUM_TRACES):
					if lh_l == -1:
						subopt_path = subopt_dir + 'buffer_' + str(b_l/MS_IN_S) + '_file_' + str(file_idx) + '.txt'
					else:
						subopt_path = subopt_dir + 'buffer_' + str(b_l/MS_IN_S) + '_lh_' + str(lh_l) +  '_file_' + str(file_idx) + '.txt'

					np.random.seed(RANDOM_SEED)

					ave_file = fns.find_file(file_idx)
					cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + ave_file)

					player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
												seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
												start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_FREEZING_TOL + b_l,
												randomSeed=RANDOM_SEED)
					server = testing_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
												start_up_th=b_l, randomSeed=RANDOM_SEED)


					initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
					print initial_delay
					log_path = LOG_FILE + '_buff' + str(b_l) + '_type_' + str(t) + '_step_' + str(lh_l) + '_file_' + str(file_idx) + '.txt'
					log_file = open(log_path, 'wb')

					upper_actions = []
					with open(subopt_path, 'rb') as f:
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
					last_bit_rate = -1
					action_reward = 0.0				# Total reward is for all chunks within on segment
					action_freezing = 0.0
					for i in range(len(upper_actions)):
						print "Current index: ", i
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
					# print(starting_time_idx, ave_file, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
					end_log_file.write('type: ' + str(t) + ' buffer: ' + str(b_l) + ' step: ' + str(lh_l) + ' file: ' + str(file_idx) + ' rewards: ' + str(np.round(np.sum(r_batch), 3)))
					end_log_file.write('\n')
					log_file.write('\t'.join(str(tp) for tp in tp_record))
					log_file.write('\n' + str(starting_time))
					log_file.write('\n')
					log_file.close()

					lh_rewards.append(str(file_idx) + ': ' + str(np.round(np.sum(r_batch), 3)))
				bl_rewards.append(lh_rewards)
			t_rewards.append(bl_rewards)
		all_rewards.append(t_rewards)
	print all_rewards
	end_log_file.close()
	# curves_show(plot_rewards)
if __name__ == '__main__':
	if IF_SIMULATING:
		main()

	collect_results()


	