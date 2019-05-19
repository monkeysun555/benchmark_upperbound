import os
import logging
import numpy as np
import live_player
import live_server
import load
import math 

# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

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
SERVER_START_UP_TH = 3000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound


DEFAULT_ACTION = 0			# lowest bitrate
TYPE = 2
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
LATENCY_MAX = USER_LATENCY_TOL/MS_IN_S
LATENCY_BIN = 0.05
BUFFER_MAX = USER_LATENCY_TOL/MS_IN_S
BUFFER_BIN = 0.05

RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others

# bitrate number is 6, no bin

DATA_DIR = '../bw_traces/'
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# TRACE_NAME = '../bw_traces/BKLYN_1.txt'
# TRACE_NAME = '../bw_traces/70ms_loss0.5_m5.txt'
TRACE_NAME = '../new_traces/test_sim_traces/norway_bus_20'
# TRACE_NAME = '../bw_traces_test/cooked_test_traces/85+-29ms_loss0.5_0_2.txt'



# TRAIN_TRACES = './traces/bandwidth/'

def ReLU(x):
	return x * (x > 0)

def lat_penalty(x):
	return 1.0/(1+math.exp(CONST-X_RATIO*x)) - 1.0/(1+math.exp(CONST))

def main():
	np.random.seed(RANDOM_SEED)
	# create result directory
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)
	# Initial server and player

	cooked_time, cooked_bw = load.new_load_single_trace(TRACE_NAME)
	# cooked_time, cooked_bw = load.load_single_trace(TRACE_NAME)

	player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

	initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
	# print "Initial delay is: ", initial_delay
	# action_num = DEFAULT_ACTION
	# last_bit_rate = action_num
	# bit_rate = action_num
	# last_bit_rate = DEFAULT_ACTION%len(BITRATE)
	# bit_rate = DEFAULT_ACTION%len(BITRATE)
	# playing_speed = NORMAL_PLAYING

	r_table_pre = [[[[float("-inf")] for _ in range(len(BITRATE))] for _ in range(int(LATENCY_MAX/LATENCY_BIN) + 1)] for _ in range(int(BUFFER_MAX/BUFFER_BIN) + 1)]
	pre_value_idx = []
	r_table_curr = [[[[float("-inf")] for _ in range(len(BITRATE))] for _ in range(int(LATENCY_MAX/LATENCY_BIN) + 1)] for _ in range(int(BUFFER_MAX/BUFFER_BIN) + 1)]
	curr_value_idx = []

	init_round_latency = int(initial_delay/MS_IN_S/LATENCY_BIN)
	init_latency_shift = initial_delay - init_round_latency*MS_IN_S*LATENCY_BIN
	r_table_pre[0][init_round_latency][0] = [0.0, [-1], 0, 0.0, init_latency_shift, 0.0]		# accu reward, bitrate seq, and state, buffer shift, time shift
	pre_value_idx = [[0, init_round_latency, 0]]
	ratio = None	# For splitting the chunks

	for seg_idx in range(TEST_DURATION):
		# Here generate several values shared by same 
		print "Current seg_idx is: ", seg_idx
		if CHUNK_IN_SEG == 5:
			ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
		else:
			ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
		for value in pre_value_idx:
			# Get state
			assert not r_table_pre[value[0]][value[1]][value[2]][0] == float("-inf")
			buffer_length = value[0] * BUFFER_BIN * MS_IN_S
			# timing = value[1] * TIMING_BIN * MS_IN_S
			latency = value[1] * LATENCY_BIN * MS_IN_S
			last_bit_rate = value[2]
			# server_timing = timing + initial_delay	# This is the server's time

			element = r_table_pre[value[0]][value[1]][value[2]]
			# print "<<<<<<<<<<||||||||||||||>>>>>>>>>>>>"
			# print "Server time is (before shift): ", server_timing, timing
			# print value, element
			for bit_rate in range(len(BITRATE)):
				action_reward = element[0]
				seq = element[1]
				state = element[2]
				buffer_shift = element[3]
				latency_shift = element[4]
				playing_time = element[5]
				temp_last_bit_rate = last_bit_rate
				# print "bit rate: " + str(bit_rate)
				if not np.round(playing_time, 4) == np.round(seg_idx * SEG_DURATION - buffer_length - buffer_shift, 4):
					print "Not equal"
					print np.round(playing_time, 4)
					print seg_idx * SEG_DURATION
					print np.round(buffer_length, 4)
					print np.round(buffer_shift, 4)
				server_timing = playing_time + latency + latency_shift
				player_timing = server_timing - initial_delay
				# print "playing time: ", playing_time
				# print "latency: ", latency + latency_shift
				# print "Server timing: ", server_timing
				server.clone_from_state(seg_idx, server_timing, bit_rate, ratio)	# Init from last saved state
				player.clone_from_state(player_timing, buffer_length + buffer_shift, state, playing_time)
				take_action = 1
				temp_latency = 0.0
				missing_count = 0.0
				while True:
					download_chunk_info = server.get_next_delivery()
					# print "chunk info is " + str(download_chunk_info)
					download_seg_idx = download_chunk_info[0]
					download_chunk_idx = download_chunk_info[1]
					download_chunk_end_idx = download_chunk_info[2]
					download_chunk_size = download_chunk_info[3]		# Might be several chunks
					chunk_number = download_chunk_end_idx - download_chunk_idx + 1
					assert chunk_number == 1
					server_wait_time = 0.0
					sync = 0
					missing_count = 0
					real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																			download_seg_idx, download_chunk_idx, take_action, chunk_number)
					take_action = 0
					# print "return info ", real_chunk_size, download_duration, freezing, time_out, player_state
					# past_time = download_duration
					temp_buffer_length = player.get_buffer_length()

					server_time = server.update(download_duration)
					if not time_out:
						# server.chunks.pop(0)
						server.clean_next_delivery()
						sync = player.check_resync(server_time)
					else:
						assert player.get_state() == 0
						assert np.round(player.buffer, 3) == 0.0
						# Pay attention here, how time out influence next reward, the smoothness
						# Bit_rate will recalculated later, this is for reward calculation
						bit_rate = 0
						sync = 1
					# Disable sync for current situation
					if sync:
						break
						# To sync player, enter start up phase, buffer becomes zero
						sync_time, missing_count = server.sync_encoding_buffer()
						player.sync_playing(sync_time)
						temp_buffer_length = player.get_buffer_length()

					temp_latency = server.get_time() - player.get_playing_time()
					# print "latency is: ", temp_latency/MS_IN_S
					player_state = player.get_state()

					log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
					if temp_last_bit_rate == -1:
						log_last_bit_rate = log_bit_rate
					else:
						log_last_bit_rate = np.log(BITRATE[temp_last_bit_rate] / BITRATE[0])
					# print(log_bit_rate, log_last_bit_rate)
					if TYPE == 1:
						reward = ACTION_REWARD * log_bit_rate * chunk_number \
								- REBUF_PENALTY * freezing / MS_IN_S \
								- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
								- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(temp_latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number \
								- MISSING_PENALTY * missing_count
								# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
					else:
						reward = ACTION_REWARD * log_bit_rate * chunk_number \
									- REBUF_PENALTY * freezing / MS_IN_S \
									- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
									- LONG_DELAY_PENALTY * lat_penalty(temp_latency/ MS_IN_S) * chunk_number \
									- MISSING_PENALTY * missing_count
					# print(reward)
					action_reward += reward
					temp_last_bit_rate = bit_rate

					# chech whether need to wait, using number of available segs
					if server.check_chunks_empty():
						# print "Enter wait"
						server_wait_time = server.wait()
						# print " Has to wait: ", server_wait_time
						assert server_wait_time > 0.0
						assert server_wait_time < CHUNK_DURATION
						# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
						player.wait(server_wait_time)
						# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
						temp_buffer_length = player.get_buffer_length()

					# print "After wait, ", server.get_time() - (seg_idx + 1) * SEG_DURATION
					server.generate_next_delivery()
					
					# Chech whether a seg is finished
					next_chunk_idx = server.get_next_delivery()[1]
					if next_chunk_idx == 0 or sync:
						# Record state and get reward
						if sync:
							# Process sync
							break
						else:
							# print "end info"
							# print action_reward
							# print "playing time is: " + str(player.get_playing_time())
							# print "real time is: " + str(player.get_real_time())
							# print server.get_time()
							# print temp_buffer_length
							# print "before quantize: ", action_reward, player.get_playing_time(), player.get_real_time(), temp_buffer_length 
							round_buffer_length = int(temp_buffer_length/MS_IN_S/BUFFER_BIN)
							temp_buffer_shift = temp_buffer_length - round_buffer_length * BUFFER_BIN * MS_IN_S
							round_latency = int(temp_latency/MS_IN_S/LATENCY_BIN)
							temp_latency_shift = temp_latency - round_latency * LATENCY_BIN * MS_IN_S 
							temp_playing_time = player.get_playing_time()
							# print "Current playing time: ", temp_playing_time
							# print "after quantize: ", action_reward, player.get_playing_time(), int(np.round(player.get_real_time()/MS_IN_S/TIMING_BIN)), int(np.round(temp_buffer_length/MS_IN_S/BUFFER_BIN)) 
							if round_buffer_length > BUFFER_MAX/BUFFER_BIN or round_latency > LATENCY_MAX/LATENCY_BIN:
								print "Exceed limit, discard!"
								break
							temp_seq = seq[:]
							temp_seq.append(bit_rate)
							# print temp_seq, " seq "
							if [round_buffer_length, round_latency, bit_rate] not in curr_value_idx:
								curr_value_idx.append([round_buffer_length, round_latency, bit_rate])
								temp_state = player.get_state()
								# Check whether there is value already
								assert r_table_curr[round_buffer_length][round_latency][bit_rate][0] == float("-inf")
								r_table_curr[round_buffer_length][round_latency][bit_rate] = [action_reward, temp_seq, temp_state, temp_buffer_shift, temp_latency_shift, temp_playing_time]
							else:
								# print action_reward, " and ", r_table_curr[round_buffer_length][round_latency][bit_rate][0]
								assert not r_table_curr[round_buffer_length][round_latency][bit_rate][0] == float("-inf")
								if action_reward >= r_table_curr[round_buffer_length][round_latency][bit_rate][0]:
									temp_state = player.get_state()
									r_table_curr[round_buffer_length][round_latency][bit_rate] = [action_reward, temp_seq, temp_state, temp_buffer_shift, temp_latency_shift, temp_playing_time]
							# print [round_buffer_length, round_latency, bit_rate], " index saved/or not"
							# print [action_reward, temp_seq, temp_state, temp_buffer_shift, temp_latency_shift, temp_playing_time], " value saved"
							# print "<============================>"
							break
		# print curr_value_idx
		pre_value_idx = curr_value_idx
		curr_value_idx = []
		r_table_pre = r_table_curr
		r_table_curr = [[[[float("-inf")] for _ in range(len(BITRATE))] for _ in range(int(LATENCY_MAX/LATENCY_BIN) + 1)] for _ in range(int(BUFFER_MAX/BUFFER_BIN) + 1)]

	max_reward = float("-inf")
	max_seq = None
	for value in pre_value_idx:
		if r_table_pre[value[0]][value[1]][value[2]][0] > max_reward:
			max_reward = r_table_pre[value[0]][value[1]][value[2]][0]
			max_seq = r_table_pre[value[0]][value[1]][value[2]][1]
	print "Max reward is: ", max_reward
	print "Max sequence is: ", max_seq
	np.savetxt('./results/paper_norway_bus_20_' + str(SERVER_START_UP_TH/MS_IN_S) + '_type_' + str(TYPE) + '.txt', max_seq, fmt='%1.2f')

if __name__ == '__main__':
	main()
