import os
import logging
import numpy as np
import live_player_testing as live_player
# import live_server_testing as live_server
import testing_server as live_server
import load
import math

# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

RANDOM_SEED = 13
RAND_RANGE = 1000
MS_IN_S = 1000.0
KB_IN_MB = 1000.0   # in ms

SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION

# Initial buffer length on server side
SERVER_START_UP_TH = 3000.0             # <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
BUFFER_LENGTHS = [2000.0, 3000.0, 4000.0]
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0                          # Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL           # Accumulate latency upperbound

DEFAULT_ACTION = 0          # lowest bitrate

ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO   
REBUF_PENALTY = 6.0     # for second
SMOOTH_PENALTY = 1.0
# LONG_DELAY_PENALTY_BASE = 1.2 # for second
LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
CONST = 6.0
X_RATIO = 1.0
MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO         # not included

# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1        # For 1
# NORMAL_PLAYING = 1.0  # For 0
# SLOW_PLAYING = 0.9        # For -1

TEST_DURATION = 100             # Number of testing <===================== Change length here
TIMING_MAX = TEST_DURATION*SEG_DURATION/MS_IN_S + 10.0
TIMING_BIN = 0.05
BUFFER_MAX = USER_LATENCY_TOL/MS_IN_S
BUFFER_BIN = 0.05

RATIO_LOW_2 = 2.0               # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0         # This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75              # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0          # This is the highest ratio between first chunk and the sum of all others

DATA_DIR = '../bw_traces_test/cooked_test_traces/'
# TRACE_NAME = '85+-29ms_loss0.5_0_2.txt'
TRACE_NAME = '70+-24ms_loss1_2_1.txt'

SUMMARY_DIR = './ton_massive_opt_result/'
LOG_FILE = SUMMARY_DIR + '/opt_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'

ALL_TESTING_DIR = '../benchmark_compare/all_results_old/'
ALL_TESTING_FILE = ALL_TESTING_DIR + 'opt_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's.txt'

OPT_SEQ = './massive_opt/actions_buffer_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '.txt'

def ReLU(x):
    return x * (x > 0)

def lat_penalty(x):
    return 1.0/(1+math.exp(CONST-X_RATIO*x)) - 1.0/(1+math.exp(CONST))

def record_tp(tp_trace, time_trace, starting_time_idx, duration):
    tp_record = []
    time_record = []
    offset = 0
    time_offset = 0.0
    num_record = int(np.ceil(duration/SEG_DURATION))
    for i in range(num_record):
        if starting_time_idx + i + offset >= len(tp_trace):
            offset = -len(tp_trace)
            time_offset += time_trace[-1]
        tp_record.append(tp_trace[starting_time_idx + i + offset])
        time_record.append(time_trace[starting_time_idx + i + offset] + time_offset)
    return tp_record, time_record

# def curves_show(sub_r):
#   assert len(sub_r) == len(TYPES) * len(BUFFER_LENGTHS)
#   for i in range(len(TYPES)):
#       for j in range(len(BUFFER_LENGTHS)):
#           curr_curve = sub_r[i*len(TYPES)+j]

def main():
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    all_testing_log = open(ALL_TESTING_FILE, 'w')

    cooked_times, cooked_bws, cooked_names = load.loadBandwidth(DATA_DIR)
    
    # load optimal bitrate sequence
    traces = {}
    with open(OPT_SEQ, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            name = line[0]
            # print(line[4:])
            seq = line[4:]
            seq = [0, 0] + seq[:-2]
            # print(len(seq))
            assert len(seq) == 100
            traces[name] = [int(x) for x in seq]

    for i in range(len(cooked_times)):
        np.random.seed(RANDOM_SEED)
        cooked_time = cooked_times[i]
        cooked_bw = cooked_bws[i]
        cooked_name = cooked_names[i]

        player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
                                    seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
                                    start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_FREEZING_TOL + SERVER_START_UP_TH,
                                    randomSeed=RANDOM_SEED)
        server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
                                start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

        print(server.get_time())
        log_path = LOG_FILE + '_' + cooked_name
        log_file = open(log_path, 'w')

        action_seq = traces[cooked_name]

        r_batch = []
        f_batch = []
        a_batch = []
        c_batch = []
        l_batch = []

        action_reward = 0.0     # Total reward is for all chunks within on segment
        action_freezing = 0.0
        action_wait = 0.0
        take_action = 1
        latency = 0.0
        starting_time = server.get_time()
        starting_time_idx = player.get_time_idx()
        init = 1
        
        # bit_rate = action_seq[0]
        last_bit_rate = action_seq[0]

        for i in range(TEST_DURATION):
            # Get action from seq at last
            # if i <= 1:
            #     bit_rate = 0
            # else:
            bit_rate = action_seq[i]
            c_batch.append(np.abs(BITRATE[bit_rate] - BITRATE[last_bit_rate]))

            # print "Current index: ", i
            # print server.get_time()
            if init: 
                if CHUNK_IN_SEG == 5:
                    ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
                else:
                    ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
                
                server.set_ratio(ratio)
                server.init_encoding()
                init = 0
            action_reward = 0.0 
            take_action = 1

            while True:  # serve video forever
                download_chunk_info = server.get_next_delivery()
                # print "chunk info is " + str(download_chunk_info)
                download_seg_idx = download_chunk_info[0]
                download_chunk_idx = download_chunk_info[1]
                download_chunk_end_idx = download_chunk_info[2]
                download_chunk_size = download_chunk_info[3][bit_rate]      # Might be several chunks
                chunk_number = download_chunk_end_idx - download_chunk_idx + 1
                if download_seg_idx >= TEST_DURATION:
                    break
                assert chunk_number == 1
                server_wait_time = 0.0
                sync = 0
                missing_count = 0
                real_chunk_size, download_duration, freezing, time_out, player_state, rtt = player.fetch(download_chunk_size, 
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
                    # break # No resync here
                    # To sync player, enter start up phase, buffer becomes zero
                    sync_time, missing_count = server.sync_encoding_buffer()
                    player.sync_playing(sync_time)
                    buffer_length = player.get_buffer_length()

                latency = server.get_time() - player.get_playing_time()
                # print "latency is: ", latency/MS_IN_S
                player_state = player.get_state()

                log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
                log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])

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
                last_bit_rate = bit_rate    # Do no move this term. This is for chunk continuous calcualtion

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
                    #   # Process sync
                    #   assert 0 == 1
                    #   pass
                    # else:
                    take_action = 1
                    # last_bit_rate = bit_rate
                    r_batch.append(action_reward)
                    f_batch.append(action_freezing)
                    a_batch.append(BITRATE[bit_rate])
                    l_batch.append(latency)
                    # print(action_reward)
                    log_file.write( str(server.get_time()) + '\t' +
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
                    action_reward = 0.0
                    action_freezing = 0.0
                    action_wait = 0.0
                    break

        time_duration = server.get_time() - starting_time
        tp_record, time_record = record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
        print(starting_time_idx, cooked_name, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
        log_file.write('\t'.join(str(tp) for tp in tp_record))
        log_file.write('\n')

        log_file.write('\t'.join(str(time) for time in time_record))
        # log_file.write('\n' + str(IF_NEW))
        log_file.write('\n' + str(starting_time))
        log_file.write('\n')
        log_file.close()

        all_testing_log.write(cooked_name + '\t')
        all_testing_log.write(str(np.sum(r_batch)) + '\t')
        all_testing_log.write(str(np.mean(a_batch)) + '\t')
        all_testing_log.write(str(np.sum(f_batch)) + '\t')
        all_testing_log.write(str(np.mean(c_batch)) + '\t')
        all_testing_log.write(str(np.mean(l_batch)) + '\t')
        print(np.sum(r_batch))

        all_testing_log.write('\n')
    all_testing_log.close()

        # curves_show(plot_rewards)


if __name__ == '__main__':
    main()


