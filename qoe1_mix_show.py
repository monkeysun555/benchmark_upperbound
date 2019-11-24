import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True

SERVER_START_UP_TH = 2000.0			#<=== change this to get corresponding log files
BUFFER_LENGTHS = [2000.0, 3000.0, 4000.0]
MS_IN_S = 1000.0
RESULT_DIR = './sub_test_results/'
FIGURES_DIR = './test_figures/'
RESULT_FILE = './test_figures/'

       
PLT_BUFFER_A = 1e-10		#ms
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
SAVE = 1

SEG_DURATION = 1000.0
CHUNK_DURATION = 200.0
START_UP_TH = 2000.0

CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)		# 4

def plt_fig_mix_bw_action(tp_trace, bitrates):
	colors = ['dodgerblue','chocolate','forestgreen']
	types = ['-','-','-', '--']
	# type1: tp,  type2: bitrate
	y_axis_upper = 10000.0
	# For negative reward
	# y_axis_lower = np.floor(np.minimum(np.min(trace)*1.1,0.0))
	y_axis_lower = 0.0
	print len(tp_trace)
	print len(bitrates[0])
	p = plt.figure(figsize=(20,5))
	for i in range(len(bitrates)):
		x_value = []
		y_value = []
		curr_x = 0.0
		for j in range(len(bitrates[i])/CHUNK_IN_SEG):
			x_value.append(curr_x)
			x_value.append(curr_x+0.999)
			y_value.append(bitrates[i][j*CHUNK_IN_SEG])
			y_value.append(bitrates[i][j*CHUNK_IN_SEG])
			curr_x += 1		# Plot in chunks

		plt.plot(x_value, y_value, types[i], color=colors[i], linewidth=3, alpha=0.5)
	
	plt.plot(range(1,len(tp_trace)+1), tp_trace*KB_IN_MB, color='k', linewidth=1.5,alpha=0.9)

	plt.legend(('Buffer Length(' + r'$\alpha$' + '=2)', \
		'Buffer Length(' + r'$\alpha$' + '=3)', \
		'Buffer Length(' + r'$\alpha$' + '=4)',\
		'Bandwidth'), loc='upper center', fontsize=24, ncol=4, borderpad=0.25, frameon=False)

	plt.grid(linestyle='dashed', axis='y',linewidth=0.5, color='gray')
	plt.axis([0, int(len(tp_trace)/SEG_DURATION*MS_IN_S), y_axis_lower, y_axis_upper])
	plt.xticks(np.arange(0, int(len(tp_trace)/SEG_DURATION*MS_IN_S)+1, 50))
	plt.xlabel('Time(s)', fontweight='bold', fontsize=22)
	plt.yticks(np.arange(0,y_axis_upper,2000), np.arange(0,10,2) )
	plt.ylabel('Bitrate (Mbps)', fontweight='bold', fontsize=22)
	p.set_tight_layout(True)
	plt.tick_params(labelsize=22)

	plt.close()
	return p


def plt_buffer_mix(time_traces, buffer_traces, state_traces, latency_traces, data_type):
	# y_axis_upper = np.ceil(np.max(latency_trace)*1.6/MS_IN_S)
	colors = ['dodgerblue','chocolate','forestgreen']
	types = ['-', '--']
	y_axis_upper = 6.5
	insert_buffer_traces = []
	insert_time_traces = []
	plt_buffer_traces = []
	plt_time_traces = []
	latency_times = []
	for i in range(len(time_traces)):
		latency_traces[i] = [lat/MS_IN_S for lat in latency_traces[i]]
		latency_time = [time/MS_IN_S for time in time_traces[i]]
		latency_times.append(latency_time)
		time_traces[i] = [0.0] + time_traces[i]
		buffer_traces[i] = [0.0] + buffer_traces[i]
		state_traces[i] = [0] + state_traces[i]
		insert_buffer_trace = []
		insert_time_trace = []
		plot_state_left = 1
		assert len(time_traces[i]) == len(buffer_traces[i])
		for j in range(0, len(time_traces[i])):
			if state_traces[i][j] == 0:
				if j >= 1:
					if state_traces[i][j-1] == 1:
						insert_time = np.minimum(time_traces[i][j] - PLT_BUFFER_A, time_traces[i][j-1]+buffer_traces[i][j-1])
						insert_buffer = np.maximum(0.0, buffer_traces[i][j-1] - (time_traces[i][j] - time_traces[i][j-1]))
						insert_buffer_trace.append(insert_buffer)
						insert_time_trace.append(insert_time)
				plot_state_left = 1
				continue
			else:
				if not plot_state_left == 0:
					plot_state_left -= 1
					continue
				insert_buffer = np.maximum(0.0, buffer_traces[i][j-1] - (time_traces[i][j] - time_traces[i][j-1]))
				insert_time = np.minimum(time_traces[i][j] - PLT_BUFFER_A, time_traces[i][j-1]+buffer_traces[i][j-1])
				insert_buffer_trace.append(insert_buffer)
				insert_time_trace.append(insert_time)
				if insert_time < time_traces[i][j] - PLT_BUFFER_A:
					assert insert_buffer == 0.0
					insert_buffer_trace.append(0.0)
					insert_time_trace.append(time_traces[i][j] - PLT_BUFFER_A)

		insert_buffer_traces.append(insert_buffer_trace)
		insert_time_traces.append(insert_time_trace)

		# Need to adjust about freezing
		# combine two buffer_traces
		plt_buffer_trace = []
		plt_time_trace = []
		# print(len(insert_time_trace), len(time_traces[i]), len(latency_traces[i]))
		# print(insert_time_trace[-1], time_trace[-1])
		# print(latency_trace)
		# print(len(insert_time_trace), len(insert_buffer_trace))
		for j in range(len(time_traces[i])):
			# if len(insert_time_trace) == 0:
			# 	plt_time_trace.append(time_trace[i:])
			# 	plt_buffer_trace.append(buffer_trace[i:])
			# 	break
			# print(i, len(time_trace))
			if len(insert_time_trace) > 0:
				while insert_time_trace[0] < time_traces[i][j]:
					plt_time_trace.append(insert_time_trace.pop(0)/MS_IN_S)
					plt_buffer_trace.append(insert_buffer_trace.pop(0)/MS_IN_S)
					# print(len(insert_time_trace), len(time_trace), i)
					if len(insert_time_trace) == 0:
						# plt_time_trace.extend(time_trace[i:])
						# plt_buffer_trace.extend(buffer_trace[i:])
						break
			plt_time_trace.append(time_traces[i][j]/MS_IN_S)
			plt_buffer_trace.append(buffer_traces[i][j]/MS_IN_S)

		plt_buffer_traces.append(plt_buffer_trace)
		plt_time_traces.append(plt_time_trace)


	p = plt.figure(figsize=(20,5))
	# for i in range(1):
	for i in range(len(plt_time_traces)):
		# print len(plt_time_traces[i])
		# print len(plt_buffer_traces[i])
		# print len(latency_times[i])
		# print len(latency_traces[i])
		plt.plot(plt_time_traces[i], plt_buffer_traces[i], types[0], color=colors[i], linewidth=1.5, alpha=0.9)
		plt.plot(latency_times[i], latency_traces[i], types[1], color=colors[i], linewidth=3, alpha=0.9)
		# plt.plot([latency_time[0], latency_time[-1]], [starting_time/MS_IN_S]*2, 'r--', linewidth=1.5,alpha=0.9)
	
	plt.legend(('Buffer Length(' + r'$\alpha$' + '=2)', r'Latency (' + r'$\alpha$' + '=2)', \
				'Buffer Length(' + r'$\alpha$' + '=3)', r'Latency (' + r'$\alpha$' + '=3)', \
				'Buffer Length(' + r'$\alpha$' + '=4)', r'Latency (' + r'$\alpha$' + '=4)'), \
				loc='upper center',fontsize=24 ,ncol=3, borderpad=0.25, frameon=False)
	# plt.grid(linestyle='dashed', axis='y',linewidth=0.5, color='gray')
	plt.axis([0, plt_time_traces[-1][-1], 0, y_axis_upper])
	plt.xticks(np.arange(0, plt_time_traces[-1][-1]+1, 50))
	plt.xlabel('Time(s)', fontweight='bold', fontsize=22)
	plt.yticks(np.arange(0,y_axis_upper,2))
	plt.ylabel('Buffer(s)', fontweight='bold', fontsize=22)
	p.set_tight_layout(True)
	plt.tick_params(labelsize=22)
	
	# plt.yticks(np.arange(200, 1200+1, 200))

	plt.close()
	return p




def main():
	if not os.path.isdir(FIGURES_DIR):
		os.makedirs(FIGURES_DIR)
	if not os.path.isdir(RESULT_FILE):
		os.makedirs(RESULT_FILE)

	results = os.listdir(RESULT_DIR)
	file_records = []
	for buffer_length in BUFFER_LENGTHS:
		file_info = []
		file_path = RESULT_DIR + 'subupper_buff' + str(buffer_length) + '_type_2_step_-1.txt'
		file_info.append(buffer_length)
		with open(file_path, 'rb') as f:
			for line in f:
				parse = line.strip('\n')
				parse = parse.split('\t')				
				file_info.append(parse)
		file_records.append(file_info)

	# For figs
	tp_figs = []
	reward_figs = []
	bitrate_figs = []
	buffer_figs = []
	freezing_figs = []
	server_wait_figs = []
	# missing_figs = []
	speed_figs = []
	server_mix_figs = []

	# For numericals
	n_files = ['files']
	n_throughput = ['throughut']
	n_reward = ['reward']
	n_freezing = ['freezing']
	n_n_freezing = ['#freezing']
	n_latency = ['latency']
	n_switch = ['switch']
	n_sync = ['sync']
	n_rate = ['rate']
	n_starting_time  = ['starting_time']

	log_path = RESULT_FILE + 'table_upper'
	log_file = open(log_path, 'wb')

	# Total plot infomations
	names = []
	tp_traces = []
	starting_times = []
	bitrate_traces = []
	freezing_traces = []
	buffer_traces = []
	latency_traces = []
	plt_time_traces = []
	state_traces = []

	for i in range(len(file_records)):
		records = file_records[i][1:-2]
		bitrate_trace = [float(info[1]) for info in records]
		buffer_trace = [float(info[2]) for info in records]
		freezing_trace = [float(info[3]) for info in records]
		server_wait_trace = [float(info[5]) for info in records]
		sync_trace = [float(info[6]) for info in records]
		latency_trace = [float(info[7]) for info in records]
		state_trace = [float(info[8]) for info in records]
		reward_trace = [float(info[-1]) for info in records]

		# For total plot
		starting_times.append(float(file_records[i][-1][0]))
		tp_traces.append(np.array(file_records[i][-2]).astype(np.float))
		names.append(file_records[i][0])
		bitrate_traces.append(bitrate_trace)
		freezing_traces.append(freezing_trace)
		buffer_traces.append(buffer_trace)
		latency_traces.append(latency_trace)
		state_traces.append(state_trace)

		# Time
		real_time_trace = [float(info[0]) for info in records]
		plt_time_trace = [r_time - starting_times[i] for r_time in real_time_trace]
		plt_time_traces.append(plt_time_trace)


	# Plot total figures
	
	buffer_fig = plt_buffer_mix(plt_time_traces, buffer_traces, state_traces, latency_traces, 'buffer')
	# buffer_figs.append([data_name, 'buffer', buffer_fig])
	
	server_mix_fig = plt_fig_mix_bw_action(tp_traces[0], bitrate_traces)

	buffer_fig.savefig(FIGURES_DIR +  't2_buffers.eps', format='eps', dpi=1000, figsize=(20, 5))
	server_mix_fig.savefig(FIGURES_DIR +  't2_bitrates.eps', format='eps', dpi=1000, figsize=(20, 5))

if __name__ == '__main__':
	main()
