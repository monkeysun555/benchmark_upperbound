import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True

qoe_file = './sub_test_results/t2_f3.txt'
# Using same trace 70 0.5 m5

new_palette = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
                  '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
patterns = [ "/" , "|" , "\\"  , "-" , "+" , "x", "o", "O", ".", "*" ]


def lat_penalty(x):
	return 1.0/(1+math.exp(6.0- x)) - 1.0/(1+math.exp(6.0))

def show_bar(buffer_2_r, new_buffer_2_r, buffer_3_r, new_buffer_3_r, buffer_4_r, new_buffer_4_r):
	N = 6 
	upper_bound = np.amax(new_buffer_4_r)
	barWidth = 0.35
	r1 = [2*x + 0.1  for x in range(N)]
	r2 = [x + 1.3*barWidth for x in r1]
	r3 = [x + 2.6*barWidth for x in r1]

	for i in range(len(new_buffer_2_r)):
		new_buffer_2_r[i] = new_buffer_2_r[i] - buffer_2_r[i]
		new_buffer_3_r[i] = new_buffer_3_r[i] - buffer_3_r[i]
		new_buffer_4_r[i] = new_buffer_4_r[i] - buffer_4_r[i]

	p = plt.figure(figsize=(7,5.5))
	
	plt.bar(r1, buffer_2_r, color='none', width=barWidth, edgecolor=new_palette[1], \
				hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='QoE(' + r'$\alpha$' + '=2'+ r'$\Delta$' + ')')
	plt.bar(r1, buffer_2_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1)

	plt.bar(r1, new_buffer_2_r, color='none', width=barWidth, edgecolor=new_palette[4], \
				hatch=patterns[3]*6, linewidth=1.0, zorder = 0, bottom=buffer_2_r, label='QoE\'(' + r'$\alpha$' + '=2'+ r'$\Delta$' + ')')
	plt.bar(r1, new_buffer_2_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1, bottom=buffer_2_r)


	####
	plt.bar(r2, buffer_3_r, color='none', width=barWidth, edgecolor=new_palette[2], \
				hatch=patterns[1]*6, linewidth=1.0, zorder = 0, label='QoE(' + r'$\alpha$' + '=3'+ r'$\Delta$' + ')')
	plt.bar(r2, buffer_3_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1)

	plt.bar(r2, new_buffer_3_r, color='none', width=barWidth, edgecolor=new_palette[5], \
				hatch=patterns[4]*6, linewidth=1.0, zorder = 0, bottom=buffer_3_r, label='QoE\'(' + r'$\alpha$' + '=3'+ r'$\Delta$' + ')')
	plt.bar(r2, new_buffer_3_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1,  bottom=buffer_3_r)
	# mpl.rcParams['hatch.linewidth'] = 4.0  # previous pdf hatch linewidth

	####
	plt.bar(r3, buffer_4_r, color='none', width=barWidth, edgecolor=new_palette[0], \
				hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='QoE(' + r'$\alpha$' + '=4'+ r'$\Delta$' + ')')
	plt.bar(r3, buffer_4_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1)

	plt.bar(r3, new_buffer_4_r, color='none', width=barWidth, edgecolor=new_palette[6], \
				hatch=patterns[5]*6, linewidth=1.0, zorder = 0, bottom=buffer_4_r, label='QoE\'(' + r'$\alpha$' + '=4'+ r'$\Delta$' + ')')
	plt.bar(r3, new_buffer_4_r, color='none', width=barWidth, edgecolor='k', linewidth=1.5, zorder = 1, bottom=buffer_4_r)

	# horizontal_x = [0., r2[-1] + 1.5*barWidth]
	# horizontal_y = [upper_bound for x in horizontal_x]
	# plt.plot(horizontal_x, horizontal_y, '-', color = 'k', linewidth=3, label='Upperbound',zorder = 2)

	plt.xlabel('Lookahead Horizon', fontsize=22)
	plt.xticks([2.*r + 2.*barWidth for r in range(N)], ['1', '2', '3', '5', '10', '100'], fontsize=22)
	plt.yticks([40, 150, 250], fontsize=22)
	plt.ylabel('Accumulate QoE (QoE\')', fontsize=22)
	plt.axis([0, r3[-1] + 1.3*barWidth, 40, 290])
	plt.legend(ncol=3, fontsize = 15, loc='upper left', columnspacing=0.7, frameon=False, handletextpad=0.1)
	p.set_tight_layout(True)
	p.show()
	raw_input()
	return p
def main():
	buffer_2_r = []
	buffer_3_r = []
	buffer_4_r = []
	new_buffer_2_r = []
	new_buffer_3_r = []
	new_buffer_4_r = []	
	with open(qoe_file, 'rb') as f:
		for line in f:
			parse = line.strip('\n')
			parse = parse.split(' ')
			buffer_name = parse[3]
			step_name = parse[5]
			rewards = parse[7]
			new_rewards = parse[-1]

			if buffer_name == '2000.0':
				buffer_2_r.append(float(rewards))
				new_buffer_2_r.append(float(new_rewards))
			elif buffer_name == '3000.0':
				buffer_3_r.append(float(rewards))
				new_buffer_3_r.append(float(new_rewards))
			else:
				buffer_4_r.append(float(rewards))
				new_buffer_4_r.append(float(new_rewards))

	# print buffer_2_r, buffer_3_r, buffer_4_r
	fig_p = show_bar(buffer_2_r, new_buffer_2_r, buffer_3_r, new_buffer_3_r, buffer_4_r, new_buffer_4_r)
	fig_p.savefig('qoe_bar.eps', format='eps', dpi=1000, figsize=(7, 4.5), bbox_inches='tight')

if __name__ == '__main__':
	main()