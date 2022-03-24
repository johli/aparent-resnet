import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

	fp = FontProperties(family="DejaVu Sans", weight="bold")
	globscale = 1.35
	LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
				"G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
				"A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
				"C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
				"UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
				"DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
				"(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
				"." : TextPath((-0.125, 0), "-", size=1, prop=fp),
				")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
	COLOR_SCHEME = {'G': 'orange', 
					'A': 'red', 
					'C': 'blue', 
					'T': 'darkgreen',
					'UP': 'green', 
					'DN': 'red',
					'(': 'black',
					'.': 'black', 
					')': 'black'}


	text = LETTERS[letter]

	chosen_color = COLOR_SCHEME[letter]
	if color is not None :
		chosen_color = color

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
	if ax != None:
		ax.add_artist(p)
	return p

def protein_letter_at(letter, x, y, yscale=1, ax=None, color='black', alpha=1.0):

	#fp = FontProperties(family="Arial", weight="bold")
	#fp = FontProperties(family="Ubuntu", weight="bold")
	fp = FontProperties(family="DejaVu Sans", weight="bold")
	
	globscale = 1.35
	LETTERS = {"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
			   "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
			   "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
			   "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
			   
			   "L" : TextPath((-0.35, 0), "L", size=1, prop=fp),
			   "M" : TextPath((-0.35, 0), "M", size=1, prop=fp),
			   "F" : TextPath((-0.35, 0), "F", size=1, prop=fp),
			   "W" : TextPath((-0.35, 0), "W", size=1, prop=fp),
			   "K" : TextPath((-0.35, 0), "K", size=1, prop=fp),
			   "Q" : TextPath((-0.35, 0), "Q", size=1, prop=fp),
			   "E" : TextPath((-0.35, 0), "E", size=1, prop=fp),
			   "S" : TextPath((-0.35, 0), "S", size=1, prop=fp),
			   "P" : TextPath((-0.35, 0), "P", size=1, prop=fp),
			   "V" : TextPath((-0.35, 0), "V", size=1, prop=fp),
			   "I" : TextPath((-0.35, 0), "I", size=1, prop=fp),
			   "Y" : TextPath((-0.35, 0), "Y", size=1, prop=fp),
			   "H" : TextPath((-0.35, 0), "H", size=1, prop=fp),
			   "R" : TextPath((-0.35, 0), "R", size=1, prop=fp),
			   "N" : TextPath((-0.35, 0), "N", size=1, prop=fp),
			   "D" : TextPath((-0.35, 0), "D", size=1, prop=fp),
			   "U" : TextPath((-0.35, 0), "U", size=1, prop=fp),
			   "!" : TextPath((-0.35, 0), "!", size=1, prop=fp),
			   
			   "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
			   "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
			   "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
			   "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
			   ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}

	
	if letter in LETTERS :
		text = LETTERS[letter]
	else :
		text = TextPath((-0.35, 0), letter, size=1, prop=fp)
	
	chosen_color = color
	
	if chosen_color is None :
		chosen_color = 'black'
		if letter in ['A', 'I', 'L', 'M', 'F', 'W', 'V'] : #Hydrophobic
			chosen_color = 'blue'
		elif letter in ['K' ,'R'] : #Positive charge
			chosen_color = 'red'
		elif letter in ['E', 'D'] : #Negative charge
			chosen_color = 'magenta'
		elif letter in ['N', 'Q', 'S', 'T'] : #Polar
			chosen_color = 'green'
		elif letter in ['C'] : #Cysteines
			chosen_color = 'pink'
		elif letter in ['G'] : #Glycines
			chosen_color = 'orange'
		elif letter in ['P'] : #Prolines
			chosen_color = 'yellow'
		elif letter in ['H', 'Y'] : #Aromatic
			chosen_color = 'cyan'

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
	if ax != None:
		ax.add_artist(p)
	return p

def plot_dna_logo(pwm, sequence_template=None, figsize=(12, 3), logo_height=1.0, plot_start=0, plot_end=164, plot_sequence_template=False, save_figs=False, fig_name=None) :

	#Slice according to seq trim index
	pwm = np.copy(pwm[plot_start: plot_end, :])
	sequence_template = sequence_template[plot_start: plot_end]

	pwm += 0.0001
	for j in range(0, pwm.shape[0]) :
		pwm[j, :] /= np.sum(pwm[j, :])

	entropy = np.zeros(pwm.shape)
	entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
	entropy = np.sum(entropy, axis=1)
	conservation = 2 - entropy

	fig = plt.figure(figsize=figsize)

	ax = plt.gca()

	height_base = (1.0 - logo_height) / 2.

	for j in range(0, pwm.shape[0]) :
		sort_index = np.argsort(pwm[j, :])

		for ii in range(0, 4) :
			i = sort_index[ii]

			nt_prob = pwm[j, i] * conservation[j]

			nt = ''
			if i == 0 :
				nt = 'A'
			elif i == 1 :
				nt = 'C'
			elif i == 2 :
				nt = 'G'
			elif i == 3 :
				nt = 'T'

			color = None
			if sequence_template[j] != '$' :
				color = 'black'
				if plot_sequence_template and nt == sequence_template[j] :
					nt_prob = 2.0
				else :
					nt_prob = 0.0

			if ii == 0 :
				dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
			else :
				prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
				dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)

	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, 2))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.axis('off')
	plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)

	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)

	plt.tight_layout()
	
	if save_figs :
		plt.savefig(fig_name + ".png", transparent=True, dpi=300)
		plt.savefig(fig_name + ".eps")

	plt.show()

def plot_dna_importance_scores(importance_scores, ref_seq, figsize=(12, 2), score_clip=None, sequence_template='', fixed_sequence_template_scores=True, plot_start=0, plot_end=96, plot_height=None, save_figs=False, fig_name=None) :

	end_pos = ref_seq.find("#")
	
	fig = plt.figure(figsize=figsize)
	
	ax = plt.gca()
	
	if score_clip is not None :
		importance_scores = np.clip(np.copy(importance_scores), -score_clip, score_clip)
	
	max_score = np.max(np.sum(importance_scores[:, :], axis=0)) + 0.01

	for i in range(0, len(ref_seq)) :
		mutability_score = np.sum(importance_scores[:, i])
		color = None
		if sequence_template is not None and sequence_template != '' and sequence_template[i] != '$' :
			color = 'black'
			if fixed_sequence_template_scores :
				mutability_score = max_score
		dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

	plt.sca(ax)
	plt.xlim(0, plot_end - plot_start)
	plt.ylim(0, max_score if plot_height is None else plot_height)
	plt.axis('off')
	#plt.yticks([0.0, max_score], [0.0, max_score], fontsize=16)
	plt.axhline(y=0.01, color='black', linestyle='-', linewidth=2)

	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
	
	plt.tight_layout()
	
	if save_figs :
		plt.savefig(fig_name + ".png", transparent=True, dpi=300)
		plt.savefig(fig_name + ".eps")
	
	plt.show()

def plot_protein_logo(residue_map, pwm, sequence_template=None, figsize=(12, 3), logo_height=1.0, plot_start=0, plot_end=164, color_reference=None, sequence_colors=None, save_figs=False, fig_name=None) :

	inv_residue_map = {
		i : sp for sp, i in residue_map.items()
	}

	#Slice according to seq trim index
	pwm = pwm[plot_start: plot_end, :]
	sequence_template = sequence_template[plot_start: plot_end]

	entropy = np.zeros(pwm.shape)
	entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(np.clip(pwm[pwm > 0], 1e-6, 1. - 1e-6))
	entropy = np.sum(entropy, axis=1)
	conservation = np.log2(len(residue_map)) - entropy#2 - entropy

	fig = plt.figure(figsize=figsize)

	height_base = (1.0 - logo_height) / 2.

	for j in range(0, pwm.shape[0]) :
		sort_index = np.argsort(pwm[j, :])

		for ii in range(0, len(residue_map)) :
			i = sort_index[ii]

			nt_prob = pwm[j, i] * conservation[j]

			nt = inv_residue_map[i]

			color = None

			if color_reference is not None :
				if sequence_colors[j] != -1 and sequence_colors[j] >= 0 and sequence_colors[j] < len(color_reference) :
					color = color_reference[sequence_colors[j]]
				else :
					color = 'black'

			if sequence_template[j] != '$' :
				color = 'black'

			if ii == 0 :
				protein_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, plt.gca(), color=color)
			else :
				prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
				protein_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, plt.gca(), color=color)

	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, np.log2(len(residue_map))))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.axis('off')
	plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)


	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)

	plt.tight_layout()

	if save_figs :
		plt.savefig(fig_name + ".png", transparent=True, dpi=300)
		plt.savefig(fig_name + ".eps")

	plt.show()

def plot_protein_importance_scores(importance_scores, ref_seq, figsize=(12, 2), score_clip=None, single_color=None, sequence_template='', fixed_sequence_template_scores=True, plot_start=0, plot_end=96, save_figs=False, fig_name=None) :

	end_pos = ref_seq.find("#")
	
	fig = plt.figure(figsize=figsize)
	
	ax = plt.gca()
	
	if score_clip is not None :
		importance_scores = np.clip(np.copy(importance_scores), -score_clip, score_clip)
	
	max_score = np.max(np.sum(importance_scores[:, :], axis=0)) + 0.01

	for i in range(0, len(ref_seq)) :
		mutability_score = np.sum(importance_scores[:, i])
		color = None if single_color is None else single_color
		if sequence_template is not None and sequence_template != '' and sequence_template[i] != '$' :
			color = 'black'
			if fixed_sequence_template_scores :
				mutability_score = max_score
		protein_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

	plt.sca(ax)
	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, max_score))
	plt.axis('off')
	plt.yticks([0.0, max_score], [0.0, max_score], fontsize=16)
	plt.axhline(y=0.01, color='black', linestyle='-', linewidth=2)
	
	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
	
	plt.tight_layout()

	if save_figs :
		plt.savefig(fig_name + ".png", transparent=True, dpi=300)
		plt.savefig(fig_name + ".eps")

	plt.show()