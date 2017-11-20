from keras.utils import np_utils
import numpy as np

# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data(filename, alphabet, seq_len, num_classes):
	f = open(filename,'r')
	
	# Create a list of the alphabet
	char_list = []
	for i in alphabet:
		char_list.append(i)

	all_seq_one_hot=[]	
	labels=[]
	
	for line in f:
		l = line.strip().split('\t')
		seq = l[0]
		category = l[1]
		
		# Cut off excess characters from the end
		if len(seq)>seq_len :
			seq=seq[:seq_len]

		# Convert characters to character indices	
		index_seq=[]
		for i in seq:
			if i not in char_list:
				index = -1
			else:
				index = char_list.index(i)
			index_seq.append(index)
		
		# Padding with index '-1' if the sequence is not long enough
		# NOTE: '-1' is also used as the index of unknown characters (padding can be treated as unknown anyway) 
		if len(index_seq)<seq_len:
			for _ in range(seq_len-len(index_seq)):
				index_seq.append(-1)

		# Convert indices to one-hot-vectors, '-1' becomes the all-zeros vector
		index_seq_one_hot = []
		for i in index_seq:
			all_zeroes = [0]*len(char_list)
			if i>=0: 
				all_zeroes[i] = 1
			index_seq_one_hot.append(all_zeroes)

		all_seq_one_hot.append(index_seq_one_hot)
		labels.append(category)

	x = np.array(all_seq_one_hot, dtype='float32')
	
	# Convert integer class labels to one-hot vectors
	all_labels_one_hot = []
	for i in labels:
		one_hot = [0]*num_classes
		one_hot[int(i)]=1
		all_labels_one_hot.append(one_hot)
	y = np.array(all_labels_one_hot, dtype='float32')
	
	return x,y











		
		




