"""******************************************************************************************
VIBRAINT PROJECT - DATA GENERATOR
THIS IS A DATA GENERATOR PROGRAM, WHICH CAN GENERATE A RANDOM SEQUENCES OF LENGTH n_steps_in
BASED ON THE PROBABILITIES SPECIFIED IN THE MENU CSV FILE. IT GENERATES n_samples NUMBER
OF SEQUENCES. 
ALSO, GENERATED SEQUENCES ARE WRITTEN IN CORRESPONDING input_seq CSV FILES
OUTPUT FROM THIS PROGRAM ARE THE SOURCE AND EXPECTED OUTPUT VALUES IN AN ONE-HOT-ENCODED ARRAY
******************************************************************************************"""
#IMPORT THE INITILIZER PROGRAM
import numpy as np
import pandas as pd
from random import choices
import csv
from keras.utils import to_categorical

# generate a sequence 
def generate_sequence(length, n_samples,menu_file,input_seq):
	df = pd.read_csv(menu_file)
	with open(input_seq, 'w', newline="") as writeFile:
		writer = csv.writer(writeFile,delimiter =',')
		for _ in range(n_samples):
			seq = np.zeros(length,dtype=int)
			seq = choices(df['selectionId'], df['Morning'], k=length)
			writer.writerow(seq)
	writeFile.close()

def get_dataset(n_in, n_out, cardinality, n_samples,menu_file,input_seq, generate_seq):
	X1, y = list(), list()
	seq = np.zeros(n_in+1,dtype=int)
	df = pd.read_csv(menu_file)
	if generate_seq == 1:
		generate_sequence(n_in+1,n_samples,menu_file,input_seq)
	with open(input_seq, 'r', newline="") as readFile:
		reader = csv.reader(readFile,delimiter =',')
		for read in reader:
			for i in range(n_in+1):
				test = int(read[i])
				for j in range(cardinality-1):
					if (df['selectionId'][j] == test):
						seq[i] = df['index'][j]
			source = seq
			source = np.squeeze(source)
			target = source[n_in]
#			target_in = source[1:n_in+1] 
			source = seq[0:n_in]
			src_encoded = to_categorical(source, num_classes=cardinality)
			tar_encoded = to_categorical(target, num_classes=cardinality)
#			tar2_encoded = to_categorical(target_in, num_classes=cardinality)
		# store
			X1.append(src_encoded)
#			X2.append(tar2_encoded)
			y.append(tar_encoded)
#			print(len(X1))
#			print(X1.shape())
	return np.array(X1), np.array(y)