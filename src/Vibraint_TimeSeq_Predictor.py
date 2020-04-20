"""*************************************************************************************
VIBRAINT PREDICTION MODULE  - PREDICT NEXT ITEM OF A SEQUENCE
THIS PROGRAM TRAINS OUR MODEL WITH THE SAMPLES GENERATED OR SPECIFIED IN THE CSV FILE
IT SAVES THE TRAINED MODEL IN .h5 FORMAT.
OUTPUT OF THIS PROGRAM ARE MODEL, ENCODER AND DECODER h5 FILES WHICH CAN BE LOADED
ANYTIME AND ANYWHERE IN THE PREDICTOR OR ANY OTHER MODULE.
NOTE: FOR RETRAINING THE MODEL WITH UPDATED MODEL PARAMETERS SUCH AS epochs, 
number of LSTM units OR NUMBER OF SAMPLE INPUTS n_samples EITHER GENERATED OR SPECIFIED
IN THE train_file CSV FILE, INDICATED USING generate_seq VARIABLE
************************************************************************************"""

# univariate bidirectional lstm example
from numpy import array
import json
import csv
import numpy as np
import pandas as pd
from keras.models import load_model
import Vibraint_TimeSeq_Initializer as gi
from Vibraint_TimeSeq_Initializer import *
import Vibraint_Seq_DataGenerator as dg_s
import Vibraint_PM_DataGenerator as dg_t

#call init function from Initializer program
s_Predicted_menu = []
s_Predicted_prob = []
#load sequence predictor
seq_model = load_model('seqmodel.h5')
# demonstrate prediction
x_input,y_test = dg_s.get_dataset(s_n_steps_in, s_n_steps_out, s_n_features, s_test_samples,menu_file,s_test_file,s_generate_seq)
x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], s_n_features))
#print("Test sequence provided:",gi.one_hot_decode1(x_input,n_features))
yhat = seq_model.predict(x_input, verbose=0)
s_Predicted_prob = -np.sort(-yhat)
s_Prob_list= np.argsort(-yhat)[::-1]
df1 = pd.read_csv(menu_file)
for i in range(s_n_features):
	for j in range(s_n_features-1):
		if df1['index'][j] == s_Prob_list[0][i]:
			s_Predicted_menu.append(int(df1['selectionId'][j]))
print("Next sequence predicted:", s_Predicted_menu[0])
print("Top 6 most probable menu Items -Sequence Prediction", s_Predicted_menu[:num_predict])
print("Top 6 most probable menu Items -Sequence Prediction", s_Predicted_prob[0][:num_predict])
#Time prediction
# load the trained Encoder-Decoder models
infenc = load_model('encoder.h5')
infdec = load_model('decoder.h5')

# Predicts output sequence for the test sequence and TOP accuracy is calculated
accuracy1 = np.zeros(accuracy_level, dtype=float)
Xtest1, Xtest2, y = dg_t.get_dataset(t_n_steps_in, t_n_steps_out, t_n_features, t_test_samples, menu_file, t_test_file, t_generate_seq)
with open(out_target, 'w', newline="") as writeFile:
	writer = csv.writer(writeFile,delimiter =',')
	for i in range(t_test_samples):
		Xt1 = np.reshape(Xtest1[i],(1,t_n_steps_in,t_n_features))
		target = gi.predict_sequence(infenc, infdec, Xt1, t_n_steps_out, t_n_features)
		y_list = gi.one_hot_decode(y[i])
		target_list = gi.one_hot_decode1(target,t_n_features)
		for i in range(t_n_steps_in-1):
			for j in range(accuracy_level):
				if y_list[i] == target_list[i][j]:
					accuracy1[j]+=1
		for k in range(1,accuracy_level):
			accuracy1[k] +=accuracy1[k-1]
		writer.writerow(target)
print("Expected Output")
print(y_list)
print("Actual Output")
print(gi.one_hot_decode(target))
accuracy1 = (accuracy1/((t_n_steps_in-1)*(t_test_samples)))*100
print("Top", accuracy_level,"Accuracy")
print(accuracy1)

Sum = np.zeros(t_n_features-2, dtype=float)
t_Predicted_menu = []
t_Predicted_prob = []
with open(out_target, 'r', newline="") as read_file:
	reader = csv.reader(read_file,delimiter =',')
	with open(out_prob, 'w', newline="") as write_file:
		writer = csv.writer(write_file,delimiter =',')
		for read in reader:
			df = pd.DataFrame(read)
			test = df.iloc[prediction_time]
			test = (test[0].split(" "))
			t1= test[0].split("[")
			test[0] = t1[1]
			t2= test[56].split("]")
			test[56] = t1[1]
			final_list = []
			for i in test:
				final_list.append(i.strip())
			writer.writerow(final_list)
			for i in range(t_n_features-2):
				Sum[i] = Sum[i]+float(test[i])
Average1 = Sum/t_test_samples
t_Prob_list= np.argsort(Average1)[::-1][:t_n_features-2]
df1 = pd.read_csv(menu_file)
for i in range(t_n_features-2):
	for j in range(t_n_features-3):
		if df1['index'][j] == t_Prob_list[i]:
			t_Predicted_menu.append(int(df1['selectionId'][j]))
			t_Predicted_prob.append(float(Average1[t_Prob_list[i]]))
print("Top Menu Items - Time Prediction:", t_Predicted_menu[:num_predict])
print("Top Menu Items probabilities are", t_Predicted_prob[:num_predict])
t_modified_Predicted_prob = []
for i in range(t_n_features-3):
    t_modified_Predicted_prob.append(t_Predicted_prob[i]*time_sensitiveness_level)

Predicted_menu = []
Predicted_prob = []
s=0
t=0
for i in range(top_predictions):
    if s_Predicted_prob[0][s] > t_modified_Predicted_prob[t]:
        Predicted_menu.append(s_Predicted_menu[s])
        Predicted_prob.append(s_Predicted_prob[0][s])
        s+=1
    else:
        Predicted_menu.append(t_Predicted_menu[t])
        Predicted_prob.append(t_modified_Predicted_prob[t])
        t+=1

print("Top Menu Items - Combined Prediction:\n", Predicted_menu)
print("Top Menu Items probabilities are - Combined:\n", Predicted_prob)

#outputData = gi.generate_json(Predicted_menu,Predicted_prob)
#with open(output_file, "w") as output1:
#	json.dump(outputData, output1)