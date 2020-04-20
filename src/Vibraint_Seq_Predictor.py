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
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
 
from keras.models import Model
from keras.layers import Input
import Vibraint_Seq_Initializer as gi
from Vibraint_Seq_Initializer import *
import Vibraint_Seq_DataGenerator as dg

#call init function from Initializer program
Predicted_menu = []
Predicted_prob = []

#call the get dataset function from the DataGenerator program
X, y = dg.get_dataset(n_steps_in, n_steps_out, n_features, n_samples,menu_file,input_file,generate_seq)
 
# choose a number of time steps
n_steps = 3

# reshape from [samples, timesteps] into [samples, timesteps, features]

print(X.shape)
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(n_units, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(55))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=20, verbose=0)
model.save('seqmodel.h5')
# demonstrate prediction
x_input,y_test = dg.get_dataset(n_steps_in, n_steps_out, n_features, 1,menu_file,test_file,generate_seq)
x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], n_features))
#print("Test sequence provided:",gi.one_hot_decode1(x_input,n_features))
yhat = model.predict(x_input, verbose=0)

Prob_list= np.argsort(-yhat)[::-1]
df1 = pd.read_csv(menu_file)
for i in range(n_features):
	for j in range(n_features-1):
		if df1['index'][j] == Prob_list[0][i]:
			Predicted_menu.append(int(df1['selectionId'][j]))
print("Next sequence predicted:", Predicted_menu[0])
print("Top 6 most probable menu Items are", Predicted_menu[:num_predict])
