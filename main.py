from keras.models import Model
from keras.layers import (Convolution1D, Activation, Input, MaxPooling1D, Flatten, Dense, Dropout)
from keras.utils import plot_model
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint, Callback, ProgbarLogger)
from keras import backend as K
import sys
import data_handling

args = sys.argv[1:]
if len(args) != 2:
	print("Please run with the command: python3 -W ignore main.py dataset_train.tsv dataset_test.tsv")
	exit()
train_file = args[0]
test_file = args[1]

filename = train_file
seq_len = 1014 # Fixed length of a sequence of chars, given
num_classes = 14 # Num of categories/concepts, given
init_step_size = 0.01 # Given
max_epochs = 33 # Num of epochs training happens for - arbitarily set to 33 to observe step size decay
mini_batch_size = 1 # Given value is 128, but I've set to 1 to run quickly on toy data 
momentum = 0.9 # Given
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" #alphabet set, given
alph_size = len(alphabet)
step_size = init_step_size
data = data_handling.read_data(filename,alphabet,seq_len,num_classes) 
x = data[0] # Training input character sequences 
y = data[1] # Training input labels

#Function to implement step size decay (halves every 3 epochs, 10 times)
def step_size_decay(epoch):
	if epoch>1 and epoch <= 30 and epoch%3==1:  
		global step_size 
		step_size = step_size/2
	return step_size

#Function to print epoch count, loss and step size (to observe decay) after every epoch
class FlushCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		optimizer = self.model.optimizer
		print('Epoch %s: loss %s' % (epoch, logs.get('loss')))
		print("Step size:",K.eval(optimizer.lr)) 

#--------------- ConvNet STRUCTURE ------------------#

#Input one-hot encoded sequence of chars
sequence_one_hot = Input(shape=(seq_len,alph_size),dtype='float32')

# 1st CNN layer with max-pooling
conv1 = Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(sequence_one_hot)
pool1 = MaxPooling1D(pool_size=3)(conv1)

# 2nd CNN layer with max-pooling
conv2 = Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=3)(conv2)

# 3rd CNN layer without max-pooling
conv3 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool2)

# 4th CNN layer without max-pooling
conv4 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv3)

# 5th CNN layer without max-pooling
conv5 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv4)

# 6th CNN layer with max-pooling
conv6 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv5)
pool6 = MaxPooling1D(pool_size=3)(conv6)

# Reshaping to 1D array for further layers
flat = Flatten()(pool6) 

# 1st fully connected layer with dropout
dense1 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(flat)
dropout1 = Dropout(0.5)(dense1)

# 2nd fully connected layer with dropout
dense2 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)

# 3rd fully connected layer with softmax outputs
dense3 = Dense(14, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='softmax')(dropout2)

# SGD: the learning rate set here doesn't matter because it gets overridden by step_size_callback
sgd = SGD(lr=init_step_size, momentum=momentum)

model = Model(input= sequence_one_hot, output=dense3)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Generates .png image with the model structure
plot_model(model, show_shapes = True, to_file='model_structure_new.png')

step_size_callback = LearningRateScheduler(step_size_decay)

# Callbacks to save and retreive the best weight configurations found during training phase
all_callbacks = [step_size_callback, FlushCallback(),
						   ModelCheckpoint('model' + '.hdf5',
										   save_best_only=True,
										   verbose=1),
						   ProgbarLogger(count_mode='samples')]


#--------------- TRAINING ------------------#
# No vaidation split was given in the paper, so I've set it to 0.2 arbitarily
hist = model.fit(x,y,
				batch_size=mini_batch_size,
				nb_epoch=max_epochs,
				verbose=2,
				validation_split=0.2,
				callbacks=all_callbacks)

# Prints to console, the summary of losses etc through all the epochs 
ch = input("\nPrint training loss and error history? (y/n)")
if ch == 'y':
	print(hist.history)

print("\nTraining completed. Beginning testing.\n")


#--------------- TESTING ------------------#

filename = test_file
data = data_handling.read_data(filename,alphabet,seq_len,num_classes)
x_test = data[0]  # Testing input character sequences
y_test= data[1]	 # Testing input labels 

# Testing 
evl= model.evaluate(x_test, y_test, batch_size=1, verbose=2)

# Prints to console, the final loss and final accuracy averaged across all test samples
print(model.metrics_names[0],":",evl[0])
print(model.metrics_names[1],":",evl[1])
