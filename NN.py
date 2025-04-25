import pandas as pd  
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt

#read the dataset
#using pd because we have and non-numeric data 
data = pd.read_csv('alzheimers_disease_data.csv')
print(data.head())
data.shape

#drop unnecessary columns 
data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
print(data.head())

#split data into input(X) and output(Y) and normalize data
X = data.drop('Diagnosis', axis=1)
X.shape
Y = data['Diagnosis'].copy()
Y.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data to training and testing data 5-Fold
SKFold=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

crossentropy_list = []
mse_list = []
accuracy_list = []
loss_history = []
val_loss_history = []
accuracy_history = []
val_accuracy_history = []
mse_history = []
val_mse_history = []

i=1
for i, (train, test) in enumerate(SKFold.split(X_scaled,Y)):
    # Create model
     model = Sequential()
     model.add(Dense(64, activation="swish", input_shape=(32, ), kernel_regularizer=regularizers.l2(0.01)))
     model.add(Dense(1, activation='sigmoid'))

    #Compile model
     keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, decay=0.0, nesterov=False)
     model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['mean_squared_error', 'accuracy'])

    #Early Stopping 
     early_stop = EarlyStopping(monitor='val_loss', verbose=0, patience=10, restore_best_weights=True)   

    #Fit model 
     history = model.fit(X_scaled[train], Y[train], validation_data=(X_scaled[test],Y[test]), batch_size=32, epochs=100, verbose=1, callbacks=[early_stop])

     accuracy_history.append(history.history['accuracy']) 
     val_accuracy_history.append(history.history['val_accuracy'])
     loss_history.append(history.history['loss'])
     val_loss_history.append(history.history['val_loss'])
     mse_history.append(history.history['mean_squared_error'])
     val_mse_history.append(history.history['val_mean_squared_error'])

    #Evaluate model
     scores = model.evaluate(X_scaled[test], Y[test], verbose=0)
     crossentropy_list.append(scores[0]) 
     mse_list.append(scores[1])
     accuracy_list.append(scores[2])
     print("Fold :", i, "Test loss : ", scores[0], "Test mse :", scores[1], "Test accuracy :", scores[2])


#max_len_loss = max(len(h) for h in loss_history)
loss_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in loss_history])
val_loss_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in val_loss_history])
accuracy_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in accuracy_history])
val_accuracy_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in val_accuracy_history])
mse_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in mse_history])
val_mse_history = np.array([np.pad(h, (0, 100 - len(h)), constant_values=np.nan) for h in val_mse_history])

print("Loss: ", np.nanmean(loss_history))
print("Accuracy: ", np.nanmean(accuracy_history))
print("Mean Squared Error: ", np.nanmean(mse_history))


#Graphs for loss and metrics 

plt.figure()
plt.plot(np.nanmean(loss_history, axis=0), label='Train Loss')
plt.plot(np.nanmean(val_loss_history, axis=0), label='Val Loss')
plt.title('Loss (Cross-Entropy)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.nanmean(accuracy_history, axis=0), label='Train Accuracy')
plt.plot(np.nanmean(val_accuracy_history, axis=0), label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.nanmean(mse_history, axis=0), label='Train MSE')
plt.plot(np.nanmean(val_mse_history, axis=0), label='Val MSE')
plt.title('Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
