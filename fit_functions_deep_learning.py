from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np

# Make sure the path is specific to test 3, 4, 5; relu 3, 4, 5 !!!!!
data_path = "./"

# Specify number of nodes for each layer
nd=50

print( "Number of Nodes: "+ str(nd) )

# Specify file name to save individual predition value
fn_res = data_path + "res_3L_relu_n500" + "_d"+ str(nd) +".txt"

# Specify input data location
full_path = data_path + "toy_data.csv"
dat2 = pd.read_csv(full_path)

# Deep learning architecture
model = Sequential()
model.add(Dense(nd, input_dim=4, activation='relu'))
model.add(Dense(nd, activation='relu'))
# uncomment below to add more layers
#model.add(Dense(nd, activation='relu'))       
#model.add(Dense(nd, activation='relu'))

model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
# Specify learning rate
learning_rate = .1
opt = optimizers.SGD(learning_rate = learning_rate)

# Compile model
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = StandardScaler()
dat2=dat2.rename(columns={"V1": "Y", "V2": "R", "V3": "X1", "V4": "X2", "V5": "X3", "V6":"X4", "V7": "W"})
X = dat2[["X1","X2","X3","X4"]].values
Y = dat2[['Y']].values
r = dat2[['R']].values
w = dat2[['W']].values.ravel()

ry = dat2.loc[dat2['R'] == 1, 'Y'].values
rx = dat2.loc[dat2['R'] == 1, 'X1':'X4'].values
rw = dat2.loc[dat2['R'] == 1, 'W'].values    

mx = dat2.loc[dat2['R'] == 0, 'X1':'X4'].values
mw = dat2.loc[dat2['R'] == 0, 'W'].values    
   
scaler.fit(ry.reshape(-1,1))
Y_scaled = scaler.transform(ry.reshape(-1,1))

model.fit(
    rx,
    Y_scaled,
    # batch_size = 32,
    validation_split = .25,
    epochs = 200,
    shuffle = True,
    callbacks = [es, mc],
    verbose = 2
)
# Call model
saved_model = load_model('best_model.h5')
# Extract Y hats from saved_model
Yp = saved_model.predict(X)
Yp_unscaled = scaler.inverse_transform(Yp)
resm=Yp_unscaled[r==1]
mesm=Yp_unscaled[r==0]
dm=np.subtract.outer(mesm,resm)
dm=np.absolute(dm)
r_id=dm.argmin(axis=1)
eN=sum(w)

# final estimate
est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
print("VOI estimate: " + str(est))

# Save individaul predictions
np.savetxt(fn_res, Yp_unscaled)



