from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np

# Make sure the path is specific to test 3, 4, 5; relu 3, 4, 5 !!!!!
data_path = "./"

ns=200
bias_array = np.zeros( (ns, 1) )
res_array_mean = np.zeros( (5,2))
nd=200

true_mean_v=[1.000927,1.000792,0.9986607]
#true_mean_v=[5.002603,5.006795,5.002549]
true_mean_v=[2.445598,4.501814,6.032328]

sz=[200,500,1000]


ndv=[50, 100, 200]

for k in sz:   
    data_path2 = "./dat_n" + str(k) + "_m3/"
    true_mean=true_mean_v[sz.index(k)]
    if k<300 or k>600:
        continue
    
    for nd in ndv:
        #if k==1000 and nd!=100:
        #    continue
        print( str(k)+" "+ str(nd) )
        
        fn_res = data_path + "res_3L_relu_n" + str(k) + "_m3_d"+ str(nd) +".txt"
        i = 1
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .1
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .01
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .0001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .00001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
        
        
        np.savetxt(fn_res, res_array_mean)
        
        
        ####4L
        fn_res = data_path + "res_4L_relu_n" + str(k) + "_m3_d"+ str(nd) +".txt"
        i = 1
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .1
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .01
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .0001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        #model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .00001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
        
        
        np.savetxt(fn_res, res_array_mean)
        
        
        ####5L
        fn_res = data_path + "res_5L_relu_n" + str(k) + "_m3_d"+ str(nd) +".txt"
        i = 1
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .1
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .01
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        #np.savetxt(fn_res, bias_array)
        res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .0001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
        
        
        model = Sequential()
        model.add(Dense(nd, input_dim=4, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        learning_rate = .00001
        opt = optimizers.SGD(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        for j in range(ns):
            i = j + 1
            full_path = data_path2 + "dt_" + str(i) + ".csv"
            dat2 = pd.read_csv(full_path)
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
            # Y
            # scaler.inverse_transform(Y_scaled)
            # Scale both the training inputs and outputs
            # scaled_training = scaler.fit_transform(training_data_df)
            # scaled_testing = scaler.transform(test_data_df)
            # Define the model
            # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
            # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            # Train the model
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
            est=(sum(rw*ry)+sum(mw*ry[r_id]))/eN
            bias_array[j,0]=est-(true_mean)
        
        
        bias_array.mean()
        print(np.mean(bias_array,axis=0))
        print(np.std(bias_array,axis=0))
        res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
        res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
        
        
        np.savetxt(fn_res, res_array_mean)
