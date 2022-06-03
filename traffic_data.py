#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:06:18 2022

@author: aparna
"""

import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler



def get_input_tensor(fos,sensors_list):
    #convert input dataframe to tensor(sensors x (days x time) x features)
    sensors = tf.convert_to_tensor(sensors_list)
    days = fos["Date"].unique()
    #print(sensors,days)
    values_from_vds = []
    for vds in sensors:
        values_per_day = []
        for day in days:
            data_point = fos[(fos['VDS'] == vds) & (fos['Date'] == day)]
            state = data_point[['AggFlow','AggOccupancy','AggSpeed']]
            values_per_day.append(tf.convert_to_tensor(state))
            #print(f"{len(values_per_day)} measurements on day {day}")

        values_from_vds.append(tf.concat(values_per_day,axis =0))
        #print(f"{len(values_from_vds)} measurements from vds {vds}")

    return (tf.stack(values_from_vds)) 


def scaled(D):
    scaler = MinMaxScaler(feature_range =(0,1),copy=True)
    rescaled = []
    for x in D:
        scaler.fit(x)
        rescaled.append(scaler.transform(x))
    return(tf.stack(rescaled))    
    
def preprocess_input(input_tensor,w=12):
        #normalize each sensor
        #X_norm, norm = tf.linalg.normalize(input_tensor,ord='euclidean', axis = 1) #(83 x 3456 x 3)
        #print("normalized input dimension:",X_norm.shape)

        #generate subsequences,time series of length w for each time stamp
        X2 = tf.transpose(input_tensor,(1,0,2)) #(3456, 83, 3)
        print("time series from:",X2.shape) 
        X_series_gen= tf.keras.preprocessing.timeseries_dataset_from_array(X2,None,sequence_length=w,sequence_stride=1,batch_size=288)
        TS =[]
        for batch in X_series_gen:
            batch = tf.transpose(batch,perm =(2,0,1,3))  # (288, 12, 26, 3) ---> (26, 288 , 12, 3)
            #print('batch shape ',batch.shape)
            # substract each value from first value in a window
            batches = []
            for s in range(0,batch.shape[0]):
                for t  in range(0,batch.shape[1]):
                    #print(batch[s][t].shape)
                    batch_sub = tf.subtract(batch[s][t][0] , batch[s][t])
                    batches.append(batch_sub)
            batches = tf.convert_to_tensor(batches)
            print("batch shape",batch.shape,"-->",batches.shape) #(23904, 12, 3)
            #TS.append(tf.reshape(batches,batch.shape))
            TS.append(batches)

        print("no. of time series :",len(TS))
        return(tf.concat(TS,axis =0))


if __name__=="__main__":
    #read data

    fos_jan =  pd.read_csv("data/fos_jan.csv",header=0)
    fos_feb =  pd.read_csv("data/fos_feb.csv",header=0)
    sensors_26 = pd.read_csv("data/26_sensors.csv",header = None)
    
    fos_jan.dropna(subset=['Date'],inplace=True)
    fos_feb.dropna(subset=['Date'],inplace=True)
    
    #get sensors,days,timestamp list 
    days1 =tf.convert_to_tensor (fos_jan['Date'].unique(),dtype=tf.int32)
    days2 = tf.convert_to_tensor (fos_feb['Date'].unique(),dtype=tf.int32)
    days = tf.concat([days1,days2],axis = 0)
    #
    
    time= tf.convert_to_tensor (fos_jan['Time'].unique())
    #t = time.shape[0]
    
    VDS = tf.convert_to_tensor (fos_jan['VDS'].unique())
    #s = VDS.shape[0]
    
    D1 = get_input_tensor(fos_jan,sensors_26)
    D2 = get_input_tensor(fos_feb,sensors_26)
    D = tf.concat([D1,D2],axis =1)
    
    X1 = scaled(D)
    Xp = preprocess_input(X1,w=12)
    
    X =tf.reshape(Xp,(26,-1,12,3))

    #save
    np.save("data/input_tensor.npy", X, allow_pickle=False)
    np.save("data/input_data.npy", D, allow_pickle=False)

    #print
    #print(fos_jan.head())
    #print(fos_feb.head())
    #print(sensors_26.head())
    #print(days)
    #print(f"jan = {D1.shape},feb = {D2.shape}")
    print("raw input D",D.shape)
    print("scaled input X1", X1.shape)
    print("preprocessed input Xp", Xp.shape)
    print("reshaped input X", X.shape)
    
    
    
