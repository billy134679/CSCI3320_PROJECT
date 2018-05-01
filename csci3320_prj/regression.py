
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate , cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import seaborn
import os
import csv
#import re
from sklearn.preprocessing import StandardScaler
seaborn.set()


# In[ ]:



def get_data(name = ''):
    df = ""
    path = os.getcwd()
    direct='data'
    file=os.path.join(path,direct, name)
    #opencsv=""
    try:
        opencsv = open(os.path.join(path,direct,name), 'r')
        #print(opencsv)#find the race-result-horse file
    except:
        while opencsv != name:  # if the file cant be found if there is an error
            print("Could not open ", "file")
            opencsv = input("\nPlease try to open file again: ")
    else:
        with open(os.path.join(path,direct,name)) as f:
            feature_names = []#f.readline()
            #print(feature_names)
            reader = csv.reader(f)
            for row in reader:
                feature_names = row
                #print(feature_names)
                break
            df =  pd.read_csv(open(os.path.join(path,direct,name)), header=None, names = feature_names) #, na_values = ['WV-A' , 'WV'])
            df.drop(0, inplace = True)
    return df
    
def get_list(df = [] , col_type = ''):
    list_set = np.empty(0)
    for index, row in df.iterrows():
        #print(row[col_type])
        if not row[col_type] in list_set:
            list_set = np.append(list_set , row[col_type])
    return list_set
    
    
    
def finish_time(df = pd.DataFrame({})):
    for index , row in df.iterrows():
        
        _min , _sec , _fract = row['finish_time'].split(".")
       # print(_min , _sec, _fract)
        time = (float)( int(_min)*60 + int(_sec) + int(_fract) /100 )
        df.loc[index , 'finish_time'] = time
        #print("df finish_time: ",df.loc[index , 'finish_time'])
        #print(type(df.loc[index,'finish_time']) )
        

def svr_(X, y , X_test , ker = 'linear' , C = 1 , epsilon = 0.1):
    svr_model = SVR(kernel = ker , degree=3, gamma = 'auto', coef0=0.0, tol=0.001, C=C,
        epsilon= epsilon, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr_model.fit(X,y)
    #print("cross_val_score:")
    #print(cross_val_score(svr_model , X , y , cv = 10 ,n_jobs = -1 ))
    print("SVR with {0} kernel score: {1}".format(ker , svr_model.score(X,y)) )
    return svr_model.predict(X_test)
    
def GBRT_(X,y , X_test, loss = 'ls' , learning_rate = 0.1 , n_estimators = 100 , max_depth = 3):
    gbrt_model = GradientBoostingRegressor(loss = loss , learning_rate = learning_rate , 
                                          n_estimators = n_estimators , max_depth = max_depth)
    gbrt_model.fit(X,y)
    #print("gbrt cross_val_score:")
    #print(cross_val_score(gbrt_model , X , y , cv = 10 ,n_jobs = -1) )
    print("GBRT score: {0}".format(gbrt_model.score(X,y)) )

    return gbrt_model.predict(X_test)

def evaluation(prediction = [] , label = [] , model = 'SVR' , df = pd.DataFrame({}) , wif = 'with'):
    rmse = math.sqrt(mean_squared_error(label  , prediction) )
    print("\n\nEvaluation of ",model,"model",wif,"normalization")
    print("\nRoot mean square error = %.4f"%rmse)

    race_info = pd.DataFrame({
        'df_id':df.index.values,
        'finish_time':df.loc[:,['finish_time']].values.ravel(),
        'prediction':prediction,
        'unique_horse_id':df.loc[:,['unique_horse_id']].values.ravel(),
        'race_id':df.loc[:,['race_id']].values.ravel(),
        'finishing_position':df.loc[:,['finishing_position']].values.ravel()
    })
    
    #print("Race_info:")
    #print(race_info.shape)
    
    race_list = get_list(race_info , 'race_id')
    #print(race_list.shape)
    
    top1_rank = 0
    top3_rank = 0
    
    predict_top1_arr = pd.DataFrame([] , columns = ['unique_horse_id' , 'top1' , 'top3' , 'actual_rank'])
    count = 0
    
    for i in range(race_list.shape[0]):
        race_for_sort = race_info[race_info["race_id"] == race_list[i]].reset_index(drop = True)
        race_sorted = race_for_sort.sort_values(by = ['prediction'] ).reset_index(drop = True)
        #print(race_sorted)
        
        
        
        predict_top1_arr.loc[count , ['unique_horse_id']] = race_sorted.loc[0,['unique_horse_id']]
        if int(race_sorted.loc[0 , 'finishing_position']) == 1:
            predict_top1_arr.loc[count , ['top1']] = True
        else:
            predict_top1_arr.loc[count , ['top1']] = False
        
        if int(race_sorted.loc[0 , 'finishing_position']) <=3:
            predict_top1_arr.loc[count , ['top3']] = True
        else:
            predict_top1_arr.loc[count , ['top3']] = False
        
        predict_top1_arr.loc[count , ['actual_rank']] = int(race_sorted.loc[0 , ['finishing_position']])        
        
        count = count +1
    
    #print(predict_top1_arr)
    print("Top1_accuracy: " , predict_top1_arr.loc[:,['top1']].values.sum() / predict_top1_arr.shape[0])
    print("Top3_accuracy: " , predict_top1_arr.loc[:,['top3']].values.sum()/predict_top1_arr.shape[0])
    print("Average Rank of all predicted top1 horse: %.4f" %(predict_top1_arr.loc[:,['actual_rank']].values.sum()/predict_top1_arr.shape[0]) )
    
    top1_horse_eval = [['horse_id' , 'Top_1' , 'Top_3' , 'ave_rank']]
    #pd.DataFrame([] , columns = ['unique_horse_id' , 'top1' ,'top3' , 'ave_rank'])
    top1_horse_list = get_list(predict_top1_arr , 'unique_horse_id')
    count = 0
    #print(top1_horse_list.shape)
    #print(top1_horse_eval)
    
    for horse_id in top1_horse_list.flatten():
        #print("first top1 prediction",horse_id)
        win_horse = predict_top1_arr[predict_top1_arr["unique_horse_id"] == horse_id]
        #print(win_horse.shape)
        _append = [  horse_id , float(win_horse.loc[:,['top1']].values.sum() / win_horse.shape[0]), 
                                     float(win_horse.loc[:,['top3']].values.sum() / win_horse.shape[0]),
                                    int(round(win_horse.loc[:,['actual_rank']].values.sum() / win_horse.shape[0] ))   ]
        #print(np.shape(_append) )
        top1_horse_eval.append(_append)
        #top1_horse_eval.loc[count , ['unique_horse_id']] = horse_id
        #top1_horse_eval.loc[count , ['top1']] = float(win_horse.loc[:,['top1']].values.sum() / win_horse.shape[0])
        #top1_horse_eval.loc[count , ['top3']] = float(win_horse.loc[:,['top3']].values.sum() / win_horse.shape[0])
        #top1_horse_eval.loc[count , ['ave_rank']] = int(round(win_horse.loc[:,['actual_rank']].values.sum() / win_horse.shape[0] ))
        #print(np.shape(top1_horse_eval))
        count = count+1
        #if count >3:
            #break
        #print(horse_id)
        #print("P(Top1) =", float(win_horse.loc[:,['top1']].values.sum() / win_horse.shape[0]) )
        #print("P(Top3) =", float(win_horse.loc[:,['top3']].values.sum() / win_horse.shape[0]) )
        #print("Average Rank =", int(round(win_horse.loc[:,['actual_rank']].values.sum() / win_horse.shape[0] )) )
        
    top1_horse_eval = np.array(top1_horse_eval)
    top1_horse_eval = pd.DataFrame(data = top1_horse_eval[1:,] , columns = top1_horse_eval[0 , 0:])
    #print(top1_horse_eval)
    #return top1_horse_eval
    
        
    


def regression():
    df_train = get_data('training.csv') #load data
    df_test = get_data('testing.csv')
    #print(df_train.describe())
    #print("training set: ",df_train.shape[0])
    #print("testing set: ",df_test.shape[0])
    #print(get_list(df_test , 'race_id').shape)
    finish_time(df_train)  #change finish_time column from str to float
    finish_time(df_test)
    
    X_train_src = df_train.loc[:, ['actual_weight' , 'declared_horse_weight' , 'draw' , 'win_odds' , 'jockey_ave_rank'
                                  ,'trainer_ave_rank' , 'recent_ave_rank' , 'race_distance'] ]
    y_train_src = df_train.loc[:, ['finish_time'] ]
    
    X_test_src = df_test.loc[:, ['actual_weight' , 'declared_horse_weight' , 'draw' , 'win_odds' , 'jockey_ave_rank'
                                  ,'trainer_ave_rank' , 'recent_ave_rank' , 'race_distance'] ]
    
    y_test_src = df_test.loc[:, ['finish_time'] ]
    
    
    
    #print(X_train_src.loc[1].values)
    
    X_scaler = StandardScaler()
    
    X_scaler.fit(X_train_src)
    
    X_train_scaled = X_scaler.transform(X_train_src)  # cannot use test data for StandardScaler fitting so use training data instead
    
    X_test_scaled = X_scaler.transform(X_test_src)
    
    X_scaler.fit(y_train_src)
    
    y_train_scaled = X_scaler.transform(y_train_src)
    
    y_test_scaled = X_scaler.transform(y_test_src)
    
    #print(X_train_scaled[0] , X_test_scaled[0])
    
    #svr_(X_train_scaled , y_train_scaled.ravel() , 'linear' , 4 , 0.4 )
    svr_wif_norm_predict = svr_(X_train_scaled , y_train_scaled.ravel() ,X_test_scaled   
                                                     ,'linear' , 2 , 0.2)   #SVR model   linear: C = 2 , 0.2 best
    #X_scaler.fit(y_train_src)
    svr_wif_norm_predict = X_scaler.inverse_transform(svr_wif_norm_predict)
    svr_wifo_norm_predict =svr_(X_train_src.values , y_train_src.values.ravel() , X_test_src.values ,'linear', 2, 0.2)
    #svr_(X_train_scaled , y_train_scaled.ravel() , 'poly' , 2.5 ,0.1)
    #svr_(X_train_scaled , y_train_scaled.ravel() , 'sigmoid' , 2.5 , 0.1)
    #svr_(X_train_scaled , y_train_scaled.ravel() , 'precomputed')
    
    gbrt_wif_norm_predict  = GBRT_(X_train_scaled , y_train_scaled.ravel() ,X_test_scaled,
          'ls' , 0.2 , 200 , 6 )  #GBRT model
    gbrt_wif_norm_predict = X_scaler.inverse_transform(gbrt_wif_norm_predict)
    
    gbrt_wifo_norm_predict = GBRT_(X_train_src.values , y_train_src.values.ravel() , X_test_src.values ,
          'ls' , 0.2 , 200 , 6 )
                     
    evaluation(svr_wif_norm_predict , y_test_src.values.ravel() , "SVR" ,df_test , "with")  #evaluation part
    evaluation(svr_wifo_norm_predict , y_test_src.values.ravel() , "SVR" , df_test , "without")
    evaluation(gbrt_wif_norm_predict , y_test_src.values.ravel() , 'GBRT' , df_test , "with")
    evaluation(gbrt_wifo_norm_predict , y_test_src.values.ravel() , 'GBRT' , df_test, "without")
    ################################## regression model implementation
    
    
    
    

    


# In[ ]:


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    regression()

