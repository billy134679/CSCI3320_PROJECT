
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
import csv
#import re


# In[10]:


def get_list(df = pd.DataFrame({}) , origin_shape = 0 , ID_type = '' ):
    id_set = np.array(df.loc[1 , ID_type])
    for index, row in df.loc[1:origin_shape].iterrows():
                
        if not row[ID_type] in id_set:
            id_set = np.append(id_set , row[ID_type])
                
                #if index > 20:
                    #break
    #print(id_set.shape[0])
    return id_set

def assign_uni_id(df = pd.DataFrame({}) , id_set= [], ID_type = '', uni_ID_type = '' ):
    
    for index in range(id_set.shape[0]):
                
        get_pos = df[df[ID_type] == id_set[index]]
        get_race =  get_pos.index.values.tolist()
        #print(id_set[index],"result: " , get_race)
                
        uni_id = index+1        
        for race_index in range(len(get_race)):
            df.loc[df.index == get_race[race_index] , (uni_ID_type)] = uni_id
    
    #print("DataFrame empty id:" , len(df[df[uni_ID_type] == ""].index.values.tolist()) )
    
    #return df
def assign_avg_rank(df = pd.DataFrame({}) , testing_set = pd.DataFrame({}) ,id_set= [] , ID_type= '', assign_type =''):
    for index in range(id_set.shape[0]):
        get_pos = df[df[ID_type] == id_set[index]]
        testing_get_pos = testing_set[testing_set[ID_type] == id_set[index]]
        get_race =  get_pos.loc[:,('finishing_position')].index.values.tolist()
        #print(get_race)
        
        if len(get_pos.loc[get_race[0:len(get_race)],('finishing_position')].tolist()) > 0:
            len_rank = len(get_pos.loc[get_race[0:len(get_race)],('finishing_position')].tolist())
            recent_sum_rank = sum(map(int , 
                                              get_pos.loc[get_race[0:len(get_race)],('finishing_position')].tolist() ) )
            rank = int(round(recent_sum_rank / len_rank))
            #print(int(round(recent_sum_rank / len_rank)))
            df.loc[get_pos.index.values.tolist(), (assign_type)] =  rank
            testing_set.loc[testing_get_pos.index.values.tolist() , (assign_type)] = rank
        else:
            #print(int(7))
            df.loc[get_pos.index.values.tolist(), (assign_type)] =  int(7)
            testing_set.loc[testing_get_pos.index.values.tolist() , (assign_type)] = int(7)
            
    
    get_pos = testing_set[testing_set[assign_type] == ""]
    testing_set.loc[get_pos.index.values.tolist() , (assign_type)] = int(7)
    
            
            
        
        
                    
def recent_6_result(df = [] , horse_id_set = [] ):
    for index in range(horse_id_set.shape[0]):
        get_pos = df[df["horse_id"] == horse_id_set[index]]
        get_race =  get_pos.loc[:,('finishing_position')].index.values.tolist()
        #print(horse_id_set[index],"result: " , get_race)
                
                
        for race_index in range(len(get_race)):
            pass_6_result_str = ""
            if race_index > 0 and race_index <6:
                #print("/".join(map(str,get_pos.loc[get_race[0:race_index],('finishing_position')].tolist()) ) )
                pass_6_result_str = "/".join(map(str,get_pos.loc[get_race[0:race_index],('finishing_position')].tolist()) )
                len_rank = len(get_pos.loc[get_race[0:race_index],('finishing_position')].tolist())
                recent_avg_rank = sum(map(int , 
                                            get_pos.loc[get_race[0:race_index],('finishing_position')].tolist() ) )
                #print(int(recent_avg_rank / len_rank)  )
                                                               
                df.loc[df.index == get_race[race_index] , 'recent_6_runs'] = pass_6_result_str
                df.loc[df.index == get_race[race_index] , 'recent_ave_rank'] = int(round(recent_avg_rank / len_rank))
                #df.update(pd.Series([pass_6_result_str], name='recent_6_runs', index=[get_race[race_index]]))
            elif race_index > 6:
                #print("/".join(map( str , get_pos.loc[get_race[race_index-6:race_index],('finishing_position')].tolist()) ) )
                pass_6_result_str = "/".join(map( str , 
                                                    get_pos.loc[get_race[race_index-7:race_index-1],('finishing_position')].tolist()))
                len_rank = len(get_pos.loc[get_race[race_index-6:race_index],('finishing_position')].tolist())
                recent_sum_rank = sum(map( int , 
                                                    get_pos.loc[get_race[race_index-7:race_index-1],('finishing_position')].tolist()) )
                #print(int(recent_avg_rank / len_rank)  )
                df.loc[df.index == get_race[race_index] , 'recent_6_runs'] = pass_6_result_str
                df.loc[df.index == get_race[race_index] , 'recent_ave_rank'] = int(round(recent_avg_rank / len_rank))
                #df.update(pd.Series(["12345"], name='recent_6_runs', index=[get_race[race_index]]))
            else:
                    #print(int(7))
                df.loc[df.index == get_race[race_index] , 'recent_ave_rank'] = int(7)
        #print("ID ",df.loc[df.index == index , 'horse_id'].values,"Index ",get_race[race_index],"Rank", df.loc[df.index == index , 'recent_avg_rank'].values , "   Recent6" 
                    #, df.loc[df.index == index , 'recent_6_runs'].values)

def get_training_or_testing(df = pd.DataFrame({}) , train_or_test = 'training'):
    separation = 0
    for index,row in df.iterrows():
        race_id = str(row['race_id'])
        get_number_str = list(race_id.split("-"))
        number_set = [int(x) for x in get_number_str]
        #print(race_id,number_set)
        if number_set[0] >=2016 and number_set[1] >=328:
            separation = index
            #print(separation)
            break
    target_frame = pd.DataFrame({}) 
    if train_or_test == 'training':
        target_frame = df[df.index < separation]
    elif train_or_test == 'testing':
        target_frame = df.loc[df.index >=separation ]
        #print(target_frame.shape[0])
    
    return target_frame
    #print(train_or_test)

def assign_distance(df = [] , race_df = []):
    #print(df["race_id"])
    #df["race_distance"] = ""
    for index , row in race_df.iterrows():
        race_id = row['race_id']
        #print(type(race_id))
        #print(df["race_id"])
        #print(df[df["race_id"] == race_id].index.values.tolist() )
        df.loc[ df[df["race_id"] == race_id].index.values.tolist() , ('race_distance')] = row['race_distance']
        #print(df.loc[ df[df["race_id"] == race_id].index.values.tolist() , ('race_distance')])
        #if index > 2:
            #break
    #print(df.loc[: , ('race_distance')])
    

def preprocess():
    direct = 'data'
    file = os.path.join(os.getcwd(),direct,'race-result-horse.csv')
    df = ""
    training_set = ""
    testing_set = ""
    try:
        opencsv = open(file , 'r').read() #find the race-result-horse file
    except:
        while opencsv != "race-result-horse.csv":  # if the file cant be found if there is an error
            print("Could not open", opencsv, "file")
            opencsv = input("\nPlease try to open file again: ")
    else:
        with open(file) as f:
            feature_names = []#f.readline()
            #print(feature_names)
            reader = csv.reader(f)
            for row in reader:
                feature_names = row
                #print(feature_names)
                break
            df =  pd.read_csv(file, header=None, names = feature_names) #, na_values = ['WV-A' , 'WV'])
            #print('Before:df.shape %d'%df.shape[0])
            
            origin_shape = df.shape[0]
            
            for index,row in df.iterrows():
                
                if (str(row['finishing_position']).isdigit() ==False ) and index !=0:
                    df.loc[df.index == index , 'finishing_position'] = '?'
                    
                    
            df["finishing_position"].fillna('?' , inplace = True);
            #print(df[df["finishing_position"] == '?'].index.values.shape[0])
            df.drop(df[df["finishing_position"] == '?'].index , inplace = True)
            df.drop(0 , inplace = True)
            #print('After:df.shape %d'%df.shape[0])
            
            
            
            test = df.loc[0:0, ['finishing_position', 'horse_id']]
            #print(test)
            
            
            
            
            ### find all horse id and assign unique id to all horses,jockeys and trainers
            df["recent_6_runs"] = ""
            df["recent_ave_rank"] = ""
            df["unique_horse_id"] = ""
            df["jockey_id"] = ""
            df["jockey_ave_rank"] = ""
            df["trainer_id"] = "" 
            df["trainer_ave_rank"]=""
            df["race_distance"] =""
            
            print("Getting id set of horse ,jockey ,trainer ...")
            
            horse_id_set = get_list(df , origin_shape ,'horse_id')
            
            jockey_id_set = get_list(df ,origin_shape ,'jockey')
            
            trainer_id_set = get_list(df ,origin_shape ,'trainer')
            
            #print(jockey_id_set)
            
            
            
            print("Number of horses:",horse_id_set.shape[0] , " Number of jockeys" , jockey_id_set.shape[0] , " Number of trainers", trainer_id_set.shape[0] )
            
            print("\nAssigning id for horse , jockey , trainer...")
            assign_uni_id(df , horse_id_set , 'horse_id' , 'unique_horse_id')
            assign_uni_id(df , jockey_id_set , 'jockey' , 'jockey_id')
            assign_uni_id(df , trainer_id_set , 'trainer' , 'trainer_id')
            
            #print(df.loc[1 , ('unique_horse_id' , 'jockey_id' , 'trainer_id')])
            
            ############ update pass 6 result for each horse
            #a = df[df["horse_id"] == horse_id_set[0]].index.values.tolist()
            #df.loc[df.index == a[0] , 'recent_6_runs'] = "12345"
            #print(df.loc[0:1,('recent_6_runs')])
            
            #print(df[df.index == 17204])
            print("Preprocessing Start ...")
            
            recent_6_result(df , horse_id_set )
            
            print("Preparing training and testing set ...") 
            training_set = get_training_or_testing(df , 'training')
            testing_set = get_training_or_testing(df , 'testing')
            print("\nAssigning rank and distance for each data ...")
            assign_avg_rank(training_set , testing_set ,trainer_id_set , 'trainer' , 'trainer_ave_rank')
            assign_avg_rank(training_set , testing_set ,jockey_id_set , 'jockey' , 'jockey_ave_rank')
            
            #race_count = 0
            
            #print(race_count)
            #print(training_set.loc[df.index == 17204 ])#, ('recent_6_runs','recent_avg_rank')])
            
            
    #df.to_csv(os.path.join(direct,'test.csv') , encoding = "utf-8" )
    
    
    
    
    
            
    ################################# open race-result-race.csv and get the distance result
    file_path=os.path.join(os.getcwd(),direct,'race-result-race.csv')
    race_df = "" #race-result-race dataframe
    try:
        opencsv = open(file_path , 'r') #find the race-result-race file
    except:
        while opencsv != "race-result-race.csv":  # if the file cant be found if there is an error
            print("Could not open ", "file")
            opencsv = input("\nPlease try to open file again: ")
    else:
        with open(file_path) as f:
            feature_names = []#f.readline()
            reader = csv.reader(f)
            for row in reader:
                feature_names = row
                #print(feature_names)
                break
            race_df =  pd.read_csv(file_path, header=None, names = feature_names) #, na_values = ['WV-A' , 'WV'])
            #print('Before:df.shape %d'%race_df.shape[0])
            #print(reader)
            assign_distance( training_set , race_df)
            assign_distance( testing_set , race_df)
      
    training_set.to_csv(os.path.join(direct,'training.csv') , encoding = "utf-8" )
    testing_set.to_csv(os.path.join(direct,'testing.csv') , encoding = "utf-8" )
    print("End of program")


# In[11]:


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    preprocess()

