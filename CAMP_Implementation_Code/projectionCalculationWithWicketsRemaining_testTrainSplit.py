# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:47:28 2020

@author: naima
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




import numpy as np
import pandas as pd
import urllib.request
import json
import csv
import os
from bs4 import BeautifulSoup
import sys
import seaborn as sns
import math

from datetime import date

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch


from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold # import KFold


from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import linear_model


from sklearn import preprocessing





columnsRemoveInng1=['matchId', 'year','team1', 'team2', 'venueCountry','homeTeam',
       'toss_winner_team', 'inning1Runs', 'inning1RunRate', 'innings1Wickets',
       'inning2Runs', 'inning2RunRate', 'inning2RunsExpected',
       'innings2Wickets', 'team2Cluster', 'team1Cluster', 'Continent','tossWinner','yearId'
       'tossWinner', 'yearId', 'bats1CurrOvrRuns', 'bats1CurrOvrBalls',
       'bats1TotalRuns', 'bats1TotalBalls', 
       'bats2CurrOvrRuns', 'bats2CurrOvrBalls', 'bats2TotalRuns',
       'bats2TotalBalls',
       'totalRunsScoredtillNow','bats1TotalBallsBeforeThisOvr','bats1TotalRunsBeforeThisOvr','bats2TotalBallsBeforeThisOvr','bats2TotalRunsBeforeThisOvr','totalRunsScoredCurrentOvr', 'inning1ProjectedScore'
                  , 'wicket',
       'wicket.1', 'wicket.2', 'wicket.3', 'wicket.4', 'wicket.5' ,'teamOpp','inning2ProjectedScoreSq',
                    
                    'onCreasebowlerOfTheOvr','onCreasebatsman1','onCreasebatsman2'
                 , 'overId','RemainingWickets','venueCountryCode'
     
                   ] #'totalRunsScoredBeforeThisOvr', 'onCreasebowlerOfTheOvr','overId', 'venueCountryCode',  'onCreasebatsman1', 'onCreasebatsman2',  , 'totalRunsScoredBeforeThisOvr'



columnsRemoveInng2=['matchId', 'year','team1', 'team2', 'venueCountry','homeTeam',
       'toss_winner_team',  'inning1RunRate',
       'inning2Runs', 'inning2RunRate', 'inning2RunsExpected',
       'innings2Wickets', 'team2Cluster', 'team1Cluster', 'Continent','tossWinner','yearId'
       'tossWinner', 'yearId', 'bats1CurrOvrRuns', 'bats1CurrOvrBalls',
       'bats1TotalRuns', 'bats1TotalBalls', 
       'bats2CurrOvrRuns', 'bats2CurrOvrBalls', 'bats2TotalRuns',
       'bats2TotalBalls',
       'totalRunsScoredtillNow','bats1TotalRunsBeforeThisOvr',
     'bats2TotalBallsBeforeThisOvr','bats2TotalRunsBeforeThisOvr','totalRunsScoredCurrentOvr', 'inning2ProjectedScore'
         , 'wicket','wicket.1', 'wicket.2', 'wicket.3', 'wicket.4', 'wicket.5' ,'teamOpp','inning2ProjectedScoreSq',
                    
                         'bats1TotalBallsBeforeThisOvr',
                    'onCreasebowlerOfTheOvr','onCreasebatsman1','onCreasebatsman2'
                , 'overId', 'innings1Wickets','venueCountryCode','inning1Runs' ,'RemainingWickets'     
                   ] # 'totalRunsScoredBeforeThisOvr', 'RemainingWickets' ,'remainingRuns', 'onCreasebowlerOfTheOvr', 'overId','venueCountryCode',  'onCreasebatsman1', 'onCreasebatsman2',  , 'totalRunsScoredBeforeThisOvr'










import random
matchinProgressDataFolder = 'C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/'
#matchinProgressDataFolder = str(matchinProgressDataFolder)+'innings2/'


#### concatenate all over records ###########

def CombinedInningsOvers(inngs):
    inningsInProgressOversCombined =pd.DataFrame()
    for inningsOver in range(1,51):
        df2 =  pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inngs)+'/inprogressOfInning'+str(inngs)+'_over'+str(inningsOver)+'Records_withAllEncodedFeatures.csv')  
        dfNew  = df2[(df2['matchId']!=914207) & (df2['matchId']!=1140381)].copy()
        #if currMatch!=914207 and currMatch!=1140381:
        inningsInProgressOversCombined  =pd.concat([inningsInProgressOversCombined, dfNew], sort=True)

    inningsInProgressOversCombined.to_csv(str(matchinProgressDataFolder)+'inprogressOfInning'+str(inngs)+'_over1_50Records_withAllEncodedFeatures_Combined.csv',index=False, header=True)          
    
    return inningsInProgressOversCombined





#########    parameters settings     ################

kfolds= 10
stdTick=2
upperLimitRuns = 400
lowerLimitRuns = 165
constraintvar  ='inning1Runs'
inngs=1


k_neighbors=50

#targetVar =  'inning2ProjectedScore'   #  'inning1ProjectedScore'   #    'totalRunsScoredCurrentOvr' #   'inning1Runs'  #    'inning2RunsExpected' #       

dataConstraint = 'g'+str(lowerLimitRuns)+'_l'+str(upperLimitRuns)

matchinProgressDataFolderResult = str(matchinProgressDataFolder)+'innings'+str(inngs)+'/Results/'

#overSplit=False


oversData=[]
MAE_Records=[]
MAE_allOvers=[]
predData=[]

######### Parameter settings end here     ############



##########     Results output file writing starts here                   ##############

#fileNameOut = str(matchinProgressDataFolderResult)+"inprogressOfInning"+str(inngs)+"_50oversPrediction_LR_dataConstraintVar_"+str(constraintvar)+"_LR_data"+str(dataConstraint)+".csv"

# fileWriteOutput =  open(fileNameOut, "w") #str(matchinProgressDataFolderResult)+"inprogressOfInning1_50oversPrediction_LR"+str(dataConstraint)+".csv",'w')
# fileWriteOutput.write('overId,kFold, dataConstraint,targetVariable,MAE, RMSE \n')
# fileWriteOutput.close() 
#########   Output file header writing ends here        ##############


#inningsOver= 1






inningStartFeatureVector = pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inngs)+'/inprogressOfInning'+str(inngs)+'_over1Records_playerClustCount.csv')
                                       

print('Before', len(inningStartFeatureVector))
print('min1', inningStartFeatureVector['inning1Runs'].min())
print('max1', inningStartFeatureVector['inning1Runs'].max())
print('std1', inningStartFeatureVector['inning1Runs'].std())
print('Avg1', inningStartFeatureVector['inning1Runs'].mean())  


print('Before', len(inningStartFeatureVector))
print('min2', inningStartFeatureVector['inning2Runs'].min())
print('max2', inningStartFeatureVector['inning2Runs'].max())
print('std2', inningStartFeatureVector['inning2Runs'].std())
print('Avg2', inningStartFeatureVector['inning2Runs'].mean())  



inningFeatureVectorOver1 = inningStartFeatureVector[(inningStartFeatureVector['team1']!='Zimbabwe') & (inningStartFeatureVector['team2']!='Zimbabwe') & (inningStartFeatureVector['team1']!='Bangladesh') & (inningStartFeatureVector['team2']!='Bangladesh')] 


stdev1 = int(inningFeatureVectorOver1['inning1Runs'].std())
stdev2 = int(inningFeatureVectorOver1['inning2Runs'].std())
mean1 = int(inningFeatureVectorOver1['inning1Runs'].mean())
mean2 = int(inningFeatureVectorOver1['inning2Runs'].mean())



inningFeatureVectorOverIn = inningFeatureVectorOver1[(inningFeatureVectorOver1['inning1Runs']>mean1 -(stdTick*stdev1)) & (inningFeatureVectorOver1['inning1Runs']<mean1+(stdTick*stdev1))].copy()
inning1FeatureVectorCompleteNew = inningFeatureVectorOverIn[(inningFeatureVectorOverIn['inning2Runs']>mean2 -(stdTick*stdev2)) & (inningFeatureVectorOverIn['inning2Runs']<mean2+(stdTick*stdev2))].copy()
#inning1FeatureVectorCompleteNew.to_csv(str(matchinProgressDataFolder)+'innings'+str(inngs)+'/filteredMatches_'+str(inngs)+'_All.csv')


print('After', len(inning1FeatureVectorCompleteNew))
print('min1', inning1FeatureVectorCompleteNew['inning1Runs'].min())
print('max1', inning1FeatureVectorCompleteNew['inning1Runs'].max())
print('std1', inning1FeatureVectorCompleteNew['inning1Runs'].std())
print('Avg1', inning1FeatureVectorCompleteNew['inning1Runs'].mean())  

print('min2', inning1FeatureVectorCompleteNew['inning2Runs'].min())
print('max2', inning1FeatureVectorCompleteNew['inning2Runs'].max())
print('std2', inning1FeatureVectorCompleteNew['inning2Runs'].std())
print('Avg2', inning1FeatureVectorCompleteNew['inning2Runs'].mean())  






matchIdsSelectData = inning1FeatureVectorCompleteNew['matchId'].unique()


# for i in range(len(matchIdsSelectData)):
#     print(matchIdsSelectData[i])
print(len(matchIdsSelectData))


MAEDataForallOvers=[]
absErrorMeanForallOvers=[]
absErrorDevForallOvers=[]
absErrorPercentForallOvers=[]
def predictRunsForMatch(currMatchId,inningsId,overSplit):
    j=0
    k=-1
    notFound=1
    
    targetVar = 'inning'+str(inningsId)+'ProjectedScore'   # 'inning'+str(inningsId)+'Runs' #
    
    for i in range(1):
        k=k+1
        inngs = inningsId

        
        randomNum = random.randint(0,len(matchIdsSelectData))
        notFound=1
        
        currMatch = currMatchId #matchIdsSelectData[randomNum]   # # # 66276 #902647 #   
       
        #inningsInProgressOversCombined = CombinedInningsOvers(1)
        if overSplit==True and ((currMatchId not in matchIdsSelectData)):# or len(inningsInProgressOversCombined[inningsInProgressOversCombined['matchId']==currMatch])<40):
            print('Match Id not found in filtered Matches')
            notFound=0
            return notFound,0
            #break
        elif overSplit==True and (currMatchId in matchIdsSelectData):
            #overSplit=True
            inningsInProgressOversCombined = CombinedInningsOvers(inngs)
            predData=[]
            withoutWicketLossPred=[]
        if(1==1):
            inningsInProgressOversCombined = CombinedInningsOvers(inngs)

            inningsInProgressOversCombined =  pd.read_csv(str(matchinProgressDataFolder)+'inprogressOfInning'+str(inngs)+'_over1_50Records_withAllEncodedFeatures_Combined.csv')          
            
            #inningsInProgressOversCombined = inningsInProgressOversCombined[inningsInProgressOversCombined['year']<=2005].copy()
            #CombinedInningsOvers(inngs)
            print('match length', len(inningsInProgressOversCombined[inningsInProgressOversCombined['matchId']==currMatch]))
            
            #inningsInProgressOversCombined = CombinedInningsOvers(inngs)
            
            
#            for inningsOver in range(1,len(inningsInProgressOversCombined[inningsInProgressOversCombined['matchId']==currMatch])+1):
#                maxVal = inningsInProgressOversCombined['overId'].max()
#                minVal = inningsInProgressOversCombined['overId'].min()
#                inningsInProgressOversCombined['overId'] = round((inningsInProgressOversCombined['overId']- minVal)/ (maxVal- minVal),2)        
#    
#           
            
            matchLength = len(inningsInProgressOversCombined[inningsInProgressOversCombined['matchId']==currMatch])+1
            
            previousOvrWickets = 0
            currentOvrWickets  = 0
            for inningsOver in range(1,len(inningsInProgressOversCombined[inningsInProgressOversCombined['matchId']==currMatch])+1):
                
                
                #oversData.append(inningsOver)
                #print('overProcessing', inningsOver)
                predOver = inningsOver
                #inningsInProgressOversCombined #
                #print("Current Over",inningsOver)

                inningsInProgressOversCombined['inning2ProjectedScoreSq'] =0 
                #inningsInProgressOversCombined['inning2ProjectedScoreSq'] = inningsInProgressOversCombined['inning2ProjectedScore']
                
                #inningsInProgressOversCombined['inning2ProjectedScoreSq'] = inningsInProgressOversCombined['inning2ProjectedScoreSq']**0.5
                runsConvert =1 #2
               
                inning1FeatureVectorComplete1 = inningsInProgressOversCombined[inningsInProgressOversCombined['overId']==inningsOver].copy() # pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inngs)+'/inprogressOfInning'+str(inngs)+'_over'+str(inningsOver)+'Records_withAllEncodedFeatures.csv')  
                
                inning1FeatureVectorCompleteNew = inning1FeatureVectorComplete1[(inning1FeatureVectorComplete1['team1']!='Zimbabwe') & (inning1FeatureVectorComplete1['team2']!='Zimbabwe') & (inning1FeatureVectorComplete1['team1']!='Bangladesh') & (inning1FeatureVectorComplete1['team2']!='Bangladesh')] 



                stdev1 = int(inning1FeatureVectorCompleteNew['inning1Runs'].std())
                stdev2 = int(inning1FeatureVectorCompleteNew['inning2Runs'].std())
                mean1 = int(inning1FeatureVectorCompleteNew['inning1Runs'].mean())
                mean2 = int(inning1FeatureVectorCompleteNew['inning2Runs'].mean())


                inning1FeatureVectorCompleteIn = inning1FeatureVectorCompleteNew[(inning1FeatureVectorCompleteNew['inning1Runs']>mean1 -(stdTick*stdev1)) & (inning1FeatureVectorCompleteNew['inning1Runs']<mean1+(stdTick*stdev1))].copy()
                inning1FeatureVectorComplete = inning1FeatureVectorCompleteIn[(inning1FeatureVectorCompleteIn['inning2Runs']>mean2 -(stdTick*stdev2)) & (inning1FeatureVectorCompleteIn['inning2Runs']<mean2+(stdTick*stdev2))].copy()
                
                #inning1FeatureVectorComplete = inning1FeatureVectorCompleteNew.copy()
                
                if inngs==2:
                    inning1FeatureVectorComplete['remainingRuns'] = 0
                    inning1FeatureVectorComplete['remainingRuns'] = inning1FeatureVectorComplete['inning1Runs'] - inning1FeatureVectorComplete['totalRunsScoredBeforeThisOvr']
                   
                    #print(inning1FeatureVectorComplete['remainingRuns'].head())
                
                
                #inning1FeatureVectorComplete['teamOpp']= inning1FeatureVectorComplete['team1'].astype(str) +" "+inning1FeatureVectorComplete['team2']
                
                #print(inning1FeatureVectorComplete.columns)
                
                
#                 labels = inning1FeatureVectorComplete['teamOpp'].astype('category').cat.categories.tolist()
#                 replace_continent_comp = {'teamOpp' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#                 inning1FeatureVectorComplete .replace(replace_continent_comp, inplace=True)
#                 

                labels = inning1FeatureVectorComplete['team1'].astype('category').cat.categories.tolist()
                replace_continent_comp = {'team1' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
                inning1FeatureVectorComplete .replace(replace_continent_comp, inplace=True)
#                 
                
                
                labels = inning1FeatureVectorComplete['team2'].astype('category').cat.categories.tolist()
                replace_continent_comp = {'team2' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
                inning1FeatureVectorComplete .replace(replace_continent_comp, inplace=True)
                
                
                #print(inning1FeatureVectorComplete['teamOpp'].unique())
                #print(len(inning1FeatureVectorComplete['teamOpp'].unique()))
                #break
#                 
                
                #print('Before', len(inning1FeatureVectorComplete))

                ##############  categorical variable to one-hot encoding ##########################   
                inning1FeatureVectorComplete['team1Cluster'] = pd.Categorical(inning1FeatureVectorComplete['team1Cluster'])
                inning1FeatureVectorComplete['team2Cluster'] = pd.Categorical(inning1FeatureVectorComplete['team2Cluster'])
                
                inning1FeatureVectorComplete['team1'] = pd.Categorical(inning1FeatureVectorComplete['team1'])
                inning1FeatureVectorComplete['team2'] = pd.Categorical(inning1FeatureVectorComplete['team2'])
               
                inning1FeatureVectorComplete['Continent'] = pd.Categorical(inning1FeatureVectorComplete['Continent'])
                inning1FeatureVectorComplete['year'] = pd.Categorical(inning1FeatureVectorComplete['year'])
                
                
                #inning1FeatureVectorComplete['venueCountryCode'] = pd.Categorical(inning1FeatureVectorComplete['venueCountryCode'])
                #inning1FeatureVectorComplete['tossWinner'] = pd.Categorical(inning1FeatureVectorComplete['tossWinner'])


                dfDummies1 = pd.get_dummies(inning1FeatureVectorComplete['team1'], prefix = 'team1c')
                dfDummies2 = pd.get_dummies(inning1FeatureVectorComplete['team2'], prefix = 'team2c')
                dfDummies3 = pd.get_dummies(inning1FeatureVectorComplete['Continent'], prefix = 'VenueClass')
                
                #dfDummies4 = pd.get_dummies(inning1FeatureVectorComplete['year'], prefix = 'myear') #,dfDummies4
                #dfDummies5 = pd.get_dummies(inning1FeatureVectorComplete['overId'], prefix = 'Over') #,dfDummies4
                #dfDummies6 = pd.get_dummies(inning1FeatureVectorComplete['tossWinner'], prefix = 'tossWinnerTeam') #,dfDummies4
                
                
                inning1FeatureVectorComplete = pd.concat([inning1FeatureVectorComplete,dfDummies1,dfDummies2 ,dfDummies3], axis=1) #dfDummies4,#dfDummies5,  ,dfDummies6

                #inning1FeatureVectorComplete = inning1FeatureVectorComplete.drop(['team1c_4'], axis=1)
                #inning1FeatureVectorComplete = inning1FeatureVectorComplete.drop(['team2c_4'], axis=1)
                #inning1FeatureVectorComplete = inning1FeatureVectorComplete.drop(['VenueClass_2'], axis=1)
                
                
                #print(len(inning1FeatureVectorComplete[(inning1FeatureVectorComplete['matchId']==currMatch)] ))
#                currentOvrWickets= inning1FeatureVectorComplete[( inning1FeatureVectorComplete['matchId']==currMatch) & ( inning1FeatureVectorComplete['overId']==inningsOver)]['RemainingWickets'].iloc[0]
#                
                previousOvrWickets = currentOvrWickets
                
                currentOvrWickets= inning1FeatureVectorComplete[( inning1FeatureVectorComplete['matchId']==currMatch) & ( inning1FeatureVectorComplete['overId']==inningsOver)]['RemainingWickets'].iloc[0]
                
            
                
                if inngs==2:
                   currentOvrRemRuns =     inning1FeatureVectorComplete[( inning1FeatureVectorComplete['matchId']==currMatch) & ( inning1FeatureVectorComplete['overId']==inningsOver)]['remainingRuns'].iloc[0]



                inning1FeatureVectorComplete[targetVar]=  inning1FeatureVectorComplete[targetVar].astype(int)
                y = inning1FeatureVectorComplete[targetVar]
                
                #print(inning1FeatureVectorComplete.columns)
                if inngs==1:
                    selectedCols = inning1FeatureVectorComplete.columns[~inning1FeatureVectorComplete.columns.isin(columnsRemoveInng1)]
                else:
                    selectedCols = inning1FeatureVectorComplete.columns[~inning1FeatureVectorComplete.columns.isin(columnsRemoveInng2)]
                
                
                
                #Data Min-Max Normalization 
# =============================================================================
      
                df = inning1FeatureVectorComplete.copy()
                for col in selectedCols:
                    df[col] = df[col].astype(int)
                    df[col].fillna(0,inplace=True)
                    df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
                    
                inning1FeatureVectorComplete= df.copy() 
                
                for col in selectedCols:
                    inning1FeatureVectorComplete[col].fillna(value=0,inplace=True)
# =============================================================================                    
#                 baselinePrediction=[]
#                 for i in range(len(inning1FeatureVectorComplete)):
#                     thisTeam1 =inning1FeatureVectorComplete['team1'].iloc[i]
#                     thisTeam2 =inning1FeatureVectorComplete['team2'].iloc[i]
                            
#                     meanVal= inning1FeatureVectorComplete[(inning1FeatureVectorComplete['team1']==thisTeam1) & (inning1FeatureVectorComplete['team2']==thisTeam2)][targetVar].mean()
#                     baselinePrediction.append(round(meanVal,2))
 
    
    
                if inningsOver>1:
                    preTrain =inning1FeatureVector_train
                    preTest =inning1FeatureVector_test
                    preYTrain = ytrain
                    preYTest  = ytest
    
                y = y.astype(int)
                if(overSplit==True):
                    inning1FeatureVector_test  =   inning1FeatureVectorComplete[( inning1FeatureVectorComplete['matchId']==currMatch) & ( inning1FeatureVectorComplete['overId']==inningsOver)].copy()
                    #inning1FeatureVector_test  = inning1FeatureVector_test[inning1FeatureVector_test['RemainingWickets']==currentOvrWickets].copy() 
                    
                    #inning1FeatureVector_test  =  inning1FeatureVectorComplete[(inning1FeatureVectorComplete['matchId']==currMatch)& (inning1FeatureVectorComplete['overId']==inningsOver)]

                    ytest                      =  inning1FeatureVector_test[targetVar]
                    #inning1FeatureVector_train,inning1FeatureVector_Y, ytrainTarget,ytest = train_test_split(inning1FeatureVector_test,ytestVar ,test_size=0.1) 
                    #inning1FeatureVector_train =  inning1FeatureVectorComplete[inning1FeatureVectorComplete['matchId']!=currMatch].copy()
                    inning1FeatureVector_train =  inning1FeatureVectorComplete[(inning1FeatureVectorComplete['matchId']!=currMatch) & (inning1FeatureVectorComplete['overId']==inningsOver) & (inning1FeatureVectorComplete['RemainingWickets']==currentOvrWickets) & (inning1FeatureVectorComplete['RemainingWickets']==currentOvrWickets)].copy()
#                    if inngs==2:# and inningsOver == matchLength:
#                        inning1FeatureVector_train  = inning1FeatureVector_train[(inning1FeatureVector_train['remainingRuns']<currentOvrRemRuns+15)  & (inning1FeatureVector_train['remainingRuns']>currentOvrRemRuns-15)].copy() 
                         
                    #print("Train data: ",len(inning1FeatureVector_train),"Test data: ",len(inning1FeatureVector_test))
                    
#                    if inningsOver==41:
#                        inning1FeatureVector_train.to_csv("thisOverData.csv")
                    
                    ytrain                     =  inning1FeatureVector_train[targetVar]
                    
                    #print(inningsOver,"train length",len(ytrain))
                 
                    if(inngs==2):
                        inning1FeatureVector_train =  inning1FeatureVector_train[inning1FeatureVector_train.columns[~inning1FeatureVector_train.columns.isin(columnsRemoveInng2)]]#
                        inning1FeatureVector_test =  inning1FeatureVector_test[inning1FeatureVector_test.columns[~inning1FeatureVector_test.columns.isin(columnsRemoveInng2)]]#
                        
                    else:
                        inning1FeatureVector_train =  inning1FeatureVector_train[inning1FeatureVector_train.columns[~inning1FeatureVector_train.columns.isin(columnsRemoveInng1)]]#
                        inning1FeatureVector_test =  inning1FeatureVector_test[inning1FeatureVector_test.columns[~inning1FeatureVector_test.columns.isin(columnsRemoveInng1)]]#
                    
                    
                    #print(inning1FeatureVector_train.iloc[0])
                    #print(inning1FeatureVector_train.head())
                    #break  
                        
                 
                        
#                     if(targetVar=='totalRunsScoredCurrentOvr'):
#                         inning1FeatureVector_train =  inning1FeatureVector_train[inning1FeatureVector_train.columns[~inning1FeatureVector_train.columns.isin(['totalRunsScoredCurrentOvr'])]]#,'totalRunsScoredCurrentOvr','totalRunsScoredtillNow'])]]#'batsmenOfClust_4','bowlerOfClust_4',
#                         inning1FeatureVector_test =  inning1FeatureVector_test[inning1FeatureVector_test.columns[~inning1FeatureVector_test.columns.isin(['totalRunsScoredCurrentOvr'])]]#,'totalRunsScoredCurrentOvr','totalRunsScoredtillNow'])]]#'batsmenOfClust_4','bowlerOfClust_4',
                    #print(inning1FeatureVector_train.columns)
                    #print(inning1FeatureVector_train['overId'].head(3))
                    
                    #break
                else:
                
                    if(inngs==1):
                        inning1FeatureVector_trainTest = inning1FeatureVectorComplete[inning1FeatureVectorComplete.columns[~inning1FeatureVectorComplete.columns.isin(columnsRemoveInng1)]]
                        #inning1FeatureVectorComplete.columns.isin(['matchId','year','venueCountry','team1','team2','homeTeam','toss_winner_team','team1Cluster','team2Cluster','Continent','overId','tossWinner','yearId','venueCountryCode','venueCountry','team1','team2','homeTeam','toss_winner_team','inning1Runs','inning1ProjectedScore','onCreasebowlerOfTheOvr','wicket','wicket.1','wicket.2','wicket.3','wicket.4','wicket.5','innings1Wickets','inning1RunRate','inning2Runs','innings2Wickets','inning2RunRate','zScore'])]]#,'totalRunsScoredCurrentOvr','totalRunsScoredtillNow'])]]#'batsmenOfClust_4','bowlerOfClust_4',
                    elif(inngs==2):
                        inning1FeatureVector_trainTest = inning1FeatureVectorComplete[inning1FeatureVectorComplete.columns[~inning1FeatureVectorComplete.columns.isin(columnsRemoveInng2)]]#,'totalRunsScoredCurrentOvr','totalRunsScoredtillNow'])]]#'batsmenOfClust_4','bowlerOfClust_4',
                    
                    #print(inning1FeatureVector_trainTest.columns)
#               
                if overSplit==False:
                    MAE_overs,resultDataFrame = RegressionForkFold(inning1FeatureVector_trainTest,y,inningsOver,kfolds,targetVar,fileNameOut,runsConvert)#,fileWriteOutput)

    # #                 #MAE_overs,resultDataFrame =KNNForkFold(inning1FeatureVector_trainTest,y,inningsOver,kfolds,k_neighbors,targetVar)#,fileWriteOutput)
    # #                 #MAE_overs = RegressionForRandomSplit(inning1FeatureVector_train,inning1FeatureVector_testData,ytrain,ytest,y,predOver,targetVar,predOver)#,fileWrite):

                    resultDataFrame['baselinePrediction'] =0
                    resultDataFrame['baselinePrediction'] = baselinePrediction
                    #resultDataFrame.to_csv(str(matchinProgressDataFolder)+'Results/inprogress_Inng'+str(inngs)+'_Over_'+str(inningsOver)+'_Combined_'+str(targetVar)+'_Result_teamIndInfo_'+str(stdTick)+'_LR_23Jan_remCurrentInfo.csv', index=False)

                    absErrorMeanForallOvers.append(np.mean(resultDataFrame['AbsError']))
                    absErrorDevForallOvers.append(np.std(resultDataFrame['AbsError']))
                    absErrorPercentForallOvers.append(MAE_overs)
                else:
                    regStart = time.time()
                    
#                    if(previousOvrWickets>currentOvrWickets):
#                        #print("previous", preTest, "current",inning1FeatureVector_test.iloc[0] )
#                        print("out")
#                        yprediction = RegressionFor1Type(preTrain,preTest,preYTrain,preYTest,targetVar)
#                    
#                        withoutWicketLossPred.append(yprediction[0])
#                        
#                    else:
#                        withoutWicketLossPred.append(0)
                        
                    #print("Regression starts" )#, regStart)
                    yprediction = RegressionFor1Type(inning1FeatureVector_train,inning1FeatureVector_test,ytrain,ytest,targetVar)
                    #meanAbsError, 
                    predData.append(yprediction[0]) #= yprediction# 
                    #print("Regression ends" , time.time() - regStart)
                    #MAE_allOvers.append(meanAbsError)
     
     

        #print('predict Runs', predData)
        #inningFeatureVectorCompletePlayerId.to_csv((str(matchinProgressDataFolderResult)+"inprogressOfInning_matchId_"+str(currMatch)+"_"+str(inngs)+"_LR_targetVar_"+str(targetVar)+".csv"))#MAE%age
        return notFound,predData,withoutWicketLossPred

        
            #################    For overwise predicton of single match :::::::: Ends here  #########

    
  
   

   
def findContributionForTeambyProjection(currMatchId,inningsId):
    
    inningFeatureVectorCompletePlayerId = pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inningsId)+'/inprogress_Inng'+str(inningsId)+'_over1_50Records_2001_19_withBeforeOverInfo.csv')  
  
    singleMatch=True
    strtTime = time.time()
    print("prediction  start")
    matchFound,predictedData,withoutWicketLossPrediction = predictRunsForMatch(currMatchId,inningsId,singleMatch)
    print('prediction done In time', time.time()- strtTime)
    
    #print('matchFound', matchFound)
    if(matchFound!=0):
         inningFeatureVectorCompletePlayerId = inningFeatureVectorCompletePlayerId[inningFeatureVectorCompletePlayerId['matchId']==currMatchId]#.copy()
        
         inningFeatureVectorCompletePlayerId.drop_duplicates(keep='first',subset=['overId'], inplace=True)
         print('Length match, predicted',len(inningFeatureVectorCompletePlayerId), len(predictedData))
         inningFeatureVectorCompletePlayerId['projectedTotal'] =  0  #runPredicted
         inningFeatureVectorCompletePlayerId['actualRunsScored']=  0  #projected
         inningFeatureVectorCompletePlayerId['absError']=  0
         inningFeatureVectorCompletePlayerId['projectedTotal'] = predictedData+ inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
#         inningFeatureVectorCompletePlayerId['projectedTotalWithoutWktLoss']= withoutWicketLossPrediction  #+ inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
         print("Projected Total", inningFeatureVectorCompletePlayerId['projectedTotal'])
         
         inningFeatureVectorCompletePlayerId['actualRunsScored'] = (inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] - inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'])
         
         actualRuns = inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] - inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr']
         
         inningsTotalRuns = inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] 

         inningFeatureVectorCompletePlayerId['absError'] = abs(predictedData- actualRuns) # projected)
         
    
         print('AbsError')
         print(np.array(inningFeatureVectorCompletePlayerId['absError']))
         inningFeatureVectorCompletePlayerId.to_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/contribution/allmatches_KnnSimWieghted_ProjectedTotal/contribution_inngs'+str(inningsId)+'_match_'+str(currMatchId)+'.csv', index=False)
    #(str(matchinProgressDataFolder)+'Results/inprogressOfInning_matchId_'+str(currMatchId)+'_'+str(inningsId)+'_LR__targetVar_PrjectedScore_allOvers_Contribution.csv', index=False)  
         return 1,1#teamPlayers,teamPlayersContribution
    
    else:
        return 0,0





from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold # import KFold
from sklearn.neighbors import KNeighborsRegressor

def RegressionFor1Type(inning1FeatureVector_train,inning1FeatureVector_testData,ytrain,ytest,targetVar):#,fileWrite):
    
#    MAE_array=[]
#    RMSE_array=[]
#    MA_array=[]
    
    inning1FeatureVector_X,inning1FeatureVector_Y, ytrain,ytest = inning1FeatureVector_train, inning1FeatureVector_testData,ytrain,ytest

#    reg = linear_model.Ridge(alpha=1.0).fit(inning1FeatureVector_X, ytrain)#   LinearRegression().normalize=True
#    y_pred = reg.predict(inning1FeatureVector_Y)
#   
#    if len(ytrain)>50:
#        K_neighbors=50
#    if len(ytrain)>20:
#        K_neighbors=20
#    else:
    #targetVar = ytrain.columns
    #if len(ytrain)>15:  
     #   K_neighbors= 15 #len(ytrain)
    #else:
    K_neighbors= len(ytrain)
#
#    #inning1FeatureVector_test = inning1FeatureVector_Y.copy()
##     inning1FeatureVector_test[targetVar]=0
##     inning1FeatureVector_test[targetVar] = ytest #**runsConvert
#
#    
    #n_neighbors=K_neighbors,

    
#    nbrs =KNeighborsRegressor(n_neighbors=K_neighbors, weights='distance')
##    
##    
#    nbrs.fit(inning1FeatureVector_X, ytrain)
#    y_pred = nbrs.predict(inning1FeatureVector_Y)
##    y_pred  = y_pred
    #print(neigh.predict([[1.5]]))
    
   
    
# =============================================================================
#     df1 = inning1FeatureVector_Y
#     inning1FeatureVector_Y_df=(df1-df1.min())/(df1.max()-df1.min())
# =============================================================================
    
    nbrs = NearestNeighbors(n_neighbors=K_neighbors,algorithm='ball_tree', metric='euclidean').fit(inning1FeatureVector_X) 
    distances, indices = nbrs.kneighbors(inning1FeatureVector_Y)
    #print(len(distances))
    #print(distances)
    minDist = distances.min()
    maxDist = distances.max()
    normDist=[]
    for i in range(len(distances)):
        for k in range(len(distances[i])):
            normDist.append((distances[i][k]-minDist)/(maxDist-minDist))
    similarityVal =np.zeros(len(normDist))    
    for i in range(len(normDist)):
        similarityVal[i] = 1- normDist[i]
    #print(similarityVal)
    inning1FeatureVector_train[targetVar]=0
    inning1FeatureVector_train[targetVar] = ytrain #**runsConvert
    similarity =0
    inning1RunsPredicted=[]
    for i in range(len(inning1FeatureVector_Y)):
        runsScored=0
        runsPred=0
        runsPredArray=[]
        similarityScore = 0 
        for k in range((len(similarityVal))):#K_neighbors
            #1-(distances[i][k])
            #print(similarity)
           
            if(similarityVal[k]):
                similarityScore = similarityScore+ similarityVal[k]
                runsPred=runsPred+(inning1FeatureVector_train[targetVar].iloc[indices[i][k]])*(similarityVal[k])
                #runsPredArray.append((inning1FeatureVector_train[targetVar].iloc[indices[i][k]]))*(similarityVal[k]))
            #runsScored  = runsScored + inning1FeatureVector_train['inning1Runs'].iloc[indices[i][k]]
        inning1RunsPredicted.append(int(runsPred/similarityScore)) 
       
    y_pred = inning1RunsPredicted
    
    
  
   # inning1FeatureVector_test['inning1RunsPred'] = y_pred #**runsConvert


  
    return y_pred #np.array(inning1FeatureVector_test['AbsError']),y_pred
     
 
    

    
#manOfMatchRecords  = pd.read_csv('F:/LUMS/Cricket Data/manOfTheMatch_ODI_2003_15.csv')

#manOfMatchPredDataFile = open('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/Results/manOfMatch/manOfTheMatch_predictionData_matchiIds_File.csv','w')
#manOfMatchPredDataFile.write('matchId,manOfTheMatch,matchFound,momPos\n')
#

 #=============================================================================
matchIds =[423793,426428,446957,455234,467886,461569,439151,578625,65662,256614,430887,520601,64859,64861,578618,350046,433596,433564,239917,902643,238198,489217,667649,65638,249213,291365,518965,446968,
           211425,430889,860277,257771,319134,249748,433586,343732,514026,489224,65642,667897,656437,
           386534,636162,567358,293078,249752,597925,474469,860269,415282,461569]
 
 #=============================================================================



autoPlayMatches = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/AutoPlay_Paper_MatchIDs.csv')

#autoPlayMatches = autoPlayMatches[(autoPlayMatches['year']>=2011) & (autoPlayMatches['year']<=2012)  ]['matchId'].unique()
#momDf = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/Results/manOfMatch/after28Feb/manOfTheMatch_predicted_current_bowlWt_0.25_wicketWeight_0.4normal_weightedsimKNN_1.csv')

#filteredAllMatches = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/innings1/filteredMatches_1_All.csv')


filteredAllMatches = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/missedMatchesInProjection.csv')

allmatchesRecord = pd.read_csv(str(matchinProgressDataFolder)+'innings1/inprogress_Inng1_over1_50Records_2001_19_withBeforeOverInfo.csv')  

print("All matches Count: ", len(allmatchesRecord['matchId'].unique()))

import time


matchIdsReq = allmatchesRecord['matchId'].unique() #[66274,66275,66277,66279,66280,66281,66282,1144516,1144987,895813,914235,1022357,1098208,1098210,1120289,1144509,1144998,1152845,1153695,902645,932853]


#matchIdsReq = np.array(allmatchesRecord['matchId'].unique()) #[]









from sklearn.linear_model import Ridge
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor

def model_training_regression(train_X,train_Y,test_X,test_Y,kNbr):
    
    
    
    
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)
    
    clf = Ridge(alpha=1.0)
    clf.fit(train_X, train_Y)
    pred_Y=clf.predict(test_X)
    
    print("Ridge Regression mean_squared_error  : ",np.sqrt(mean_squared_error(test_Y,pred_Y)))
    
    
    

    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(train_X, train_Y)
    pred_Y1 =regr.predict(test_X)
    
    print("SVR Regression mean_squared_error  : ",np.sqrt(mean_squared_error(test_Y,pred_Y1)))
    
    
    
    randForest = RandomForestRegressor(max_depth=None, random_state=0)
    randForest.fit(train_X, train_Y)
    pred_randForest = randForest.predict(test_X)
    
    print("Random Forest Regression mean_squared_error  : ",np.sqrt(mean_squared_error(test_Y,pred_randForest)))
    
    
    
   
    
    
    neigh = KNeighborsRegressor(n_neighbors=kNbr)
    neigh.fit(train_X, train_Y)
    knn =neigh.predict(test_X)
    
    
    return pred_Y,pred_randForest,knn
    #Ridge()


from matplotlib import pyplot as plt




def aggregate_dataAt_over_level(combinedDf,inngs):
    
    inning1FeatureVectorComplete1 =combinedDf.copy()

    projectionDataFile =  inning1FeatureVectorComplete1[(inning1FeatureVectorComplete1['team1']!='Zimbabwe') & (inning1FeatureVectorComplete1['team2']!='Zimbabwe') & (inning1FeatureVectorComplete1['team1']!='Bangladesh') & (inning1FeatureVectorComplete1['team2']!='Bangladesh')] 
    projectionDataFile = projectionDataFile.dropna()


    
    
    OverErrorAll=[]
    OverError=[]
    OverError.append('over')
    OverError.append('MeanAbsError')
    OverError.append('Mean_absError_randForest')
    OverError.append('Mean_absError_ridgRegr')
    OverError.append('Mean_absError_knn')
#    OverError.append('MeanAbsErrorLewis35')
#    OverError.append('MeanAbsErrorLewisMean')
#    OverError.append('MeanAbsErrorLewisMed')
#    OverError.append('MedianAbsError')
    OverError.append('StdAbsError')
#    OverError.append('MeanPercentAbsError')
#    OverError.append('RMSE')
    
    OverErrorAll.append(OverError)
        
        
    for ovr in projectionDataFile['overId'].unique():
        OverError=[]
        projectedOver = projectionDataFile[projectionDataFile['overId']==ovr]
        meanAbs = projectedOver['absError_randForest'].mean()
        medianAbs = projectedOver['absError_randForest'].median()
        stdAbs = projectedOver['absError_randForest'].std()
        #rmse = np.sqrt(mean_squared_error(projectedOver['projectedTotal'], projectedOver['inning'+str(inngs)+'Runs']))
        #meanPercentAbs = np.zeros(len(projectedOver['absError']))
        
#        for i in range(len(projectedOver['absError'])):
#            if projectedOver['projectedTotal'].iloc[i]!=0:
#                meanPercentAbs[i] = abs((projectedOver['absError'].iloc[i]/projectedOver['projectedTotal'].iloc[i])*100)
#            else:
#                meanPercentAbs[i] = abs((projectedOver['absError'].iloc[i])*100)
#                
#        #stdAbs = projectedOver['RMSE'].std()
        OverError.append(ovr)
        OverError.append(meanAbs)
        OverError.append(projectedOver['absError_randForest'].mean())
        OverError.append(projectedOver['absError_ridgeRegr'].mean())
        OverError.append(projectedOver['absError_knn'].mean())
        
        #OverError.append(projectedOver['dlsProjectionError35'].mean())
        #OverError.append(projectedOver['dlsProjectionErrorMean'].mean())
        #OverError.append(projectedOver['dlsProjectionErrorMed'].mean())
        #OverError.append(medianAbs)
        OverError.append(stdAbs)
                             
       # OverError.append(int(np.mean(meanPercentAbs)))
        #OverError.append(rmse)
        OverErrorAll.append(OverError)
        
    OverErrorAllDf = pd.DataFrame(OverErrorAll) 
    
    #return OverErrorAllDf
    
    OverErrorAllDf.to_csv("projectionErrorMean_OverWiseMean_innings_"+str(inngs)+".csv",index=False,header=0)
        
                          #C:/Users/naima/Google Drive/Cricket Data/Naimat Ullah/WriteUp/KDD/Data/inprogressOfInning2_projectionError.csv', index=False, header=0)    
    return 0   



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
def plot_errors(combinedDf,inngs,kNbr):
    
    
    
    if inngs==1:
        combinedDf = pd.read_csv('newPredictionwithRegression_kfold_1.csv') #pd.DataFrame(date=None)
        for i in range(2,fold+1):
            df = pd.read_csv('newPredictionwithRegression_kfold_'+str(fold)+'.csv')
            combinedDf = combinedDf.append([df])
            #combinedDf = pd.concat(combinedDf,df,axis=1)
        combinedDf.to_csv("combinedforAll_folds_innings_"+str(inngs)+".csv",index=False,header=0)
    else:
        combinedDf = pd.read_csv('newPredictionwithRegression_kfold_11.csv') #pd.DataFrame(date=None)
        
        for i in range(12,21):
            df = pd.read_csv('newPredictionwithRegression_kfold_'+str(fold)+'.csv')
            combinedDf = combinedDf.append([df])
        print("Lenght of Combined Df" ,len(combinedDf))
            #combinedDf = pd.concat(combinedDf,df,axis=1)
        combinedDf.to_csv("combinedforAll_folds_innings_"+str(inngs)+".csv",index=False,header=0)
    #combinedDf = combinedDf[combinedDf[']]
    aggregate_dataAt_over_level(combinedDf,inngs)
    OverErrorAllDf = pd.read_csv("projectionErrorMean_OverWiseMean_innings_"+str(inngs)+".csv")
        
    
    print(OverErrorAllDf.columns)
    
    fig, ax1 = plt.subplots(figsize=(10,10))
    
 
    # multiple line plot
    #plt.plot(np.arange(1,51) ,'MeanAbsError', data=OverErrorAllDf,label='Ours', marker='+',color='green', markersize=4, linewidth=4)
    
    plt.plot( np.arange(1,51), 'Mean_absError_randForest', data=OverErrorAllDf,label='Random Forest', marker='x', color='olive', linewidth=2)
    plt.plot( np.arange(1,51), 'Mean_absError_ridgRegr', data=OverErrorAllDf,label='Ridge Regression', marker='o', color='red', linewidth=2)
    plt.plot( np.arange(1,51), 'Mean_absError_knn', data=OverErrorAllDf,label='KNN(K='+str(kNbr)+')', marker='o', color='blue', linewidth=2)
   
    
    #plt.plot( np.arange(1,51), 'MeanAbsErrorLewisMed', data=OverErrorAllDf,label='DLS with median runs', marker='', color='blue', linewidth=2)
    
    
    #plt.plot( np.arange(1,51), 'RMSE', data=OverErrorAllDf, marker='', color='olive', linewidth=2, linestyle='dashed', label="EMSE")
    plt.xticks(np.arange(0,55,5))
    
    plt.yticks(np.arange(0,50,5))
    plt.ylabel(" MAE in Projected Runs for Innings-"+ str(inngs))
    plt.xlabel(" Over Id")
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig("rigression_randomForest_knn_MeanAbs_errorPlot_Innings_"+str(inngs)+".jpeg")
    #plt.savefig('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/contribution/ProjectionError_Inning_'+str(inngs)+'_Ours_Lewis.jpeg')


def training_model(train_match,test_match,inngs,kNbr,fold):
    
    
    
    inningsInProgressOversCombined =  pd.read_csv(str(matchinProgressDataFolder)+'inprogressOfInning'+str(inngs)+'_over1_50Records_withAllEncodedFeatures_Combined.csv')          
   
    inning1FeatureVectorComplete1 = inningsInProgressOversCombined
    #[inningsInProgressOversCombined['overId']==inningsOver].copy() # pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inngs)+'/inprogressOfInning'+str(inngs)+'_over'+str(inningsOver)+'Records_withAllEncodedFeatures.csv')  
                
    inning1FeatureVectorCompleteNew = inning1FeatureVectorComplete1[(inning1FeatureVectorComplete1['team1']!='Zimbabwe') & (inning1FeatureVectorComplete1['team2']!='Zimbabwe') & (inning1FeatureVectorComplete1['team1']!='Bangladesh') & (inning1FeatureVectorComplete1['team2']!='Bangladesh')] 



    stdev1 = int(inning1FeatureVectorCompleteNew['inning1Runs'].std())
    stdev2 = int(inning1FeatureVectorCompleteNew['inning2Runs'].std())
    mean1 = int(inning1FeatureVectorCompleteNew['inning1Runs'].mean())
    mean2 = int(inning1FeatureVectorCompleteNew['inning2Runs'].mean())


    inning1FeatureVectorCompleteIn = inning1FeatureVectorCompleteNew[(inning1FeatureVectorCompleteNew['inning1Runs']>mean1 -(stdTick*stdev1)) & (inning1FeatureVectorCompleteNew['inning1Runs']<mean1+(stdTick*stdev1))].copy()
    inning1FeatureVectorComplete = inning1FeatureVectorCompleteIn[(inning1FeatureVectorCompleteIn['inning2Runs']>mean2 -(stdTick*stdev2)) & (inning1FeatureVectorCompleteIn['inning2Runs']<mean2+(stdTick*stdev2))].copy()
                

    if inngs==2:
        inning1FeatureVectorComplete['remainingRuns'] = 0
        inning1FeatureVectorComplete['remainingRuns'] = inning1FeatureVectorComplete['inning1Runs'] - inning1FeatureVectorComplete['totalRunsScoredBeforeThisOvr']
                   
    

    labels = inning1FeatureVectorComplete['team1'].astype('category').cat.categories.tolist()
    replace_continent_comp = {'team1' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
    inning1FeatureVectorComplete .replace(replace_continent_comp, inplace=True)
                 
                
                
    labels = inning1FeatureVectorComplete['team2'].astype('category').cat.categories.tolist()
    replace_continent_comp = {'team2' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
    inning1FeatureVectorComplete .replace(replace_continent_comp, inplace=True)
                

    ##############  categorical variable to one-hot encoding ##########################   
    inning1FeatureVectorComplete['team1Cluster'] = pd.Categorical(inning1FeatureVectorComplete['team1Cluster'])
    inning1FeatureVectorComplete['team2Cluster'] = pd.Categorical(inning1FeatureVectorComplete['team2Cluster'])
    inning1FeatureVectorComplete['team1'] = pd.Categorical(inning1FeatureVectorComplete['team1'])
    inning1FeatureVectorComplete['team2'] = pd.Categorical(inning1FeatureVectorComplete['team2'])
    inning1FeatureVectorComplete['Continent'] = pd.Categorical(inning1FeatureVectorComplete['Continent'])
    inning1FeatureVectorComplete['year'] = pd.Categorical(inning1FeatureVectorComplete['year'])
    

    dfDummies1 = pd.get_dummies(inning1FeatureVectorComplete['team1'], prefix = 'team1c')
    dfDummies2 = pd.get_dummies(inning1FeatureVectorComplete['team2'], prefix = 'team2c')
    dfDummies3 = pd.get_dummies(inning1FeatureVectorComplete['Continent'], prefix = 'VenueClass')
                
    #dfDummies4 = pd.get_dummies(inning1FeatureVectorComplete['year'], prefix = 'myear') #,dfDummies4
    #dfDummies5 = pd.get_dummies(inning1FeatureVectorComplete['overId'], prefix = 'Over') #,dfDummies4
    #dfDummies6 = pd.get_dummies(inning1FeatureVectorComplete['tossWinner'], prefix = 'tossWinnerTeam') #,dfDummies4
    
                
    inning1FeatureVectorComplete = pd.concat([inning1FeatureVectorComplete,dfDummies1,dfDummies2 ,dfDummies3], axis=1) #dfDummies4,#dfDummies5,  ,dfDummies6

   
    
    training_data = inning1FeatureVectorComplete[inning1FeatureVectorComplete['matchId'].isin(train_match)]
    testing_data = inning1FeatureVectorComplete[inning1FeatureVectorComplete['matchId'].isin(test_match)]
    
    #print("Data Columns",training_data.columns)
    
    
    if inngs==1:
        removeCols = columnsRemoveInng1
    elif inngs==2:
        removeCols = columnsRemoveInng2
    
    X_train =  training_data[training_data.columns[~training_data.columns.isin(removeCols)]]#

    #print("Data Colulmns",X_train.columns)
    X_test =  testing_data[testing_data.columns[~testing_data.columns.isin(removeCols)]]#
    
    
    
    targetVar = 'inning'+str(inngs)+'ProjectedScore'   # 'inning'+str(inningsId)+'Runs' #
    
    
    #inning1FeatureVector_test[targetVar]
    
    Y_train=    training_data[targetVar]
    Y_test =    testing_data[targetVar]
    
    
    print("X_test",len(X_test),"X_train",len(X_train),"Y_train",len(Y_train),"Y_test", len(Y_test))
    print("\n")
                  
    print('Training Data',len(training_data['matchId'].unique()))
    print('Testing Data',len(testing_data['matchId'].unique()))  
    #kNbr=20
    predicted_Y, randForestOut,knnOut= model_training_regression(X_train,Y_train,X_test,Y_test,kNbr)
    
    
    testing_data['projectRemaining_knn'] = knnOut
    testing_data['projectRemaining_ridgeRegr'] = predicted_Y
    testing_data['projectRemaining_randForest'] = randForestOut# predicted_Y
    
    testing_data['absError_randForest'] = abs(Y_test -randForestOut)# predicted_Y)
    
    
    testing_data['absError_ridgeRegr'] = abs(Y_test - predicted_Y)
    testing_data['absError_knn'] = abs(Y_test - knnOut)
    
    print("Mean Absolute Error Ridge Regression", np.mean(testing_data['absError_ridgeRegr']))
    print("Mean Absolute Error Random Forest", np.mean(testing_data['absError_randForest']))
    print("Mean Absolute Error KNN ", np.mean(testing_data['absError_knn']))
    
    
    testing_data.to_csv("newPredictionwithRegression_kfold_"+str(fold)+".csv")
    
    #print("test columns",testing_data.columns)
#    plot_errors(testing_data,inngs,kNbr)
    


    return testing_data
 
    

def findOverbyOverProject(currMatchId,inningsId,testingData,fold):
    
    print("Current Match",currMatchId)
    inningFeatureVectorCompletePlayerId = pd.read_csv(str(matchinProgressDataFolder)+'innings'+str(inningsId)+'/inprogress_Inng'+str(inningsId)+'_over1_50Records_2001_19_withBeforeOverInfo.csv')  
  
    
    predictedData = np.array(testingData[testingData['matchId']==currMatchId]['projectRemaining_knn'])
    
    #print(predictedData)
    singleMatch=True
    strtTime = time.time()
    
    #print('matchFound', matchFound)
    try:
         inningFeatureVectorCompletePlayerId = inningFeatureVectorCompletePlayerId[inningFeatureVectorCompletePlayerId['matchId']==currMatchId]#.copy()
        
         inningFeatureVectorCompletePlayerId.drop_duplicates(keep='first',subset=['overId'], inplace=True)
         print('Length match, predicted',len(inningFeatureVectorCompletePlayerId), len(predictedData))
         inningFeatureVectorCompletePlayerId['projectedTotal'] =  0  #runPredicted
         inningFeatureVectorCompletePlayerId['actualRunsScored']=  0  #projected
         inningFeatureVectorCompletePlayerId['absError']=  0
         inningFeatureVectorCompletePlayerId['projectedTotal_knn'] = np.array(testingData[testingData['matchId']==currMatchId]['projectRemaining_knn']) + inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
         inningFeatureVectorCompletePlayerId['projectedTotal_ridgeRegr'] = np.array(testingData[testingData['matchId']==currMatchId]['projectRemaining_ridgeRegr']) + inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
         inningFeatureVectorCompletePlayerId['projectedTotal_randForest'] = np.array(testingData[testingData['matchId']==currMatchId]['projectRemaining_randForest']) + inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
         
         
#        inningFeatureVectorCompletePlayerId['projectedTotalWithoutWktLoss']= withoutWicketLossPrediction  #+ inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'] 
         #print("Projected Total", inningFeatureVectorCompletePlayerId['projectedTotal'])
         
         inningFeatureVectorCompletePlayerId['actualRunsScored'] = (inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] - inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr'])
         
         actualRuns = inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] - inningFeatureVectorCompletePlayerId['totalRunsScoredBeforeThisOvr']
         
         inningsTotalRuns = inningFeatureVectorCompletePlayerId['inning'+str(inningsId)+'Runs'] 

         inningFeatureVectorCompletePlayerId['absError'] = abs(predictedData- actualRuns) # projected)
         
    
         print('AbsError')
         #print(np.array(inningFeatureVectorCompletePlayerId['absError']))
         inningFeatureVectorCompletePlayerId.to_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/contribution/allmatches_RandomForest_ProjectedTotal/contribution_inngs'+str(inningsId)+'_match_'+str(currMatchId)+'.csv', index=False)
    #(str(matchinProgressDataFolder)+'Results/inprogressOfInning_matchId_'+str(currMatchId)+'_'+str(inningsId)+'_LR__targetVar_PrjectedScore_allOvers_Contribution.csv', index=False)  
         return 1#teamPlayers,teamPlayersContribution
    
    except:
        print("some error in projection mapping")




#inningsInProgressContOversCombined.to_csv("C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/contribution/inprogressOfInning"+str(inngs)+"_projectionCombined_AllMatches.csv", index=False)

inningsInProgressContOversCombined =  pd.read_csv("C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/contribution/inprogressOfInning"+str(inngs)+"_projectionCombined_AllMatches.csv")



#filteredAllMatches = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchInProgress/InningsOverRecordsBeforeOverBowled/missedMatchesInProjection.csv')

#allmatchesRecord = pd.read_csv(str(matchinProgressDataFolder)+'innings1/inprogress_Inng1_over1_50Records_2001_19_withBeforeOverInfo.csv')  

print("All matches Count: ", len(inningsInProgressContOversCombined['matchId'].unique()))

import time


matchIdsReq = inningsInProgressContOversCombined['matchId'].unique()
#matchIdsReq = np.array(filteredAllMatches['matchId'].unique())
#[66274,66275,66277,66279,66280,66281,66282,1144516,1144987,895813,914235,1022357,1098208,1098210,1120289,1144509,1144998,1152845,1153695,902645,932853]
#


#filteredAllMatches['matchId'] 
#train_match, test_match = train_test_split(matchIdsReq, test_size=0.15, random_state=None)

#print('Train',len(train_match),'Test', len(test_match))

#print(train_match)
#print(test_match)


X = matchIdsReq#["a", "b", "c", "d"]
kf = KFold(n_splits=10)
fold=0
kNbr=10
for inngs in range(1,3):
    for train, test in kf.split(matchIdsReq):
        print("train test split")
        train_match=[]
        test_match=[]
        for i in train:
            train_match.append(matchIdsReq[i])
        for j in test:
            test_match.append(matchIdsReq[j])
#        print("%s %s" % (train, test))
#        print("test_match", test_match)
#        print("train_match", train_match)
        fold+=1
        print(fold)
        test_output =training_model(train_match,test_match,inngs,kNbr,fold)
        for match in test_output['matchId'].unique():
            findOverbyOverProject(match,inngs,test_output,fold)
    plot_errors(fold,inngs,kNbr)
      
        


    
#combinedDf = pd.read_csv('newPredictionwithRegression_kfold_1.csv') #pd.DataFrame(date=None)
#for i in range(2,fold+1):
#     df = pd.read_csv('newPredictionwithRegression_kfold_'+str(fold)+'.csv')
#     combinedDf = combinedDf.append([df])
#     #combinedDf = pd.concat(combinedDf,df,axis=1)
#combinedDf.to_csv("combinedforAll_folds.csv")
#print(len(combinedDf))
##from sklearn.model_selection import KFold
##
##
#matchIdsReq = [66274,66275,66277,66279,66280,66281,66282,1144516,1144987,895813,914235,1022357,1098208,1098210,1120289,1144509,1144998,1152845,1153695,902645,932853]
##
##
#X = matchIdsReq#["a", "b", "c", "d"]
#kf = KFold(n_splits=10)
#
#for train, test in kf.split(X):
#    print("train test split")
#    train_match=[]
#    test_match=[]
#    for i in train:
#        train_match.append(X[i])
#    for j in test:
#        test_match.append(X[j])
##    print("%s %s" % (train, test))
#    print("test_match", test_match)
#    print("train_match", train_match)

        
#from sklearn.model_selection import RepeatedKFold
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#random_state = 12883823
#rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
#for train, test in rkf.split(X):
#    print("%s %s" % (train, test))
#


#for match in range(len(filteredAllMatches)):# lewisComparisonMatch: #  #matchIdsReq: #range(1):#(1020035,1020036):#  len(manOfMatchRecords)):
#   try:
#        currMatchId = filteredAllMatches['matchId'].iloc[match]# matchId # 848845 #1144516#65641# 848845#  1120289 #1144516#914235 #momDf['matchId'].iloc[matchId] #inningsData['matchId'].iloc[match] #238189 # 237568# 247505 #  #int(len(manOfMatchRecords)
#        manOfMatch =  0 #manOfMatchRecords['mom_id'].iloc[match] #  33335 # 36185# 49289 # 
#        if currMatchId!=914207 and currMatchId!=1140381:
#            print('current Match id', currMatchId,'MoM', manOfMatch)
#            team1Players,team1PlayersContribution  = findContributionForTeambyProjection(int(currMatchId),1) # findContributionForTeam(int(currMatchId),1)
#    
#            team2Players,team2PlayersContribution  = findContributionForTeambyProjection(int(currMatchId),2) #findContributionForTeam(int(currMatchId),2)
#        else:
#            print('not good match')
#      
#   except:
#        print('some Error')

