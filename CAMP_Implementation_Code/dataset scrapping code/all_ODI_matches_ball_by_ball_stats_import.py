# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 07:47:48 2018

@author: Naimat
"""


##########################################################################
#Here we read data of match summeries for ODIs.

import csv,json
import pandas as pd
import os
from pprint import pprint

import_fileLocation = 'F:/LUMS/Research With Dr.Imdad Ullah Khan/Cricket Data/Data Files/All ODIs/'
#python code to read from multiple files at a time and combining in python Dataframe



#Import from CSV file in Python dictionary
player_id                       =   "player_id"
batting_hand                    =   "batting_hand"
bowling_hand                    =   "bowling_hand"
name                            =   "name"
batting_style                   =   "batting_style"
age_years                       =   "age_years"
bowling_style                   =   "bowling_style"
batting_style_long              =   "batting_style_long"
player_primary_role             =   "player_primary_role"
bowling_pacespin                =   "bowling_pacespin"
player_team                     =   "player_team"
bowling_style_long              =   "bowling_style_long"




"""
allPlayersData = {player_id:[], batting_hand:[], bowling_hand:[],name:[],batting_style:[],age_years:[],bowling_style:[],
                 batting_style_long:[],player_primary_role:[],bowling_pacespin:[],player_team:[],bowling_style_long:[]}


 
"""
i=22
score1 =0
ballsPlayed=0
#fid = open(import_fileLocation+'All_Players.json', 'r')
#ids = json.load(fid)

with open(import_fileLocation+'64857_Pakistan_New Zealand_2_2004.json') as data_file:    
    data = json.load(data_file)
"""
    print(data['teams']['t1']['p'][0]['id'])
    for i in range(0,11):
       if(data['runs']['inn2'][i]['bat']=='37000'):
           score =score + int(data['runs']['inn2'][i]['r'])
           ballsPlayed = ballsPlayed + 1
           
    print(data['pvp']['bat']['37712']['39024']['8']['runs'])

print(len(data['runs']['inn2']))
print('Score:',score)
print('ballsPlayed:',ballsPlayed) 
"""

#This fucntion computes runs scored and balls played by given bastman Id
# It take parameter data['runs']['inn2'] this list and total ball played in the innings along with playerId
def computeRunsScoredBallPlayedByaBatsmanInMatch(battingData,totalBallinInnings,batsmanId):
    score = 0
    ballsPlayed = 0
    for i in range(0,totalBallinInnings):
       if(battingData[i]['bat']==batsmanId):
           score = score +  int(battingData[i]['r'])
           ballsPlayed = ballsPlayed + 1
    print('Score:',score)
    print('ballsPlayed:',ballsPlayed) 
    
  
computeRunsScoredBallPlayedByaBatsmanInMatch(data['runs']['inn2'],len(data['runs']['inn2']),'37000')



print(type(data['runs']['inn2']))
"""
    allPlayersData[player_id]       = data["Players"]["4144"]["batting_hand"]
    allPlayersData[batting_hand]    = data["Players"]["4144"]["batting_hand"]
    allPlayersData[bowling_hand]    = data["Players"]["4144"]["batting_hand"]
    allPlayersData[name]            = data["Players"]["4144"]["batting_hand"]
    allPlayersData[batting_style]   = data["Players"]["4144"]["batting_hand"]

    allPlayersData[age_years]           = data["Players"]["4144"]["batting_hand"]
    allPlayersData[bowling_style]       = data["Players"]["4144"]["batting_hand"]
    allPlayersData[batting_style_long]  = data["Players"]["4144"]["batting_hand"]
    allPlayersData[player_primary_role] =  data["Players"]["4144"]["batting_hand"]
    allPlayersData[bowling_pacespin]    = data["Players"]["4144"]["batting_hand"]
    allPlayersData[player_team]         = data["Players"]["4144"]["batting_hand"]
    allPlayersData[bowling_style_long]  = data["Players"]["4144"]["batting_hand"]
"""