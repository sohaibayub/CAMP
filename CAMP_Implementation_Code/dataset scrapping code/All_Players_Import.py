# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 06:01:44 2018

@author: Naimat
"""


##########################################################################
#Here we read data of match summeries for ODIs.

import csv,json
import pandas as pd
import os
from pprint import pprint

import_fileLocation = 'F:/LUMS/Research With Dr.Imdad Ullah Khan/Cricket Data/Data Files/players/'
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





allPlayersData = {player_id:[], batting_hand:[], bowling_hand:[],name:[],batting_style:[],age_years:[],bowling_style:[],
                 batting_style_long:[],player_primary_role:[],bowling_pacespin:[],player_team:[],bowling_style_long:[]}


csvFile = csv.reader(open(import_fileLocation+'All_Players.csv', "r"))
a_line_after_header = next(csvFile)
for row in csvFile:
    allPlayersData[player_id].append(row[0])
    allPlayersData[batting_hand].append(row[1])
    allPlayersData[bowling_hand].append(row[2])
    allPlayersData[name].append(row[3])
    allPlayersData[batting_style].append(row[4])

    allPlayersData[age_years].append(row[5])
    allPlayersData[bowling_style].append(row[6])
    allPlayersData[batting_style_long].append(row[7])
    allPlayersData[player_primary_role].append(row[8])
    allPlayersData[bowling_pacespin].append(row[9])
    allPlayersData[player_team].append(row[10])
    allPlayersData[bowling_style_long].append(row[12])

 
"""
#fid = open(import_fileLocation+'All_Players.json', 'r')
#ids = json.load(fid)
with open(import_fileLocation+'All_Players.json') as data_file:    
    data = json.load(data_file)
    
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
    
print(len(data["Players"]))

#print(ids['runs']['inn1'][1])
"""
allPlayersDataDf = pd.DataFrame.from_dict(allPlayersData, orient='columns', dtype=None)    
print(allPlayersDataDf.shape)
#print(allPlayersDataDf.head())

print('Total Number of players records in file : ', len(allPlayersDataDf))

print(allPlayersDataDf[allPlayersDataDf['player_id']=='56143'])
#print(allPlayersDataframe.head())
#print(allPlayersDataframe.shape)


#print(allODIMatchesSummaryDataframe.head(1))
#allODIMatchesSummaryDataframe.rename(columns={'Team 1': 'Team1', 'Team 2': 'Team2', 'Id':'matchId'}, inplace=True)

#print(allODIMatchesSummaryDataframe.head(1))