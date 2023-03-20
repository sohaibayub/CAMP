###  Here ball by ball records in json files are extracted for given match ids  from below given links. 
#### Works for Python3
import ast
import pandas as pd
import numpy as np
import csv
import sys
import json
import matplotlib.pyplot as plt
#import seaborn as sns
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
#year= '2018'  # Year in which match was happening

# match Ids, for which match records needs to be extracted. 
#matchIds=["1134034", "1134035", "1122283", "1134036", "1122284", "1134037", "1115775"]


allMatchSummary = pd.read_csv('C:/python/Scripts/JupyterNotebooks/matchSummary/allODIMatchesSummary_DetailInfo_2001_2019.csv', sep=',')

allMatchSummary.drop(['Unnamed: 0'], axis=1, inplace=True)

#allMatchSummary.year.unique()

errlist=[]
f = open("errors", 'w')
teamName =  ['England','Australia','South Africa','West Indies','New Zealand','India','Pakistan','Sri Lanka','Zimbabwe','Bangladesh']
years = [2019]

filter1 =  allMatchSummary['team1'].isin(teamName)
filter2 =  allMatchSummary['team2'].isin(teamName)
filter3 =  allMatchSummary['rain_rule']!=1
filter4 =  allMatchSummary['year'].isin(years)
filter5 =  allMatchSummary['matchId']>1144523

matchSummariesFiltered = allMatchSummary[filter1 & filter2 & filter3 & filter4 & filter5]
print("matchCount", len(matchSummariesFiltered))
matchCount=0
for matid in matchSummariesFiltered['matchId']: 
    matchCount= matchCount+1
    if(matchCount%50==0):
        print('done',matchCount)
    year = matchSummariesFiltered[matchSummariesFiltered['matchId']==int(matid)]['year'].iloc[0]
    #print(year)
#for matid in matchIds:
    if(1==1):#try:
    
    #setting up the master json.
        master={}
        master['teams'] = {}
        master['runs'] = {}

    #getting runs[innings1]	
        wagon1= "http://www.espncricinfo.com//ci/engine/match/live/gfx/"+str(matid)+".json?inning=1;template=wagon"
       
        html = urlopen(wagon1)

        soupwagon1 = BeautifulSoup(html.read(), 'lxml')#'lxml')
       
        soupwagon1.prettify()
        #print(soupwagon1)
        w1=soupwagon1.find('p')
        #print('w1', w1)
        b1=w1.text
        js1=str(b1)
        inp1=json.loads(js1)
        master['runs']['inn1']= inp1['runs']

        time.sleep(2)
    

    #getting ball by ball records for innings 
        wagon2= "http://www.espncricinfo.com//ci/engine/match/live/gfx/"+str(matid)+".json?inning=2;template=wagon"
        soupwagon2 = BeautifulSoup(urlopen(wagon2).read(), "lxml")
        soupwagon2.prettify()
        w2=soupwagon2.find('p')
        b2=w2.text
        js2=str(b2)
        inp2=json.loads(js2)
        master['runs']['inn2']= inp2['runs']
        time.sleep(2)

    #getting teams, wickets

        player1= "http://www.espncricinfo.com/ci/engine/match/live/gfx/"+str(matid)+".json?template=pie_wickets"
        soupplayer1 = BeautifulSoup(urlopen(player1).read(), "lxml")
        soupplayer1.prettify()
        wp1=soupplayer1.find('p')
        bp1=wp1.text
        jsp1=str(bp1)
        pinp1=json.loads(jsp1)

        for i,v in enumerate(pinp1['t2']['p']):
            for key,value in v.items():
                pinp1['t2']['p'][i]={}
                pinp1['t2']['p'][i]['id']=key
                pinp1['t2']['p'][i]['name']=value

        for i,v in enumerate(pinp1['t1']['p']):
            for key,value in v.items():
                pinp1['t1']['p'][i]={}
                pinp1['t1']['p'][i]['id']=key
                pinp1['t1']['p'][i]['name']=value


        master['teams'] = pinp1
        
        #Save file with below name .
        with open('C:/python/All_ODIs_2001_19_top10Teams/'+str(matid)+'_'+master['teams']['t1']['n']+'_'+master['teams']['t2']['n']+'_2_'+str(year)+'.json', 'w') as json_file:
                json.dump(master, json_file)

        time.sleep(1)

    #except Exception: 
        #errlist.append(matid)

f.write(str(errlist))


