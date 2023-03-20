###### This code extract the batting overall performance of all players from history till today ################

bowlersFolder = 'C:/python/Scripts/JupyterNotebooks/bowlersFeatures/'

from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
import sys

for page in range(1,786): #776 # pages count verify from the given link by checking last data page
   
    #theurl ="http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;page="+str(page)+";template=results;type=batting"
    theurl='http://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;page='+str(page)+';filter=advanced;opposition=1;opposition=2;opposition=25;opposition=3;opposition=4;opposition=5;opposition=6;opposition=8;opposition=9;orderby=start;spanmax1=20+Oct+2019;spanmin1=01+Jan+2001;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=5;team=6;team=7;team=8;team=9;template=results;type=bowling;view=innings'
    print(page)   
    #print(theurl)
    soup = BeautifulSoup(urlopen(theurl).read(), "html")
    soup.prettify()
    if(page==1):
        file = open(str(bowlersFolder)+"All_Players_Bolwing_Stats_ODI_from01Jan2001_to_20Oct19.csv", 'w')

        ### player batting performance records header
       
        #file.write("Name,Id,Span,Match,Inns,totalBalls,totalRuns,wickets,BBI,Avg,Economy,StrikeRate,4 wicketsInInnings,5 wicketsInInnings"+"\n")
        
     ######################## Table header start    #######################    
        tablestats=soup.findAll("table", {"class" : "engineTable"}, limit=3)[2]
        data=tablestats.find("thead")
   
        row=data.findAll('tr', {"class" : "headlinks"})
        for col in row:
            data = col.findAll('th')
            for i in range(len(data)):               
                if data[i].string!=None:
                    file.write(data[i].string)           
                    file.write(',')
                if(i==0):
                    file.write('playerId,')
                    file.write('playerTeam,')
#                 if(i==1):
#                     file.write('notOut,')
                if(i==9):
                    file.write('date,')
            file.write('\n')
        
    ######################## Table header ended    #######################
    tablestats=soup.findAll("table", {"class" : "engineTable"}, limit=3)[2]
    data=tablestats.find("tbody")
   
    row=data.findAll('tr', {"class" : "data1"})
    for col in row:
        data = col.findAll('td')       
        for i in range(len(data)):
            if(data[1].string!='DNB'):
                if i<9 and data[i].find('a') is not None:
                    link = data[i].a['href']
                    file.write(str(data[i].a.string))
                    if(i==0):  
                        pteam = data[i].text.split('(')[1].split(')')[0]    # get player team which is present after his name
                        playerId = link.split('/')[4].split('.')[0]   # get player Id from hyper link of the row
                        file.write(','+playerId+',')
                        file.write(pteam)
                    file.write(',')
#                 if i==1:
#                     if(len(data[1].string.split('*'))>1):
#                         file.write(str(data[1].string.split('*')[0])+',')
#                         file.write('1')
#                     else:
#                         file.write(data[1].string)
#                         file.write(',')
#                         #file.write(',')

                
                if data[i].string!=None:  # i!=1 andi!=5 and 
                    if(i==5 and data[i].text!='-'):
                        file.write(str(float(data[i].text)))    ## Bowlers economy       
                    else:    
                        file.write(data[i].string)
                    file.write(',')
        if(data[1].string!='DNB'):     
            file.write('\n')
file.close()