import requests
import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import ast
from constants import API_KEY

class FootballDataAPI:
    def __init__(self, api_key = None):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': self.api_key}


    #get the Current Standings for each of the Gamweeks in a Season
    def get_current_standings(self, season=None, gameweek = None):
        url = f"{self.base_url}/competitions/PL/standings"
        params = {}
        if season:
            params['season'] = season
        if gameweek:
            # url = f"{url}?matchday={gameweek}"
            params['matchday'] = gameweek

        response = requests.get(url, headers = self.headers, params = params)

        if response.status_code == 200:
            data = response.json()
            standings = data.get('standings', []) 
            return standings
        else:
            return None

    #get the winner for the season in order to compare   
    def determine_season_winner(self, season = None):
        standings = self.get_current_standings(season, 38)
        if standings:
            for table in standings:
                for tables in table['table']:     
                    if tables['position'] == 1:
                        return tables['team']['id']
                    
    #conver the text file into csv in order to colelct data
    def convert_table_to_csv(self, file = None, season = None):
        rows = []
        with open(file, newline="") as f:
           reader = csv.reader(f)
           for line in reader:
               for table_str in line:
                   table_data = ast.literal_eval(table_str)
                   
                   if table_data['type'] != "TOTAL":
                       continue
                   
                   for team in table_data['table']:
                       rows.append({
                           "season": season,
                           "gameweek": team['playedGames'],
                           "teamID": team['team']['id'],
                           "position": team['position'],
                           "form": team['form'],
                           "won": team['won'],
                           "draw": team['draw'],
                           "lost": team['lost'],
                           "points": team['points'],
                           "goalsFor": team['goalsFor'],
                           "goalsAgainst": team['goalsAgainst'],
                           "goalDifference": team['goalDifference']
                       })
        df = pd.DataFrame(rows)
        df.to_csv(f"prem_table_{season}.csv", index=False)
        return df  
            

def main():
    a = FootballDataAPI()
    # with open('predict.txt', 'w', newline='') as output:
    #     writer = csv.writer(output)
    #     for i in range(22):
    #         standings = a.get_current_standings(2025, i + 1)
    #         if standings is None:
    #             break

    #         writer.writerow([str(table) for table in standings])
    #         time.sleep(15)

    print(a.convert_table_to_csv("predict.txt", 2025))


if __name__ == '__main__':
    main()



#TODO
#Need proper API key in order to get all seasons worth of data, and then use it as necessary 