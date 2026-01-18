import requests
import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import ast
from data_aquisition import FootballDataAPI


class Prepare_Data:
    def __init__(self, df = None):
        self.df_ = df

    #assign win or no win for the season label to the dataframe
    def assign_label(self, cur_season = None):
        df = self.df_.copy()
        df["games_remaining"] = 38 - df["gameweek"]
        if not cur_season:
            df["won_league"] = 0
            a = FootballDataAPI()
            for season in df["season"].unique():
                winner_id = a.determine_season_winner(season)
                df.loc[
                    (df["season"] == season) & (df["teamID"] == winner_id),
                    "won_league"
                    ] = 1
        # print(type(df["form"]))
        df["form"] = df["form"].apply(self.numerical_form)
        self.df_ = df
    
    def numerical_form(self, form_string):
        """
        Convert form string to numerical representation.
        Win (W) = 3, Draw (D) = 1, Loss (L) = 0
        """
        if pd.isna(form_string) or form_string is None:
            return 0.0
        
        # Convert to string and clean up
        form_str = str(form_string).strip()
        if form_str == "":
            return 0.0
        
        total = 0
        # Go through each character in the string
        for char in form_str:
            if char.upper() == 'W':
                total += 3
            elif char.upper() == 'D':
                total += 1
            elif char.upper() == 'L':
                total += 0
        
        return float(total)


    def get_data(self):
        return self.df_



def main():
    df = pd.read_csv("prem_table_2025.csv")
    p_Data = Prepare_Data(df)
    p_Data.assign_label(False)
    df_store = p_Data.get_data()
    df_store.to_csv("test.csv")


if __name__ == '__main__':
    main()