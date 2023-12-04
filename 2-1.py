import pandas as pd
from pandas import Series, DataFrame


data=pd.read_csv('/Users/gomsuman/DownLoads/2019_kbo_for_kaggle_v2.csv')
data_filter = data[data['year'].between(2015, 2018)]

def get_top_player(stat, year):
    year_data = data_filter[data_filter['year'] == year]
    return year_data.nlargest(10, stat)[['batter_name', 'year', stat]]

top_player = {}

for year in range(2015, 2019):
    top_player[year] = {
        'H': get_top_player('H', year),
        'avg': get_top_player('avg', year),
       'HR': get_top_player('HR', year),
        'OBP': get_top_player('OBP', year)
    }

print(top_player[2015])
print(top_player[2016])
print(top_player[2017])
print(top_player[2018])

data_2018 = data[data['year'] == 2018]
top_war_players_2018 = data_2018.loc[data_2018.groupby('cp')['war'].idxmax()]

print(top_war_players_2018[['batter_name', 'cp', 'war']])


correlations = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()['salary']
correlations = correlations.drop('salary') 

highest_correlation_statistic = correlations.idxmax()
highest_correlation_value = correlations.max()

print(highest_correlation_statistic)





