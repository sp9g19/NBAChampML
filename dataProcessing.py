import math

import pandas as pd
from scrapeStats import check_and_update_code
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modelling import train_and_predict_year

team_codes = ['ATL', 'BOS', 'NJN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM',
              'MIA', 'MIL', 'MIN', 'NOH', 'NYK', 'SEA', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

# winners = {2003: 'SAS', 2004: 'DET', 2005: 'SAS', 2006: 'MIA', 2007: 'SAS', 2008: 'BOS', 2009: 'LAL', 2010: 'LAL',
#            2011: 'DAL', 2012: 'MIA', 2013: 'MIA', 2014: 'SAS', 2015: 'GSW', 2016: 'CLE', 2017: 'GSW', 2018: 'GSW',
#            2019: 'TOR', 2020: 'LAL', 2021: 'MIL', 2022: 'GSW'}

team_names_to_codes = {'Atlanta Hawks': 'ATL',
                       'Boston Celtics': 'BOS',
                       'New Jersey Nets': 'NJN',
                       'Brooklyn Nets': 'BRK',
                       'Charlotte Bobcats': 'CHA',
                       'Charlotte Hornets': 'CHO',
                       'Chicago Bulls': 'CHI',
                       'Cleveland Cavaliers': 'CLE',
                       'Dallas Mavericks': 'DAL',
                       'Denver Nuggets': 'DEN',
                       'Detroit Pistons': 'DET',
                       'Golden State Warriors': 'GSW',
                       'Houston Rockets': 'HOU',
                       'Indiana Pacers': 'IND',
                       'Los Angeles Clippers': 'LAC',
                       'Los Angeles Lakers': 'LAL',
                       'Memphis Grizzlies': 'MEM',
                       'Miami Heat': 'MIA',
                       'Milwaukee Bucks': 'MIL',
                       'Minnesota Timberwolves': 'MIN',
                       'New Orleans Hornets': 'NOH',
                       'New Orleans Pelicans': 'NOP',
                       'New York Knicks': 'NYK',
                       'Seattle Supersonics': 'SEA',
                       'Oklahoma City Thunder': 'OKC',
                       'Orlando Magic': 'ORL',
                       'Philadelphia 76ers': 'PHI',
                       'Phoenix Suns': 'PHO',
                       'Portland Trailblazers': 'POR',
                       'Sacramento Kings': 'SAC',
                       'San Antonio Spurs': 'SAS',
                       'Toronto Raptors': 'TOR',
                       'Utah Jazz': 'UTA',
                       'Washington Wizards': 'WAS'}
team_codes_to_names = {v: k for k, v in team_names_to_codes.items()}

column_numbers_to_stat = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB',
                          'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'OMP', 'OFG', 'OFGA', 'OFG%', 'O3P',
                          'O3PA', 'O3P%', 'O2P', 'O2PA', 'O2P%', 'OFT', 'OFTA', 'OFT%', 'OORB', 'ODRB', 'OTRB', 'OAST',
                          'OSTL', 'OBLK', 'OTOV', 'OPF', 'OPTS', 'RSW', 'RSL', 'PyW', 'PyL', 'MOV', 'SOS',
                          'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'OeFG%',
                          'OTOV%', 'ODRB%', 'OFT/FGA', 'PlW']


def collect_season_stats():
    seasonStatsNp = None
    team_and_opp_stats_by_team = {}
    team_misc_stats_by_team = {}
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == 'NO CHA':
                seasonStatsNp = np.vstack((seasonStatsNp, np.empty(64)))
                continue
            if team in team_and_opp_stats_by_team.keys():
                team_and_opp_stats_by_team[team][year] = pd.read_csv(
                    'C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML'
                    '\\dataframes\\' + team + '\\' + year + '\\'
                    + team + year + '.csv',
                    skiprows=[1, 2, 4, 5, 6, 8],
                    usecols=range(3, 25)).astype(int).rename(
                    index={0: 'Own', 1: 'Opp'})
                team_misc_stats_by_team[team][year] = pd.read_csv(
                    'C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML'
                    '\\dataframes\\' + team + '\\' + year + '\\'
                    + team + year + 'Misc.csv',
                    skiprows=[0, 2],
                    usecols=range(2, 22)).astype(int)
            else:
                team_and_opp_stats_by_team[team] = {
                    year: pd.read_csv(
                        'C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML'
                        '\\dataframes\\' + team + '\\' + year + '\\'
                        + team + year + '.csv',
                        skiprows=[1, 2, 4, 5, 6, 8],
                        usecols=range(3, 25)).astype(int).rename(
                        index={0: 'Own', 1: 'Opp'})}
                team_misc_stats_by_team[team] = {
                    year: pd.read_csv(
                        'C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML'
                        '\\dataframes\\' + team + '\\' + year + '\\'
                        + team + year + 'Misc.csv',
                        skiprows=[0, 2],
                        usecols=range(2, 22)).astype(int)
                }
            if seasonStatsNp is None:
                seasonStatsNp = np.append(team_and_opp_stats_by_team[team][year].to_numpy().flatten(),
                                          team_misc_stats_by_team[team][year].to_numpy().flatten())
            else:
                seasonStatsNp = np.vstack((seasonStatsNp,
                                           np.append(team_and_opp_stats_by_team[team][year].to_numpy().flatten(),
                                                     team_misc_stats_by_team[team][year].to_numpy().flatten())))

    return seasonStatsNp, team_and_opp_stats_by_team, team_misc_stats_by_team


def print_season_dataframes(team_and_opp_dataframes, team_misc_dataframes):
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == 'NO CHA':
                continue
            print(team, year, ':\n',
                  'Team and Opp Stats:\n', team_and_opp_dataframes[team][year],
                  'Team Misc Stats:\n', team_misc_dataframes[team][year])


def collect_playoff_stats():
    playoff_dataframes_by_year = {}
    for y in range(3, 23):
        year = str(2000 + y)
        playoff_dataframes_by_year[year] = pd.read_csv('C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML'
                                                       '\\dataframes\\playoffs\\' + year + '.csv', skiprows=1,
                                                       index_col=0, usecols=[2, 4])
    return playoff_dataframes_by_year


def print_playoff_dataframes(playoff_dataframes):
    for y in range(3, 23):
        year = str(2000 + y)
        print(year, ' playoffs: \n', playoff_dataframes[year])


def collect_playoff_wins(playoff_dataframes):
    playoff_wins_by_team = {}
    playoff_wins_array = np.empty((600, 1))
    current_index = 0
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == 'NO CHA':
                playoff_wins_array[current_index, 0] = float('nan')
                current_index += 1
                continue
            if team == 'NOK':
                team_name = 'New Orleans Pelicans'
            else:
                team_name = team_codes_to_names[team]
            if team in playoff_wins_by_team.keys():
                if team_name in playoff_dataframes[year].index:
                    wins = playoff_dataframes[year].at[team_name, 'W']
                    playoff_wins_by_team[team][year] = wins
                    playoff_wins_array[current_index, 0] = wins
                else:
                    playoff_wins_by_team[team][year] = float('nan')
                    playoff_wins_array[current_index, 0] = float('nan')
            else:
                if team_name in playoff_dataframes[year].index:
                    wins = playoff_dataframes[year].at[team_name, 'W']
                    playoff_wins_by_team[team] = {year: wins}
                    playoff_wins_array[current_index, 0] = wins
                else:
                    playoff_wins_by_team[team] = {year: float('nan')}
                    playoff_wins_array[current_index, 0] = float('nan')
            current_index += 1
    return playoff_wins_by_team, playoff_wins_array


def print_playoff_wins_by_team(playoff_wins_dict):
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == 'NO CHA':
                continue
            print(team, year, ': \n', playoff_wins_dict[team][year])


def collect_playoff_team_stats_by_year(team_and_opp_stats_by_team, team_misc_stats_by_team, playoff_wins_by_team):
    stats_by_year = {}
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == 'NO CHA':
                continue
            if math.isnan(playoff_wins_by_team[team][year]):
                continue
            if year in stats_by_year.keys():
                stats_by_year[year][team] = np.concatenate((team_and_opp_stats_by_team[team][year].to_numpy().flatten(),
                                                            team_misc_stats_by_team[team][year].to_numpy().flatten(),
                                                            np.array([playoff_wins_by_team[team][year]])))
            else:
                stats_by_year[year] = {team: np.concatenate((team_and_opp_stats_by_team[team][year].to_numpy().flatten(),
                                                             team_misc_stats_by_team[team][year].to_numpy().flatten(),
                                                             np.array([playoff_wins_by_team[team][year]])))}
    return stats_by_year


def print_stats_by_year(stats_by_year_dict):
    for y in range(3, 23):
        year = str(2000 + y)
        for team, stats in stats_by_year_dict[year].items():
            print(year, team, ': \n', stats)


def test_on_all_years(stats_by_year, stats_to_use):
    for y in range(3, 23):
        year = str(2000 + y)
        train_and_predict_year(stats_by_year, year, stats_to_use)


seasonStatsNp1, team_and_opp_stats_by_team1, team_misc_stats_by_team1 = collect_season_stats()
# print_season_dataframes(team_and_opp_stats_by_team1, team_misc_stats_by_team1)
playoff_dataframes_by_year1 = collect_playoff_stats()
# print_playoff_dataframes(playoff_dataframes_by_year1)
playoff_wins_by_team1, playoff_wins_array1 = collect_playoff_wins(playoff_dataframes_by_year1)
# print_playoff_wins_by_team(playoff_wins_by_team1)
statsWithWinsDf = pd.DataFrame(np.hstack((seasonStatsNp1, playoff_wins_array1)),
                               columns=column_numbers_to_stat).dropna()
statsWithWinsDf.to_csv("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\Training.csv")
winCorrs = statsWithWinsDf.astype('float64').corr()['PlW']
# print(winCorrs[winCorrs < -0.25])
# sns.barplot(x=(winCorrs[winCorrs < -0.25]*-1).index, y=(winCorrs[winCorrs < -0.25]*-1), palette='crest')
# plt.xticks(rotation=90)
# plt.xlabel('Regular Season Statistic Ranking')
# plt.ylabel('Pearson Correlation With Playoff Wins')
# plt.title('Pearson Correlation Between League Ranking in \nEach Category With Playoff Wins')
# plt.tight_layout()
# plt.show()
# # print(winCorrs[winCorrs < -0.25].index.tolist())
# meansByPlWDf = \
#     statsWithWinsDf.groupby('PlW', as_index=False)[winCorrs[winCorrs < -0.25].index.tolist()].mean().iloc[:, 1:]
# sns.heatmap(meansByPlWDf, annot=True)
# plt.xlabel('Regular Season Stat Rankings')
# plt.ylabel('Playoff Wins')
# plt.title('Average League Ranking in Each Category,by Playoff Wins')
# plt.tight_layout()
# plt.show()

stat_names = winCorrs.index.tolist()
useful_stat_names = winCorrs[winCorrs < -0.25].index.tolist()
# print(stat_names)
# print(useful_stat_names)
selected_stat_idxs = [stat_names.index(s) for s in useful_stat_names] + [len(stat_names)-1]
# print(selected_stat_idxs)
stats_by_year1 = collect_playoff_team_stats_by_year(team_and_opp_stats_by_team1, team_misc_stats_by_team1, playoff_wins_by_team1)
# print_stats_by_year(stats_by_year1)
test_on_all_years(stats_by_year1, selected_stat_idxs)
