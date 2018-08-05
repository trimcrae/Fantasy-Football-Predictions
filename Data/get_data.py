# Create a SQL database file called 'football_database'
import sqlite3
import pandas as pd


def drop_table(table, con):
    con.execute(f'DROP TABLE IF EXISTS {table}')


def get_data():
    links = {}
    prefix = r'https://raw.githubusercontent.com/ryurko/nflscrapR-data/master/data'

    z = f'{prefix}/game_player_stats'
    links['game_player_passing'] = f'{z}/game_passing_df.csv'
    links['game_player_receiving'] = f'{z}/game_receiving_df.csv'
    links['game_player_rushing'] = f'{z}/game_rushing_df.csv'

    z = f'{prefix}/game_team_stats'
    links['game_team_passing'] = f'{z}/game_passing_df.csv'
    links['game_team_receiving'] = f'{z}/game_receiving_df.csv'
    links['game_team_rushing'] = f'{z}/game_rushing_df.csv'

    z = f'{prefix}/season_player_stats'
    links['season_player_passing'] = f'{z}/season_passing_df.csv'
    links['season_player_receiving'] = f'{z}/season_receiving_df.csv'
    links['season_player_rushing'] = f'{z}/season_rushing_df.csv'

    z = f'{prefix}/season_team_stats'
    links['season_team_def_passing'] = f'{z}/team_def_season_passing_df.csv'
    links['season_team_def_receiving'] = f'{z}/team_def_season_receiving_df.csv'
    links['season_team_def_rushing'] = f'{z}/team_def_season_rushing_df.csv'
    links['season_team_off_passing'] = f'{z}/team_season_passing_df.csv'
    links['season_team_off_receiving'] = f'{z}/team_season_receiving_df.csv'
    links['season_team_off_rushing'] = f'{z}/team_season_rushing_df.csv'

    seasons = range(2009, 2018)
    for season in seasons:
        links[f'season_games_{season}'] = f'{prefix}/season_games/games_{season}.csv'
        links[f'season_pbps_{season}'] = f'{prefix}/season_play_by_play/pbp_{season}.csv'
        links[f'season_team_rosters_{season}'] = f'{prefix}/team_rosters/team_{season}_rosters.csv'
    
    return links


if __name__ == '__main__':
    conn = sqlite3.connect('football_database')
    c = conn.cursor()

    links = get_data()
    for table_name, csv in links.items():
        try:
            drop_table(table_name, c)
            df = pd.read_csv(csv)
            df.to_sql(name=table_name, con=conn, index=False)
        except:
            print(f'\nFailed to make the table {table_name} at the following link:\n\t{csv}\n\n')

    print('Successfully created the following tables:')
    print('\n'.join(links.keys()))

    c.close()
