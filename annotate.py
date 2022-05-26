import pandas as pd
import os

if __name__ == '__main__':
    train = True
    if train:
        anime_path = 'data/anime/'
        human_path = 'data/humans/'
    else:
        anime_path = 'data/supertest/anime/'
        human_path = 'data/supertest/human/'
    anime_list = os.listdir(anime_path)
    human_list = os.listdir(human_path)
    dict = {}
    for item in anime_list:
        path = os.path.join(os.getcwd(), anime_path, item)
        dict[path] = 0
    for item in human_list:
        path = os.path.join(os.getcwd(), human_path, item)
        dict[path] = 1
    df = pd.DataFrame.from_dict(dict, orient='index')
    stage = 'train' if train else 'test' 
    df.to_csv(f'./data/anime_humans_{stage}.csv')
