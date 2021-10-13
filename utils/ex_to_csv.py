import pandas as pd
from csv import reader

if __name__ == '__main__':

    fname = '../data/maabarot_trima_1hr_data_05_10_2021.xlsx'
    df = pd.read_excel(fname)


    # df[df.columns[0].split(',')] = df[df.columns[0]].str.split(',', expand=True)
    df = df.drop([df.columns[0]], axis=1)
    print(df.head(10))

    df.to_csv('../data/maabarot_trima_1hr_data_05_10_2021.csv', index=None)