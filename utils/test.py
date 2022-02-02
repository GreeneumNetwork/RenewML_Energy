from data import Data


if __name__ == '__main__':
    dat = Data.get_data(datafile='data/4Y_Historical.csv',
                         powerfile='data/maabarot_trima_15min.csv')
    transformed = dat.transform(lag='day')
    inv = transformed.inverse_transform(transformed.df)
    print(inv)



