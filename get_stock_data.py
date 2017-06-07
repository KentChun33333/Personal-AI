from yahoo_finance import Share
import pandas as pd
import datetime


STOCK_ref  =  'tw_stock_data/all_stock_201701.txt'
OPTION_ref = 'tw_stock_data/all_stock_options_201702.txt'

def get_stock(id):
    stock = Share(str(id)+'.TW')
    today = datetime.date.today()
    data = stock.get_historical('2017-01-01', str(today))
    return data

def get_id(address):
    with open(address, 'r') as f :
        for line in f:
            id = line.split(' ')[0]
            try:
                assert len(id)== 4 or 6
                yield id
            except Exception as e :
                print (e)




if __name__=='__main__':
    # test :: pass
    # data =  getStock(1517)
    # df = pd.DataFrame(data)

    # test :: pass
    for i in get_id(STOCK_ref):
        print (get_stock(i))
    for i in get_id(OPTION_ref):
        print (i)
