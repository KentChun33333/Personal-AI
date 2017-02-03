from yahoo_finance import Share
import pandas as pd
import datetime

def getStock(id):
    stock = Share(str(id)+'.TW')
    today = datetime.date.today()
    data = stock.get_historical('2016-12-01', str(today))
    return data


if __name__=='__main__':
    data =  getStock(1517)
    df = pd.DataFrame(data)
