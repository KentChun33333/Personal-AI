
def read_data(data_dir='../PySCT_Demo/data/'):
    cols = ['trading date', 'traded sum stockshard' , 'traded sum price' , 
            'open price', 'highest proce', 'lowest price' , 
            'close price', 'price changed', 'trading numbers' ]

    df_list=[]
    for stock_id in os.listdir(data_dir):
        try:
            df = pd.read_csv(data_dir + stock_id, header=None, encoding = "ISO-8859-1" )
            df.columns = cols
            df['stock_id'] = stock_id.split('.')[0]
        except:
            print(stock_id)
        
    df_list.append(df)
    return pd.concat(df_list)
