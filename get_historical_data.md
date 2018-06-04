

```python 
 def get_historical_stock_data(tar_dir = '../PySCT_Demo/data/'):
    df_all = pd.DataFrame()
    for i in os.listdir(tar_dir)[:1000]:
        temp_df = pd.read_csv(os.path.join(tar_dir, i),encoding = 'big5', header=None)
        temp_df.columns =  ['trading date' ,  'traded sum stockshard' , 'traded sum price' , 
                            'open price' ,'highest proce' , 'lowest price' , 'close price' ,
                            'price changed' , 'trading numbers' , ]
        temp_df['stock_id'] = i.split('.')[0]
        df_all = df_all.append(temp_df, ignore_index=True)
    return df_all 
 ```
