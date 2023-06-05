import  pandas as pd
import numpy as np
from data import status1,gender1
# close=pd.read_csv('zapa_predict_close.csv')
# high=pd.read_csv('zapa_predict_high.csv')
# low=pd.read_csv('zapa_predict_low.csv')
# open=pd.read_csv('zapa_predict_open.csv')
# volume=pd.read_csv('zapa_predict_volume.csv')
# ori=pd.read_csv('zgpa_test.csv')

# ori=ori['date']
# close=close['predict_price_test']
# high=high['predict_price_test']
# low=low['predict_price_test']
# open=open['predict_price_test']
# volume=volume['predict_price_test']


# data=pd.concat([ori,open,high,low,close,volume],axis=1)
# df=pd.DataFrame(data)
# df.columns={'date':ori,'open':open,'high':high,'low':low,'close':close,'volume':volume}
# df.to_csv('1.csv')
# print(df)

import pandas as pd
import numpy as np
data = pd.read_csv('zgpa_train.csv')
# data.dropna(axis=0, how="any") # 传入这个参数后将只丢弃全为缺失值的那些行
# data=data[:713]
# 归一化处理
data.fillna(data.mean(),inplace=True)
price = data.loc[:,'close']
price.head()

