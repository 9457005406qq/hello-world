
from tensorflow.keras.layers import LSTM, Dense,Dropout
import pandas as pd
import numpy as np
data = pd.read_csv('zgpa_train.csv')
# data.dropna(axis=0, how="any") # 传入这个参数后将只丢弃全为缺失值的那些行
# 归一化处理
data.fillna(data.mean(),inplace=True)
price = data.loc[:,'close']
price.head()

#归一化处理
price_norm = price/max(price)
print(price_norm)

#可视化
from matplotlib import pyplot as plt
# fig1=plt.figure(figsize=(8,5))#建立图形
# plt.plot(price)#画图
#加标注
# plt.title('close price')
# plt.xlabel('time')
# plt.ylabel('price')
# plt.show()

#xy的赋值

#define X and y
#define method to extract X and y

#提取数据序列
def extract_data(data,time_step):
    X=[]
    y=[]
    #0,1,2,3...9:10个样本；time_step=8;0,1,2..7;12,3,...9
    for i in range(len(data)-time_step):
        X.append([a for a in data[i:i+time_step]])#X数组是训练集
        y.append(data[i+time_step])#y是预测集
    X = np.array(X)#转换为一个数组
    X = X.reshape(X.shape[0],X.shape[1],1)
    return X,y

time_step = 1
X,y = extract_data(price_norm,time_step)
print("Xshape:")
print(X.shape)
print(y)

from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
model = Sequential()

# #单层有五个神经元units，input_shape(样本数，序列长度，序列维度）:数据是一维的，activation：激活函数
# model.add(SimpleRNN(units = 5, input_shape=(time_step,1),activation='relu'))
# model.add(Dense(units=1,activation='linear'))

model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1,activation='linear'))


#(优化器，损失函数）
#平方差mean_squared_errar
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

#模型训练
#打印X，y的维度
#print(X.shape(X),len(y))
#训练样本，每次60个样本，共训练200次
y=np.array(y);
model.fit(X,y,batch_size=60,epochs=200)


#预测结果可视化
y_train_predict = model.predict(X)*max(price)
#y_train = [i*max(price) for i in y]#把归一化之后的数值转换过来
y_train = y*max(price)
print(y_train_predict,y_train)

# fig2=plt.figure(figsize=(8,5))#建立图形
# plt.plot(y_train,label='real price')#画图
# plt.plot(y_train_predict,label='predict price')#画图
# #加标注
# plt.title('close price')
# plt.xlabel('time')
# plt.ylabel('price')
# plt.legend()
# plt.show()


#predict

data_test = pd.read_csv('zgpa_test.csv')
data_test.head()
price_test = data_test.loc[:,'close']
price_test.head()
price_test_norm = price_test/max(price)#归一化
X_test_norm,y_test_norm = extract_data(price_test_norm,time_step)
print(X_test_norm.shape,len(y_test_norm))

y_test_predict = model.predict(X_test_norm)*max(price)
y_test_norm=np.array(y_test_norm);
y_test = y_test_norm*max(price)

time=data_test['date'][:19]
pd.to_datetime(time)

# import matplotlib.dates as mdates    #處理
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  #設置x軸主刻度顯示格式（日期）
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  


fig3=plt.figure(figsize=(12,6))#建立图形
plt.plot(time,y_test,label='real price test')#画图
plt.plot(time,y_test_predict,label='predict price test')#画图
#加标注
plt.xticks(rotation=-90) 
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

for i in range(0,len(y_test_predict)):
    totforenum=abs(y_test_predict[i]-y_test[i]);
    totnum=y_test[i];
totnum/=len(y_test_predict);
totforenum/=len(y_test_predict);
print("平均误差率MAE:",end="");
print(totforenum/totnum*100,end="");
print("%");


#存储数据
# result_y_test = np.array(y_test).reshape(-1,1)
# result_y_test_predict = y_test_predict
# print(result_y_test.shape,result_y_test_predict.shape)
# result = np.concatenate((result_y_test,result_y_test_predict),axis = 1)
# print(result.shape)
# result = pd.DataFrame(result,columns=['real_price_test','predict_price_test'])
# result.to_csv('zapa_predict_close.csv')
