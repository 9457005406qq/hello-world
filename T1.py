import pandas as pd
import numpy as np

data = pd.read_csv("newtrain1.csv")

print(data)

price = data.loc[:,'close']
print(price)
#归一化处理
price_norm = price/max(price)
print(price_norm)

#可视化归一化的price数据
from matplotlib import pyplot as plt
fig1 = plt.figure(figsize=(18,20))
plt.subplot(121)
plt.plot(price)#归一化前
plt.subplot(122)
plt.plot(price_norm)#归一化后
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()

#define X and y
#define nethod to extract X and y

def extract_data(data,time_step):
    X = []
    y = []
    #0,1,2,3....9 :10个样本 time_step=8; 0-7,1-8,2-9 三组
    for i in range(len(data) - time_step ):
        X.append([a for a in data[ i: i+time_step ]])
        y.append(data[i+time_step])
    X = np.array(X)
    X = X.reshape(X.shape[0],X.shape[1],1)#维度1
    return X,y

#define time_step
time_step = 1

#defin X and y

X,y = extract_data(price_norm,time_step)
print(X.shape) #(723, 8, 1)
print(len(y))

y = np.array(y)  #核心，必须要转换y的类型，不然会报错

# set up model
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
model = Sequential()
#input_shape 训练长度 每个数据的维度
model.add(SimpleRNN(units=5,input_shape=(time_step,1),activation="relu"))
#输出层
#输出数值 units =1 1个神经元 "linear"线性模型
model.add(Dense(units=1, activation="linear"))
#配置模型 回归模型y
model.compile(optimizer="adam",loss="mean_squared_error")
model.summary()

#train the model
model.fit(X,y,batch_size=30,epochs=200)

#预测 训练数据
y_train_predict = model.predict(X) * max(price)
y_train = [i*max(price) for i in y]#归一化数据转换回来

print(y_train_predict,y_train)

#画出预测的训练结果
fig2 = plt.figure(figsize=(10,5))
plt.plot(y_train,label = "real price")
plt.plot(y_train_predict,label = "predict price")
plt.title("price")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.show()

#用训练好的模型对测试的数据进行预测
#生成数据

# 读取中国平安（601318）数据
# zgpa = ts.get_hist_data('601318', start='2020-01-01', end='2020-07-24')
zgpa=pd.read_csv('zgpa_train(1).csv')
# 查看数据
print(zgpa)

data_test = zgpa
price_test = data_test.loc[:,'close']
price_test.head()

price_test_norm = price_test/max(price) #归一化操作
X_test_norm,y_test_norm = extract_data(price_test_norm,time_step)
print(X_test_norm.shape,len(y_test_norm))

#对测试数据进行预测
y_test_predict = model.predict(X_test_norm)*max(price)
y_test = [i*max(price) for i in y_test_norm]

fig3 = plt.figure(figsize=(10,5))
plt.plot(y_test,label = "real price test")
plt.plot(y_test_predict,label = "predict price test")
plt.title("price")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.show()


#储存结果数据
result_y_test = np.array(y_test).reshape(-1,1)
result_y_test_predict = y_test_predict
print(result_y_test.shape,result_y_test_predict.shape)
result = np.concatenate((result_y_test,result_y_test_predict),axis=1)#合并结果
print(result.shape)
reslut = pd.DataFrame(result,columns=['real_predict_test','predict_price_test'])
reslut.to_csv("zgpa_predict.csv")