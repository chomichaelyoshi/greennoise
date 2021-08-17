#練習用　２０２１０６１８
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

from PIL import Image
import glob 
from sklearn.model_selection import train_test_split
from numpy import eye
import pickle
import csv
data_root_dir="c:/GNdata/"
folder=["00","01","10","11","H"]
img_size=16

x=[]
t=[]

for index,name in enumerate(folder):
    print(name)
    dir=data_root_dir + name
    print(dir)
    files=glob.glob(dir+"/*.bmp")

    for i,file in enumerate(files):
        img = Image.open(file)

        img=img.convert("L")
        img = img.resize((img_size,img_size))

        data = np.asarray(img)
        x.append(data)
        t.append(index)

x=np.array(x)
t=np.array(t)
x=x/255.0

print("読み取りデータの形")
print(x.shape)
print(t.shape)
print(len(x))

x=x.reshape(len(x),-1)
print("reshape後のデータ")
print(x.shape)

x_train,x_test,t_train,t_test=train_test_split(x,t,test_size=0.2)

#one-hotラベルの作成
t_train=np.eye(5)[t_train]
t_test=np.eye(5)[t_test]

print("split後のx,t確認")
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

print(t_test[0])
print(t_test[1])


network = TwoLayerNet(input_size=256,hidden_size=50,output_size = 5)

iters_num = 10000
train_size = x_train.shape[0]
batch_size =100
learning_rate =0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch =max(train_size/batch_size,1)


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


#パラメータの保存
network.save_params("parmas.pkl")
print("セーブしました")

#グラフの描画
markers={'train':'o','test':'s'}
x=np.arange(len(train_acc_list))
plt.plot(x,train_acc_list,label='train acc')
plt.plot(x,test_acc_list,label='test acc',linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0,1.0)
plt.legend(loc='lower right')
plt.show
