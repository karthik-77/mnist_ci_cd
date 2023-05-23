from tensorflow.keras.utils import to_categorical as tc
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

def split_dataset():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    return (x_train,y_train),(x_test,y_test)

def reshape_data(x_train,x_test):
    img_rows,img_cols,channels=28,28,1
    if K.image_data_format()== 'channels_first':
        x_train=x_train.reshape(x_train.shape[0],channels,img_rows,img_cols)
        x_test=x_test.reshape(x_test.shape[0],channels,img_rows,img_cols)
        input_shape=(channels,img_rows,img_cols)
    else:
        x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,channels)
        x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,channels)
        input_shape=(img_rows,img_cols,channels)
    return x_train,x_test,input_shape


def normalize_data(x_train,x_test):
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train/=255
    x_test/=255
    return x_train,x_test

def encode_labels(y_train,y_test,num_classes):
    num_classes=10
    y_train=tc(y_train,num_classes)
    y_test=tc(y_test,num_classes)
    return y_train,y_test

def build_model(input_shape,num_classes):
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),activation='tanh',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='sigmoid'))
    model.add(Dense(84,activation='sigmoid'))
    model.add(Dense(num_classes,activation='softmax'))
    return model

def model_compile_train(model,epochs,batch_size,x_train,y_train,x_test,y_test):
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))
    return model

def predict(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss score: {score[0]}')
    print(f'Test accuracy score:{score[1]}')
    with open("results.txt",'w') as fp:
        fp.write(f'Test loss score: {score[0]}')
        fp.write('\n')
        fp.write(f'Test accuracy score:{score[1]}')



if __name__=='__main__':
    (x_train,y_train),(x_test,y_test)=split_dataset()
    x_train,x_test,input_shape=reshape_data(x_train,x_test)
    x_train,x_test=normalize_data(x_train,x_test)
    num_classes=10
    y_train,y_test=encode_labels(y_train,y_test,num_classes)
    model=build_model(input_shape,num_classes)
    print(f'Model Summary :{model.summary()}')
    num_epochs=20
    batch_size=128
    model=model_compile_train(model,num_epochs,batch_size,x_train,y_train,x_test,y_test)
    predict(model,x_test,y_test)






