#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

#[라이브러리 Import]

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mlt

#[라이브러리 설치]

#!pip install 설치라이브러리명

#[데이터(파일) 불러오기]

#데이터프레임명 = pd.read_csv('파일명.csv')
#*파일명은 반드시 확장자까지!!!! 

'../data/파일명.csv'

#%%
"""
[데이터전처리]
0. 결측치처리
데이터프레임명.isnull().sum()

1. 칼럼 제거
데이터프레임명.drop(columns=['칼럼명'], inplace=True)

2. 특정 칼럼의 값 변환
데이터프레임명['칼럼명'].replace('A', 'B', inplace = True)

3. 칼럼의 타입 변환
데이터프레임명['칼럼명']=데이터프레임명['칼럼명'].astype(타입)

* 정수형 : int , 실수형 : float, 문자열 : str 

4. 독립변수와 종속변수의 분리
독립변수명=데이터프레임명.drop(columns=['종속변수명'])
종속변수명(데이터)=데이터프레임명['종속변수명']

* 종속변수 : 예측하고자 하는 데이터, 독립변수 : 그외

5. shape 출력 / 데이터의 행과 열 수 출력
데이터프레임명(시리즈).shape 

6. 데이터의 분리
from sklearn.model_selection import train_test_split
X_train , X_valid , y_train, y_valid = train_test_split(독립명수명,종속변수명,stratify = 문제조건, test_size = 문제조건, random_state=10)


*변수 주의 : 문제에서 주어짐
*학습 데이터 및 검증 데이터 비율 주의 : 문제에서 주어짐

7. 더미처리(Pandas에서 제공해주는걸로)
데이터프레임명 = pd.get_dummies(데이터프레임명, columns=['칼럼1','칼럼2'], drop_first=True)

"""
#%%

#8. LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_list = ['칼럼1','칼럼2']
for col in label_list:
    데이터프레임명[col] = le.fit_transform(데이터프레임명[col])

#9. 스케일링
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns )
X_valid = pd.DataFrame(mms.transform(X_valid), columns=X_valid.columns)


# StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train))
X_valid = pd.DataFrame(ss.transform(X_valid))

# RobustScaler
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
X_train = pd.DataFrame(rbs.fit_transform(X_train))
X_valid = pd.DataFrame(rbs.transform(X_valid))

#%%

"""
[AI모델링]
* 학습데이터 및 검증데이터의 변수명 주의
* 하이퍼파라미터 변경 : 문제에서 주어짐
"""
1. kNN
#라이브러리 Import
from sklearn.neighbors import KNeighborsClassifier

#모델 생성
knn = KNeighborsClassifier(n_neighbors=?)

#모델 학습
knn.fit(X_train, y_train)

#모델 성능 평가
score=knn.score(X_valid, y_valid)
print(score)


2. LogisticRegression

#라이브러리 Import
from sklearn.linear_model import LogisticRegression

#모델 생성
lg=LogisticRegression(C=?, max_iter=?)

#모델 학습
lg.fit(X_train, y_train)

#모델 성능 평가
score = lg.score(X_valid, y_valid)
print(score)

3. Decision Tree 

#라이브러리 Import
from sklearn.tree import DecisionTreeClassifier

#모델 생성
tree=DecisionTreeClassifier(max_depth=?, random_state=?)

#모델학습
tree.fit(X_train, y_train)

#모델 성능평가
score = tree.score(X_valid, y_valid)
print(score)


4. RandomForest

#라이브러리 Import 
from sklearn.ensemble import RandomForestClassifier

#모델생성
rf =RandomForestClassifier(n_estimators=?, random_state=?)

#모델학습
rf.fit(X_train, y_train)

[모델 성능평가(일반적인 Accuracy 확인)]
score = rf.score(X_valid, y_valid)
print(score)

[모델 성능평가(Confusion_matrix)]
# Confusion_matrix
from sklearn.metrics import confusion_matrix 

y_pred = 모델명.predict(X_valid)
cm=confusion_matrix(y_valid, y_pred)

sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(cm,
            annot=True)

[모델 성능평가(classification_report)]
# classification_report
from sklearn.metrics import classification_report
y_pred = 모델명.predict(X_valid)
print(classification_report(y_valid, y_pred))



#%%
5. 딥러닝(DNN)
* 파라미터 설정 주의 : 문제에서 주어짐

#Deep Neural Network

#라이브러리 Import
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#하이퍼파라미터 설정
batch_size = 1024
epochs = 20

#callbacks
es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)

#모델 생성
model = Sequential()

model.add(Dense(128,activation='relu',input_shape=(19,)))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

#One-Hot-Encoding
y_train_ohe = to_categorical(y_train)
y_valid_ohe = to_categorical(y_valid)

#모델 학습
history = model.fit(X_train, y_train_ohe, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    callbacks=[es,mc], 
                    validation_data=(X_valid, y_valid_ohe), 
                    verbose=1)



#이중 분류일 떈 binary_crossentropy
#다중 분류일떈 categorical_crossentropy


[DNN epoch에 따른 acc, loss그래프 그리기]
plt.figure(figsize=(8,6))
plt.plot (history.history['acc'])
plt.plot (history.history['loss'])
plt.plot (history.history['val_acc'])
plt.plot (history.history['val_loss'])

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc','loss','val_acc','val_loss'],loc='lower right')


[상관관계 히트맵 그리기]
* 그래프에 상관계수 포함 : annot = True // 미포함 annot = False 혹은 그냥 제거

sns.set(rc={'figure.figsize':(15,15)})
sns.heatmap(데이터프레임명.corr(), annot = True )

[히스토그램 그리기]
# matplotlib.pyplot사용
plt.hist(데이터프레임명['칼럼명'])

[Barplot]
# seaborn 사용
sns.barplot(x=데이터프레임명['X축칼럼명'], y=데이터프레임명['Y축칼럼명'])

