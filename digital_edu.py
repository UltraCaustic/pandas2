import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')

print(df2.info())
print(df.info())

pd.isnull(df["relation"])
print(pd.isnull(df))

s=df[df["result"]==1]["education_form"].value_counts()




print(s)
s.plot(kind = 'pie', grid = True)

plt.show()
education=df[df['result']==1]['education_form'].value_counts()
print(education)
education.plot(kind = 'pie', grid = True)

plt.show()

pd.get_dummies(df['education_form'])
print(pd.get_dummies(df['education_form']))
df[list(pd.get_dummies(df['education_form']).columns)] =pd.get_dummies(df['education_form'])
df.drop('education_form', axis = 1, inplace = True)

pd.get_dummies(df2['education_form'])
print(pd.get_dummies(df2['education_form']))
df2[list(pd.get_dummies(df2['education_form']).columns)] =pd.get_dummies(df2['education_form'])
df2.drop('education_form', axis = 1, inplace = True)

#преобразование столбца occupation_type - текущее занятие пользователя (школа, университет, работа) в три фиктивых переменные
pd.isnull(df['occupation_type'])
pd.isnull(df2['occupation_type'])
education = df[df['result']==1]['occupation_type'].value_counts()
print(education)

pd.get_dummies(df['occupation_type'])
print(pd.get_dummies(df['occupation_type']))
df[list(pd.get_dummies(df['occupation_type']).columns)]=pd.get_dummies(df['occupation_type'])
df.drop('occupation_type',axis = 1, inplace = True)



df.drop(['bdate','has_photo','has_mobile','followers_count','graduation','relation','education_status','langs','life_main','people_main','city','last_seen','occupation_name','career_start','career_end'],axis = 1, inplace = True)
pd.isnull(df['id'])
df2.drop(['bdate','has_photo','has_mobile','followers_count','graduation','relation','education_status','langs','life_main','people_main','city','last_seen','occupation_name','career_start','career_end'],axis = 1, inplace = True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
X = df.drop('result', axis = 1)
y = df['result']
X_train=X
y_train=y
X_test=df2
y_test=df.tail(len(df2))
y_test=y_test['result']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))


ID = df2['id']

result = pd.DataFrame({'id': ID, 'result':y_pred})
result.to_csv('res4.csv', index = False)








































































