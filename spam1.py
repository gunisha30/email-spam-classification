import pandas as pd
import nltk
import time
#nltk.download('stopwords')
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support as score

df = pd.read_csv("C:\\Users\\hp1\\Documents\\coursera\\ml\\spamd\\spam.csv")

#frequency of ham and spam
pd.crosstab(df['Label'],columns='freq')

#stemming
ps = nltk.PorterStemmer()

#list of stopwords
stopwords = nltk.corpus.stopwords.words('english')

#returns percentage of punctuations in text
def punct_pc(text):
    punct_count = sum([1 for char in text if char in string.punctuation])
    return (punct_count/(len(text) - text.count(' ')))*100

#creating two new features
df['text_length'] = df['EmailText'].apply(lambda x : len(x)-x.count(' '))
df['punct'] = df['EmailText'].apply(lambda x : punct_pc(x))

#remove punctuations
def clean_data(text):
    punct = "".join([word.lower() for word in text if word not in string.punctuation])
    splt = re.split('\W+',punct)
    txt = [ps.stem(word) for word in splt if word not in stopwords]
    return txt

x_train,x_test,y_train,y_test = train_test_split(df[['EmailText','text_length','punct']],df['Label'],test_size=0.2,random_state=123)

#vectorising using count vectorizer
vect=CountVectorizer(analyzer=clean_data)

#fit- done on training data, tranform- applied to training and testing data
vectfit = vect.fit(x_train['EmailText'])
xtrainvect = vectfit.transform(x_train['EmailText'])
xtestvect = vectfit.transform(x_test['EmailText'])
xtrainvect= pd.concat([x_train[['text_length','punct']].reset_index(drop=True),pd.DataFrame(xtrainvect.toarray())],axis=1)
xtestvect=pd.concat([x_test[['text_length','punct']].reset_index(drop=True),pd.DataFrame(xtestvect.toarray())],axis=1)

#classifier- random forest
rf = RandomForestClassifier(random_state=123,n_jobs=1)

#get the list of hyperparameters, n_estimators=num of trees in the forest, max_features= num of features considered for splitting
print(rf.get_params())

param = {'n_estimators' : [10,25,50,100,300], 'max_depth' : [10, 20, 50,100, None],'max_features' : [10,50,'auto']}
grid = GridSearchCV(rf,param,cv=5,n_jobs=4)

rfgrid = grid.fit(xtrainvect, y_train)
pd.DataFrame(rfgrid.cv_results_).sort_values('mean_test_score',ascending=False)[0:10]


print(rfgrid.best_params_) #output:max_depth=100, max_features=auto,n_estimators=300
rf_final_3 = RandomForestClassifier(n_estimators = 50, max_depth = 100, max_features='auto' , n_jobs=-1,random_state=123)

start = time.time()
rf_model_3 = rf_final_3.fit(xtrainvect, y_train)
end = time.time()
fit_time = end - start

start = time.time()
Y_pred = rf_model_3.predict(xtestvect)
end = time.time()
predict_time = end-start

precision, recall, fscore, train_support = score(y_test, Y_pred, pos_label='spam', average='binary')
print('Fit_time : {} / Predict_time : {} / Precision: {} / Recall: {} / Accuracy: {}'.format(round(fit_time,3),round(predict_time,3),
round(precision, 3), round(recall, 3), round((Y_pred==y_test).sum()/len(Y_pred), 3)))
