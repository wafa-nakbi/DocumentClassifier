import numpy as np
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import pickle

data_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42)
y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)
#fit the model
model=RidgeClassifier(tol=1e-2, solver="sag")
model.fit(X_train, y_train)
pred = model.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
#save the model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(vectorizer, open('vectorizer.pickle', 'wb'))
