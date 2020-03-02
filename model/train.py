import numpy as np
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn import metrics
import pickle

#load data set
data_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes'))
y_train, y_test = data_train.target, data_test.target
#Base learners for the StackingClassifier
base_learners = [
                 ('rf_1', RidgeClassifier(tol=1e-2, solver="sag")),
                 ('rf_2', MultinomialNB(alpha=0.01))
                ]
#Pipeline: extract features from training data then define model
model = Pipeline([('tfid', TfidfVectorizer(sublinear_tf=True,stop_words='english')), ('clf',StackingClassifier(estimators=base_learners) )])
model.fit(data_train.data, y_train)
pred = model.predict(data_test.data)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
print ('f1 score: %0.3f' % metrics.f1_score(y_test, pred,average='macro'))
#save the model
pickle.dump(model, open('finalized_model.sav', 'wb'))
#pickle.dump(vectorizer, open('vectorizer.pickle', 'wb'))
