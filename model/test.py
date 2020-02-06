import pickle
import matplotlib.pyplot as plt
import numpy as np

categories=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
d={}
for i in range(20):
    d[i]=categories[i]
filename = 'finalized_model.sav'
vectorizer_name='vectorizer.pickle'
model = pickle.load(open(filename, 'rb'))
vectorizer=pickle.load(open(vectorizer_name, 'rb'))
X_test=vectorizer.transform([''])
pred = model.predict(X_test)
print ('Class',d[pred[0]])
pb = model.decision_function(X_test)[0]
probs = np.exp(pb) / np.sum(np.exp(pb))
fig1, ax1 = plt.subplots()
ax1.pie(probs, labels=categories, autopct='%1.1f%%',
            shadow=True, startangle=90)
ax1.axis('equal')
fig1.savefig('../app/static/images/new_plot.png')