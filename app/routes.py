from app import app
from flask import render_template,request,flash
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



@app.route('/')
@app.route('/index',methods=['GET', 'POST'])
def index():
    error=None
    text=request.form.get("text")
    if not text:
        error="Please enter text to classify"
        return render_template('index.html', pred_class='', error=error,plot=False)
    cl,image = getClass(text)
    return render_template('index.html',pred_class=cl,image =image,error=error,plot=True)

#Use the saved model to classify text and plot the pie chart
def  getClass(text):
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                  'talk.religion.misc']
    d = {}
    for i in range(20):
        d[i] = categories[i]
    dn = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dn,'finalized_model.sav')
    #load model
    model = pickle.load(open(filename, 'rb'))
    #predict test
    pred = model.predict([text])
    #classes scores
    pb = model.decision_function([text])[0]
    #classes probabilities
    probs = np.exp(pb) / np.sum(np.exp(pb))
    #plot pie chart
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax1.pie(probs, autopct='%1.1f%%',textprops=dict(color="w"),startangle=0)
    ax1.axis('equal')
    ax1.legend(wedges, categories,
              title="Topics",
              loc="center right",
               bbox_to_anchor=(1, 0.5),bbox_transform=plt.gcf().transFigure
              )
    plt.setp(autotexts, size=8, weight="bold")
    plt.subplots_adjust(top=1.,left=0.0, bottom=0.0, right=0.6)
    # convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig1).print_png(pngImage)
    # encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return d[pred[0]],pngImageB64String

