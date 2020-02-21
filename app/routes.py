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
    cl=''

    text=request.form.get("text")
    if not text:
        error="Error. No text entered."
        return render_template('index.html', text=cl, error=error)

    cl,image = getClass(text)

    return render_template('index.html',text=cl,image =image,error=error,plot=True)

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
    vectorizer_name = os.path.join(dn,'vectorizer.pickle')
    image_path=os.path.join(dn,'static/images')
    model = pickle.load(open(filename, 'rb'))
    vectorizer = pickle.load(open(vectorizer_name, 'rb'))
    X_test = vectorizer.transform([text])
    pred = model.predict(X_test)
    pb = model.decision_function(X_test)[0]
    probs = np.exp(pb) / np.sum(np.exp(pb))
    fig1, ax1 = plt.subplots()
    ax1.pie(probs, labels=categories, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    image_path=os.path.join(image_path,'new_plot.png')
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig1).print_png(pngImage)
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return d[pred[0]],pngImageB64String

