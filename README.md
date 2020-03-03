# Text Classifier Web App
This project consists of a single page containing a form to submit a text, then showing the predicted class of this text using 
the saved model. 

# Usage
### Local 
  ```
  git clone https://github.com/wafa-nakbi/DocumentClassifier
  cd DocumentClassifier
  pip install -r requirements.txt
  python textclassifier.py
  ```
### Docker
Install docker then run
```
  git clone https://github.com/wafa-nakbi/DocumentClassifier
  cd DocumentClassifier
  sudo docker build -t textclassifier .
  sudo docker run -p 5000:5000 textclassifier
    
```
Then open  http://localhost:5000
  



