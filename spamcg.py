from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

with open("SMS_SpamdetectionBOWN.pkl",'rb') as f:
    model=pickle.load(f)

with open('BOWNvectorizer.pkl', 'rb') as file:
    TcN=pickle.load(file)

def text_cleaning(cor):
    corpus=str(cor)

    #lower case the corpus
    corpus = corpus.lower()
    
    #removing punctuations
    import string
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))

    #removing digits in the corpus
    import re
    corpus = re.sub(r'\d+','', corpus)

    #removing trailing whitespaces
    corpus = ' '.join([token for token in corpus.split()])
    
    tokenized_corpus_nltk = word_tokenize(corpus)
    
    stop_words_nltk = set(stopwords.words('english'))
    
    tokenized_corpus_without_stopwords = [i for i in tokenized_corpus_nltk if i not in stop_words_nltk]
    
    return " ".join(tokenized_corpus_without_stopwords)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/detectpage')
def detectpage():
    return render_template('htmlpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    name=''
    max_word=['call', '£', 'free', 'u', 'txt', 'ur', 'mobile', 'text', 'stop', 'claim', 'reply', 'prize', 'get', 'p', 'new', 'nokia', 'send', 'cash', 'urgent', 'win', 'service', 'contact', 'please', 'guaranteed', 'customer', 'week', 'tone', 'box', 'phone', 'per', 'chat', 'ppm', 'awarded', 'mins', 'latest', 'draw', 'line', 'po', 'every', 'camera', 'receive', 'go', 'message', 'holiday', 'landline', 'shows', 'apply', 'st', 'number', 'pobox', 'video', 'code', 'live', 'tcs', 'want', 'award', 'msg', 'chance', 'collection', 'entry', 'ringtone', 'pmin', 'orange', 'tones', 'know', 'selected', 'network', 'offer', 'sms', 'mob', 'weekly', 'c', 'find', 'help', 'back', 'valid', 'r', 'cost', 'hrs', 'dont', 'collect', 'word', 'bonus', 'gift', 'delivery', 'attempt', 'yes', 'sae', 'tc', 'club', 'music', 'todays', 'tscs', 'vouchers', 'day', 'rate']
    
    if request.method == 'POST':
        
        message = request.form['message']
        
        textresult=word_tokenize(text_cleaning(message))
        predicted_class=0
        count=1
        for i in textresult:
            if i in max_word:
                transformed_text = TcN.transform([" ".join(textresult)]).toarray()
                predicted_class = model.predict(transformed_text)[0]
                break    
            elif len(textresult)==count:
                predicted_class=0  
            else:
                count=count+1
        #transformed_text = TcN.transform([textresult]).toarray()

        #predicted_class = model.predict(transformed_text)[0]
        
        if predicted_class ==1:
            name="SPAM"
        else:
            name="NOT SPAM"
            
    return render_template('htmlpage.html',predicted_class=name)
    
    
if __name__ == '__main__':
    app.run(debug=True)
