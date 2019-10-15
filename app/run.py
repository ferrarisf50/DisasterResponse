import json
import plotly
import pandas as pd
import re
import nltk

#nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('wordnet') 
#nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()

#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)

#    return clean_tokens

def tokenize(text):
    '''
    INPUT  a text string
    
    OUTPUT  a list of tokenized words 
    '''
    text = text.lower()
    
    # remove urls:   
    text = re.sub(r'http(s)?://[^ ]+','',text)
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    result = stemmed
    return result


# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
print(df)
print(df.columns)
# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    # graph 2 top 10 message categories
    df2 = df.drop(columns=['id','message','original','genre']).mean().reset_index().sort_values(by=0, ascending=0)
    
    genre_counts2 = list(df2['index'][:10])
    df2[0] = df2[0]*100
    genre_names2 = list(df2[0][:10].apply('{:,.2f}%'.format))
    
    # graph 3 correlation
    
    df3 = df.drop(columns=['id','message','original','genre','child_alone']).corr(method ='pearson')
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_counts2,
                    y=genre_names2,
                    text=genre_names2,
                    textposition='auto'
                   
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=df3,
                    x=df3.columns,
                    y=df3.columns
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories'
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    print(query)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()