from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sid = SentimentIntensityAnalyzer()
    sentiment_results = []

    if request.method == 'POST':
        sentence = request.form['sentence']
        ss = sid.polarity_scores(sentence)

        if ss['compound'] >= 0.05:
            sentiment = "Positive"
        elif ss['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        sentiment_results.append({
            'sentence': sentence,
            'sentiment_scores': ss,
            'overall_sentiment': sentiment,
            'sentiment_label': sentiment,
            'sentiment_score': round(ss['compound'], 2)
        })

    return render_template('index.html', results=sentiment_results)

if __name__ == '__main__':
    app.run(debug=True)
