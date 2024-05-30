from pysentimiento import create_analyzer

analyzer = create_analyzer(task='sentiment',lang='es')


def analyze_sentiment (message):
    sentiment = analyzer.predict(message).output
    return sentiment