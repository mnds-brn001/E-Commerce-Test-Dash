import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Custom Portuguese stop words for e-commerce context
CUSTOM_STOP_WORDS = {
    'produto', 'produtos', 'loja', 'compra', 'comprei', 'recebi', 'chegou',
    'dia', 'dias', 'ainda', 'apenas', 'agora', 'hoje', 'depois', 'ante',
    'site', 'contato', 'pedido', 'veio', 'foi', 'ser', 'estar', 'ter'
}

def preprocess_text(text):
    """Preprocess text for NLP analysis."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers, but keep letters with accents
    text = re.sub(r'[^a-záéíóúâêîôûãõçà\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (Portuguese + custom)
    stop_words = set(stopwords.words('portuguese')).union(CUSTOM_STOP_WORDS)
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def generate_wordcloud(text, title, background_color='white'):
    """Generate and return a wordcloud figure."""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig

def extract_topics(text, n_topics=3, n_words=10, method='lda'):
    """Extract topics using either LDA or NMF."""
    # Create document-term matrix using CountVectorizer instead of TfidfVectorizer
    # This is more suitable for our single-document case
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words=list(CUSTOM_STOP_WORDS)
    )
    dtm = vectorizer.fit_transform([text])
    
    # Choose topic modeling method
    if method == 'lda':
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
    else:  # NMF
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200
        )
    
    # Fit model and extract topics
    model.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(f"Tópico {topic_idx + 1}: {', '.join(top_words)}")
    
    return topics

def analyze_sentiment_patterns(reviews):
    """Analyze patterns in reviews to identify common sentiments."""
    # Define sentiment patterns
    positive_patterns = {
        'qualidade': r'(boa|ótima|excelente)\s+qualidade',
        'entrega': r'(entrega\s+rápida|chegou\s+antes)',
        'recomendação': r'(recomendo|voltarei\s+a\s+comprar)',
        'satisfação': r'(muito\s+satisfeito|adorei|gostei\s+muito)',
        'preço': r'(bom\s+preço|preço\s+justo|custo\s+benefício)'
    }
    
    negative_patterns = {
        'atraso': r'(atrasado|não\s+chegou|demora)',
        'qualidade': r'(má\s+qualidade|péssimo|ruim)',
        'problema': r'(defeito|problema|quebrado)',
        'atendimento': r'(péssimo\s+atendimento|sem\s+resposta)',
        'preço': r'(caro|não\s+vale|preço\s+alto)'
    }
    
    patterns_found = {
        'positive': {k: 0 for k in positive_patterns},
        'negative': {k: 0 for k in negative_patterns}
    }
    
    # Count pattern occurrences
    for review in reviews:
        if isinstance(review, str):
            review = review.lower()
            for category, pattern in positive_patterns.items():
                if re.search(pattern, review):
                    patterns_found['positive'][category] += 1
            
            for category, pattern in negative_patterns.items():
                if re.search(pattern, review):
                    patterns_found['negative'][category] += 1
    
    return patterns_found

def analyze_reviews(df):
    """Analyze customer reviews and return insights."""
    # Separate positive, neutral and negative reviews
    positive_reviews = df[df['review_score'] >= 4]['review_comment_message'].dropna()
    neutral_reviews = df[df['review_score'] == 3]['review_comment_message'].dropna()
    negative_reviews = df[df['review_score'] <= 2]['review_comment_message'].dropna()
    
    # Preprocess reviews
    positive_text = ' '.join(positive_reviews.apply(preprocess_text))
    neutral_text = ' '.join(neutral_reviews.apply(preprocess_text))
    negative_text = ' '.join(negative_reviews.apply(preprocess_text))
    
    # Generate wordclouds with different colors
    positive_wordcloud = generate_wordcloud(positive_text, "Palavras mais frequentes em avaliações positivas", 'white')
    neutral_wordcloud = generate_wordcloud(neutral_text, "Palavras mais frequentes em avaliações neutras", '#f0f0f0')
    negative_wordcloud = generate_wordcloud(negative_text, "Palavras mais frequentes em avaliações negativas", '#ffeded')
    
    # Calculate word frequencies
    def get_word_frequencies(text):
        words = text.split()
        freq = pd.Series(words).value_counts().head(20)
        return freq
    
    positive_freq = get_word_frequencies(positive_text)
    neutral_freq = get_word_frequencies(neutral_text)
    negative_freq = get_word_frequencies(negative_text)
    
    # Extract topics using both LDA and NMF
    positive_topics_lda = extract_topics(positive_text, method='lda')
    neutral_topics_lda = extract_topics(neutral_text, method='lda')
    negative_topics_lda = extract_topics(negative_text, method='lda')
    
    positive_topics_nmf = extract_topics(positive_text, method='nmf')
    neutral_topics_nmf = extract_topics(neutral_text, method='nmf')
    negative_topics_nmf = extract_topics(negative_text, method='nmf')
    
    # Analyze sentiment patterns
    positive_patterns = analyze_sentiment_patterns(positive_reviews)
    neutral_patterns = analyze_sentiment_patterns(neutral_reviews)
    negative_patterns = analyze_sentiment_patterns(negative_reviews)
    
    # Calculate additional metrics
    sentiment_metrics = {
        'avg_positive_length': positive_reviews.str.len().mean(),
        'avg_neutral_length': neutral_reviews.str.len().mean(),
        'avg_negative_length': negative_reviews.str.len().mean(),
        'positive_count': len(positive_reviews),
        'neutral_count': len(neutral_reviews),
        'negative_count': len(negative_reviews),
    }
    
    return {
        'positive_wordcloud': positive_wordcloud,
        'neutral_wordcloud': neutral_wordcloud,
        'negative_wordcloud': negative_wordcloud,
        'positive_freq': positive_freq,
        'neutral_freq': neutral_freq,
        'negative_freq': negative_freq,
        'positive_topics_lda': positive_topics_lda,
        'neutral_topics_lda': neutral_topics_lda,
        'negative_topics_lda': negative_topics_lda,
        'positive_topics_nmf': positive_topics_nmf,
        'neutral_topics_nmf': neutral_topics_nmf,
        'negative_topics_nmf': negative_topics_nmf,
        'sentiment_patterns': {
            'positive': positive_patterns,
            'neutral': neutral_patterns,
            'negative': negative_patterns
        },
        'metrics': sentiment_metrics
    } 