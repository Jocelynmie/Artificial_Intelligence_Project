import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_classifier(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred, pos_label='spam')
    recall = recall_score(y_true, y_pred, pos_label='spam')
    f1 = f1_score(y_true, y_pred, pos_label='spam')
    
    print("---------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1

def main():
    #load date 
    df = pd.read_csv('/Users/yangwenyu/Desktop/CS5100/Hw08/Data.csv')
    
    #split 
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], 
        df['Category'], 
        test_size=0.2, 
        random_state=42
    )
    
    #naive Bayes with Count Vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_counts, y_train)
    nb_predictions = nb_classifier.predict(X_test_counts)
    
    evaluate_classifier(y_test, nb_predictions, "Naive Bayes (Scikit-learn)")
    
    #TF-IDF with Logistic Regression
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    lr_classifier.fit(X_train_tfidf, y_train)
    lr_predictions = lr_classifier.predict(X_test_tfidf)
    
    evaluate_classifier(y_test, lr_predictions, "TF-IDF + Logistic Regression (Default)")
    
    #TF-IDF Parameter Experiments
    #n-gram
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3)  #2-3 n-gram
    )
    X_train_char = char_tfidf.fit_transform(X_train)
    X_test_char = char_tfidf.transform(X_test)
    
    lr_char = LogisticRegression(max_iter=1000, random_state=42)
    lr_char.fit(X_train_char, y_train)
    char_predictions = lr_char.predict(X_test_char)
    
    evaluate_classifier(y_test, char_predictions, "TF-IDF + LR (Character n-grams)")
    
    # stop words 
    tfidf_stop = TfidfVectorizer(
        stop_words='english'
    )
    X_train_stop = tfidf_stop.fit_transform(X_train)
    X_test_stop = tfidf_stop.transform(X_test)
    
    lr_stop = LogisticRegression(max_iter=1000, random_state=42)
    lr_stop.fit(X_train_stop, y_train)
    stop_predictions = lr_stop.predict(X_test_stop)
    
    evaluate_classifier(y_test, stop_predictions, "TF-IDF + LR (With Stop Words)")
    
    print("---------------------------")
    print(f"Default TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Character n-gram vocabulary size: {len(char_tfidf.vocabulary_)}")
    print(f"Vocabulary size with stop words: {len(tfidf_stop.vocabulary_)}")

if __name__ == "__main__":
    main()