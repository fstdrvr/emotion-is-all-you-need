import spacy
from transformers import pipeline, AutoModelForSequenceClassification

def load_bert_model():
    """Loads a pre-trained BERT model for sentiment analysis.

    Returns:
        transformers.pipelines.Pipeline: A Hugging Face pipeline for sentiment analysis.
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
    return pipeline("sentiment-analysis", model=model_name)

def load_spacy_model():
    """Loads a spaCy language model.

    Returns:
        spacy.language.Language: A spaCy language model.
    """

    return spacy.load("en_core_web_sm")

def preprocess_text(text, spacy_model):
    """Preprocesses a text string using spaCy.

    Args:
        text: The text string to preprocess.
        spacy_model: The loaded spaCy language model.

    Returns:
        A preprocessed text string (cleaned, tokenized, etc.).
    """

    doc = spacy_model(text)
    # remove stop words and punctuation
    processed_text = " ".join([token.text for token in doc 
                               if not (token.is_stop or token.is_punct)]) 
    return processed_text

if __name__ == "__main__":
    bert_pipeline = load_bert_model()
    nlp = load_spacy_model()

    comment = "This video was absolutely amazing!"
    preprocessed_comment = preprocess_text(comment, nlp)

    # Use the BERT pipeline for sentiment analysis (would be done in sentiment_analysis.py)
    sentiment_result = bert_pipeline(preprocessed_comment) 
    print(sentiment_result) 