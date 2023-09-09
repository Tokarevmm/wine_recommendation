from collections import Counter
import spacy
nlp = spacy.load('en_core_web_sm')

def generate_summaries(df, input_column, output_column, num_sentences=3):
    # Define a function to summarize text

    def calculate_sentence_importance(sentence):
    # Tokenize the sentence into words
      words = sentence.split()

    # Calculate word frequencies using Counter
      word_freq = Counter(words)

    # Calculate the sentence importance as the sum of word frequencies
      sentence_importance = sum(word_freq.values())

      return sentence_importance

    def summarize_text(text):
        doc = nlp(text)

        # Tokenize the text into sentences
        sentences = [sent.text for sent in doc.sents]

        # Calculate sentence importance scores
        sentence_scores = [calculate_sentence_importance(sent) for sent in sentences]

        # Rank sentences by importance and select the top 'num_sentences' sentences
        ranked_sentences = sorted(
            ((sentence, score) for sentence, score in zip(sentences, sentence_scores)),
            key=lambda x: x[1],
            reverse=True)[:num_sentences]

        # Generate the summarized text
        summarized_text = [sentence for sentence, _ in ranked_sentences]

        return ' '.join(summarized_text)

    # Apply the summarization function to the input_column
    df[output_column] = df[input_column].apply(summarize_text)

    return df

