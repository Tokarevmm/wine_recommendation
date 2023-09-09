from data_load import read_csv
from preprocessing import preprocess_text
from summaryzation import generate_summaries
from tagging import create_tagged_documents
from Doc2Vec_training import train_doc2vec_model


if __name__ == '__main__':
    data = read_csv('./homework_6/data/wine_reviews.csv', dropna=True, dropduplicates=True)
    data['Description_Cleaned'] = data['description'].apply(lambda x: preprocess_text(x))
    data = generate_summaries(data, 'Description_Cleaned', 'summary')
    tagged_data = create_tagged_documents(data, 'summary')
    train_doc2vec_model(tagged_data, vec_size=100, alpha=0.05, max_epochs=10)

