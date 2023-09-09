from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import pickle

def create_tagged_documents(dataframe, text_column='summary'):
    """
    Create tagged documents for Doc2Vec model from a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing text data.
        text_column (str): The name of the column containing the text data.

    Returns:
        list of gensim.models.doc2vec.TaggedDocument: List of tagged documents.
    """
    data = dataframe[text_column].astype(str).tolist()

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    save_path = './homework_6/data/tagged_data.pkl'

    with open(save_path, 'wb') as file:
        pickle.dump(tagged_data, file)
    
    return tagged_data
