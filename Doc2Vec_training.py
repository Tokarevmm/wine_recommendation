
from gensim.models.doc2vec import Doc2Vec
import datetime

def train_doc2vec_model(tagged_data, vec_size=100, alpha=0.05, max_epochs=10):
    """
    Train a Doc2Vec model using the provided tagged data and parameters.

    Args:
        tagged_data (list of gensim.models.doc2vec.TaggedDocument): List of tagged documents.
        vec_size (int): The size of the vector representation for documents.
        alpha (float): The initial learning rate for training.
        max_epochs (int): The maximum number of training epochs.

    Returns:
        gensim.models.doc2vec.Doc2Vec: Trained Doc2Vec model.
    """
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    

    model.save('./homework_6/models/doc2vec.model')
    
    return model
