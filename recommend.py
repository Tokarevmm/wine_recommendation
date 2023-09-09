from gensim.models.doc2vec import Doc2Vec
import pickle

def get_most_similar_docs(test_data, model_path):
   # Load the Doc2Vec model
   model = Doc2Vec.load(model_path)

   # Split the test_data string into a list of words
   test_data_words = test_data.split()

   # Infer the vector for the test document
   inferred_vector = model.infer_vector(test_data_words)

   with open('./homework_6/data/tagged_data.pkl', 'rb') as file:
        loaded_tagged_data = pickle.load(file)
   # Get the 5 most similar documents based on the inferred vector
   sims = model.dv.most_similar([inferred_vector], topn=5)
   idx = [sims[i][0] for i in range(5)]

   # Print the most similar documents
   print('Test Document: «{}»\n'.format(' '.join(test_data_words)))
   print(u'SIMILAR DOCS PER MODEL %s:\n' % model)
   for label, index in [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('5', 4)]:
       print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(loaded_tagged_data[int(sims[index][0])].words)))
   return idx

test_data = 'exotic yellow spice note meet lean lime pith light crisp nose old vine expression historic winery . meyer lemon rind juice show brightly palate grippy chalkiness complement rich lemon curd flavor'
get_most_similar_docs(test_data, './homework_6/models/doc2vec.model')
