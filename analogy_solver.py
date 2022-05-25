
"""
The code uses torchtext to use the GloVe model to solve word analogy problems.
"""



import torchtext.vocab

#The name field specifies what the vectors have been trained on, here the 6B means a corpus of 6 billion words.
#The dim argument specifies the dimensionality of the word vectors.
#Loading the the GloVe vectors.
glove = torchtext.vocab.GloVe(name = '6B', dim = 100)  

print(f'There are {len(glove.itos)} words in the vocabulary')

#stoi: string-to-index returns a dictionary of words to indexes
#itos : index-to-string returns an array of words by index
#vectors: returns the actual vectors. 
#get_vector() : a function that takes in embeddings and a word then returns the associated vector. It'll also throw an error if the word doesn't exist in the vocabulary.

def get_vector(embeddings, word):
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]

# The original code used normal distances, therefore altered it to use cosine similarity.
import torch 


def closest_words(embeddings, vector, n = 5):
    
    cos = torch.nn.CosineSimilarity(dim=0)
    distances = [(word, cos(vector, get_vector(embeddings, word)).item())
                 for word in embeddings.itos]
    
    return sorted(distances, key = lambda w: w[1], reverse=True)[:n]

#To print the resulting word and the cosine distance
def print_tuples(tuples):
    for w, d in tuples:
        print(f'({d:02.04f}) {w}')

#Function to solve the word analogy problem

def analogy(embeddings, word1, word2, word3, n=5):
    
    #get vectors for each word
    word1_vector = get_vector(embeddings, word1)
    word2_vector = get_vector(embeddings, word2)
    word3_vector = get_vector(embeddings, word3)
    
    #calculate analogy vector
    analogy_vector = word2_vector - word1_vector + word3_vector
    
    #find closest words to analogy vector
    candidate_words = closest_words(embeddings, analogy_vector, n+3)
    
    #filter out words already in analogy
    candidate_words = [(word, dist) for (word, dist) in candidate_words 
                       if word not in [word1, word2, word3]][:n]
    
    print(f'{word1} is to {word2} as {word3} is to...')
    
    return candidate_words

# Reading the inputs from text file and solving analogy for each problem

with open('analogy_problems.txt') as f:
  for i in range(0,10):
    lines = f.readline().split()
    print( )
    print_tuples(analogy(glove, lines[0], lines[1], lines[2]))

