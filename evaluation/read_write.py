import numpy
import math

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename,is_normalization=True):
    word_vecs = {}
    with open(filename, 'r', encoding="utf-8") as f:
        line=f.readline().strip().lower()
        while line:
            word = line.split()[0]
            word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
            for index, vec_val in enumerate(line.split()[1:]):
                word_vecs[word][index] = float(vec_val)
            ''' normalize weight vector '''
            if is_normalization:word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)
            line = f.readline().strip().lower()

    print("Vectors read from: "+filename+" \n")
    return word_vecs
