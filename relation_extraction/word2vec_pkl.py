import os
import sys
import pickle as pkl
import numpy as np


pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(pwd))

import init_django_env
from data_process.models import SemEval2010Data
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import Word2Vec
from utils import MyWord2VecPKL


word2vec_path = r'/Users/wanglei/Documents/programs/other/word2vec'
word2vec_file = os.path.join(word2vec_path, 'wiki_en_vector_vec.model')
word2vec_dim = 200

myword2vecpkl_file = os.path.join(pwd, 'pkl', 'myword2vecpkl.pkl')



if __name__ == "__main__":
    if not os.path.exists(myword2vecpkl_file):
        word2vector_model = Word2Vec.load(word2vec_file)
        all_data = SemEval2010Data.objects.all()
        word_index = dict()
        word_vector = dict()
        index = 0
        with StanfordCoreNLP('http://127.0.0.1', 9000) as nlp:
            for data in all_data:
                word_list = nlp.word_tokenize(data.sent)
                for word in word_list:
                    if word not in word_index:
                        word_index[word] = index
                        index += 1
                        if word in word2vector_model:
                            word_vector[word] = word2vector_model[word]
                        else:
                            word_vector[word] = np.zeros((word2vec_dim, ))
        word_index['@@@'] = index
        word_vector['@@@'] = np.zeros((word2vec_dim, ))
        
        word_num = len(word_index)
        embedding_matrix = np.zeros((word_num, word2vec_dim))
        for word, i in word_index.items():
            embedding_matrix[i, :] = word_vector[word]
        myword2vecpkl = MyWord2VecPKL(word_index, word_vector, embedding_matrix)

        pkl.dump(myword2vecpkl, open(myword2vecpkl_file, 'wb'))
        print('生成词典文件！！！')
