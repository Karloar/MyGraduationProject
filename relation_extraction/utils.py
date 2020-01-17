

def generate_x_y(data_list, myword2vecpkl, max_words=100):
    import numpy as np
    X = []
    y = []
    at_index = myword2vecpkl.word_index['@@@']
    for data in data_list:
        array = np.zeros((max_words, )) * at_index
        for i in range(min([max_words, len(data.word_list)])):
            array[i] = myword2vecpkl.word_index[data.word_list[i]]
        X.append(array)
        y.append(data.relation)
    X = np.array(X)
    y = np.array(y)
    return X, y


class MyWord2VecPKL:
    '''
    词向量工具类, 含有单词的索引、词向量、以及Embedding矩阵
    '''
    def __init__(self, word_index, word_vector, embedding_matrix):
        self.word_num, self.word_dim = embedding_matrix.shape
        self.word_index = word_index
        self.word_vector = word_vector
        self.embedding_matrix = embedding_matrix

    @staticmethod
    def getMyWord2vecPKL(file_path=None):
        '''
        加载myword2vecpkl文件
        '''
        import os
        import pickle as pkl
        if not file_path:
            pwd = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(pwd, 'pkl', 'myword2vecpkl.pkl')
        return pkl.load(open(file_path, 'rb'))
