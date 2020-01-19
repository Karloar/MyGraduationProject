

def generate_x_y(
    data_list, myword2vecpkl, 
    max_words=100, use_trigger_words=False, 
    remove_other=False, remove_nearby=False, 
    add_entity_feature=False
):
    import numpy as np
    X = []
    y = []
    trigger_words_array = []
    at_index = myword2vecpkl.word_index['@@@']
    for data in data_list:
        if remove_nearby and data.entity2_idx - data.entity1_idx == 1:
            continue
        if remove_other and data.relation == 0:
            continue
        array = np.zeros((max_words, )) * at_index
        for i in range(min([max_words, len(data.word_list)])):
            array[i] = myword2vecpkl.word_index[data.word_list[i]]
        X.append(array)
        if remove_other:
            y.append(data.relation-1)
        else:
            y.append(data.relation)
        if use_trigger_words:
            trigger_words_vector = np.zeros((max_words, ))
            if add_entity_feature:
                trigger_words_vector[data.entity1_idx] = 1
                trigger_words_vector[data.entity2_idx] = 1

            for idx in data.trigger_list:
                trigger_words_vector[idx] = 1
            trigger_words_array.append(trigger_words_vector)
    X = np.array(X)
    y = np.array(y)
    trigger_words_array = np.array(trigger_words_array)
    if use_trigger_words:
        return [X, trigger_words_array], y
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
