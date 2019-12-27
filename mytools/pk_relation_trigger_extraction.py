import numpy as np


class PageRank:

    def __init__(self, word_list, dependency_tree, postag_list, entity1_idx, entity2_idx, postag_set=None, beta=0.5, max_iter=20):
        self._word_list = word_list
        self._dependency_tree = dependency_tree
        self._postag_list = postag_list
        self._entity1_idx = entity1_idx
        self._entity2_idx = entity2_idx
        self._beta = beta
        self._max_iter = max_iter
        if postag_set:
            self._postag_set = postag_set
        else:
            # 满足要求的词性集合
            self._postag_set = {
                # 名词
                'NN', 'NNS', 'NNP', 'NNPS',
                # 副词
                'RB', 'RBR', 'RBS',
                # 动词
                'VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN',
                # 形容词
                'JJ', 'JJR', 'JJS',
                # 介词
                'IN',
                # 动词to 不定式
                'TO',
                # 小品词
                'RP'
            }

    def __get_pr_vector(self, entity_idx):
        pr = np.zeros((len(self._word_list), 1))
        pr[entity_idx, 0] = 1
        return pr
    
    def __get_a_matrix(self):
        word_list_len = len(self._word_list)
        a_matrix = np.zeros((word_list_len, word_list_len))
        for i in range(word_list_len):
            neighbor = self.__get_neighbor(self._word_list, i, self._dependency_tree)
            neighbor_num = len(neighbor)
            for x in neighbor:
                a_matrix[x, i] = 1 / neighbor_num
        return a_matrix

    def __get_neighbor(self, word, idx, dependency_tree):
        '''
        Get the neighbors of the word in dependency tree
        :param  word
        :param  idx: the index of word in word_list
        :param  dependency_tree
        '''
        neighbor = set()
        for _, from_idx, to_idx in dependency_tree:
            if from_idx - 1 == idx:
                if to_idx != 0:
                    neighbor.add(to_idx - 1)
            if to_idx - 1 == idx:
                if from_idx != 0:
                    neighbor.add(from_idx - 1)
        return neighbor

    def __get_pi_vector(self, entity_idx):
        a_matrix = self.__get_a_matrix()
        pr_vector = self.__get_pr_vector(entity_idx)
        word_list_len = len(self._word_list)

        old_pi_vector = np.ones((word_list_len, 1)) * (1 / word_list_len)
        iternum = 0
        while iternum < self._max_iter:
            pi_vector = (1 - self._beta) * np.dot(a_matrix, old_pi_vector) + self._beta * pr_vector
            if len(np.where(np.abs(old_pi_vector - pi_vector) >= 1e-5)[0]) <= 0:
                break
            old_pi_vector = pi_vector
            iternum += 1
        return pi_vector

    def get_trigger_center(self):
        pi_vector1 = self.__get_pi_vector(self._entity1_idx)
        pi_vector2 = self.__get_pi_vector(self._entity2_idx)
        i_vector = pi_vector1 + pi_vector2 + pi_vector1 * pi_vector2
        
        trigger_center, trigger_score, trigger_center_idx = '', -99999, -1
        for i in range(0, len(self._word_list)):
            # 判断词性是否满足要求, 并且评分低于当前的分数
            if all([i != self._entity1_idx, i != self._entity2_idx, self._postag_list[i][1].upper() in self._postag_set, i_vector[i, 0] > trigger_score]):
                trigger_center, trigger_score, trigger_center_idx = self._word_list[i], i_vector[i, 0], i
        return trigger_center, trigger_center_idx

