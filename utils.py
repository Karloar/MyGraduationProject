import numpy as np


def get_word_entity_vector(word_list, entity_idx):
    '''
    辅助函数：得到句子中的每个单词到实体的序列距离, 形成一个向量
    '''
    word_entity_vector = np.zeros((len(word_list),))
    for i in range(len(word_list)):
        word_entity_vector[i] = np.abs(i -  entity_idx)
    return word_entity_vector


def get_order_distance_vector(word_list, entity1_idx, entity2_idx):
    '''
    计算每个单词的Order Distance, 形成一个向量
    '''
    word_entity_vector1 = get_word_entity_vector(word_list, entity1_idx)
    word_entity_vector2 = get_word_entity_vector(word_list, entity2_idx)
    return (word_entity_vector1 + word_entity_vector2) / (
        2 * max(list(word_entity_vector1) + list(word_entity_vector2))
    )

def get_word_entity_dependency_vector(word_list, dependency_tree, entity_idx):
    '''
    辅助函数：计算句子中的每个单词到集体的依存距离
    '''
    from queue import Queue

    temp_queue = Queue(len(word_list))
    word_entity_dependency_vector = np.zeros((len(word_list),))
    word_entity_dependency_vector[entity_idx] = 1

    temp_queue.put(entity_idx)
    while not temp_queue.empty():
        idx = temp_queue.get()
        for _, from_idx, to_idx in dependency_tree:
            if all([from_idx - 1 == idx, to_idx != 0, word_entity_dependency_vector[to_idx - 1] == 0]):
                word_entity_dependency_vector[to_idx - 1] = word_entity_dependency_vector[idx] + 1
                temp_queue.put(to_idx - 1)
            if all([to_idx - 1 == idx, from_idx != 0, word_entity_dependency_vector[from_idx - 1] == 0]):
                word_entity_dependency_vector[from_idx - 1] = word_entity_dependency_vector[idx] + 1
                temp_queue.put(from_idx - 1)
    return word_entity_dependency_vector - 1


def get_syntactic_distance_vector(word_list, dependency_tree, entity1_idx, entity2_idx):
    '''
    计算每个单词的Syntactic Distance, 形成一个向量
    '''
    word_entity_dependency_vector1 = get_word_entity_dependency_vector(word_list, dependency_tree, entity1_idx)
    word_entity_dependency_vector2 = get_word_entity_dependency_vector(word_list, dependency_tree, entity2_idx)
    return np.sqrt(word_entity_dependency_vector1 * word_entity_dependency_vector2) / max(
        list(word_entity_dependency_vector1) + list(word_entity_dependency_vector2)
    )


def get_pos_vector(postag_list):
    postag_len = len(postag_list)
    pos_vector = np.zeros(postag_len)
    pos_mark = dict(
        {'IN': 0.1},
        **{vb: 0.15 for vb in ['VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN']},
        **{nn: 0.2 for nn in ['NN', 'NNS', 'NNP', 'NNPS']},
        **{rb: 0.3 for rb in ['RB', 'RBR', 'RBS']},
        **{jj: 0.3 for jj in ['JJ', 'JJR', 'JJS']}
    )
    for i in range(postag_len):
        pos_vector[i] = pos_mark.get(postag_list[i][1].upper(), 1)
    return pos_vector
    
    

def get_relation_trigger_center(word_list, postag_list, dependency_tree, entity1_idx, entity2_idx, beta=0.5):
    '''
    获取中心触发词
    '''
    # 计算序列距离, 得到向量
    order_distance_vector = get_order_distance_vector(word_list, entity1_idx, entity2_idx)
    # 计算依存距离, 得到向量
    syntactic_distance_vector = get_syntactic_distance_vector(word_list, dependency_tree, entity1_idx, entity2_idx)
    # 计算词性得分, 得到向量
    pos_vector = get_pos_vector(postag_list)
    # 计算每个单词的评分, 评分越低的越可能成为中心触发词
    score_vector = (beta * syntactic_distance_vector + (1 - beta) * order_distance_vector) * pos_vector
    # print('---------- order distance ---------')
    # print(order_distance_vector)
    # print('---------- Syntactic distance ---------')
    # print(syntactic_distance_vector)
    # print('----------------- Score -----------------')
    # print(score_vector)
    # print('-----------------------------------------')
    # 满足要求的词性集合
    postag_set = {
        # 名词
        'NN', 'NNS', 'NNP', 'NNPS',
        # 副词
        'RB', 'RBR', 'RBS',
        # 动词
        'VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN',
        # 形容词
        'JJ', 'JJR', 'JJS',
        # 介词
        'IN'
    }
    # 初始化中心触发词与分数
    trigger_center, trigger_score, trigger_center_idx = '', 99999, -1
    # 倒序遍历词列表, 避免have been moving这种多个动词的情况
    for i in range(len(word_list)-1, -1, -1):
        if all([i != entity1_idx, i != entity2_idx, postag_list[i][1].upper() in postag_set, score_vector[i] < trigger_score]):
            trigger_center, trigger_score, trigger_center_idx = word_list[i], score_vector[i], i
    return trigger_center, trigger_center_idx


def get_entity_idx(word_list, entity):
    '''
    从分词之后的词列表中获取实体的索引
    '''
    word_list_lower = [word.lower() for word in word_list]
    entity_lower = entity.lower()
    entity_idx = [word_list_lower.index(entity_lower)]
    while True:
        try:
            entity_idx.append(word_list_lower.index(entity_lower, entity_idx[-1] + 1))
        except ValueError:
            break
    return entity_idx



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
                'IN'
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

