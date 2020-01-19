import numpy as np


class TriggerSeedExtraction:
    
    def __init__(
        self, word_list, dependency_tree, postag_list, entity1_idx, entity2_idx, 
        trigger_seed_postag_set=None, trigger_words_postag_set=None, beta=0.5
    ):
        self._word_list = word_list
        self._dependency_tree = dependency_tree
        self._postag_list = postag_list
        self._entity1_idx = entity1_idx
        self._entity2_idx = entity2_idx
        self._beta = beta
        if trigger_seed_postag_set:
            self._trigger_seed_postag_set = trigger_seed_postag_set
        else:
            # 满足要求的触发词种子词性集合
            self._trigger_seed_postag_set = {
                # 名词
                'NN', 'NNS', 'NNP', 'NNPS',
                # 动词
                'VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN',
                # 形容词
                'JJ', 'JJR', 'JJS',
                # 介词
                'IN',
            }
        if trigger_words_postag_set:
            self._trigger_words_postag_set = trigger_words_postag_set
        else:
            # 满足要求的关系触发词词性集合
            self._trigger_words_postag_set = {
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

    def get_relation_trigger_seed(self):
        '''
        获取触发词种子
        '''
        # 计算序列距离, 得到向量
        self.order_distance_vector = get_order_distance_vector(self._word_list, self._entity1_idx, self._entity2_idx)
        # 计算依存距离, 得到向量
        self.syntactic_distance_vector = get_syntactic_distance_vector(
            self._word_list, self._dependency_tree, self._entity1_idx, self._entity2_idx
        )
        # 计算词性得分, 得到向量
        self.pos_vector = get_pos_vector(self._postag_list)
        # 计算每个单词的评分, 评分越低的越可能成为中心触发词
        self.score_vector = (
            self._beta * self.syntactic_distance_vector + (1 - self._beta) * self.order_distance_vector
        ) * self.pos_vector
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
        # 初始化触发词种子与分数
        trigger_seed, trigger_seed_score, trigger_seed_idx = '', 99999, -1
        # 倒序遍历词列表, 避免have been moving这种多个动词的情况
        for i in range(len(self._word_list)-1, -1, -1):
            if all([i != self._entity1_idx, i != self._entity2_idx, self._postag_list[i][1].upper() in postag_set, self.score_vector[i] < trigger_seed_score]):
                trigger_seed, trigger_seed_score, trigger_seed_idx = self._word_list[i], self.score_vector[i], i
        return trigger_seed, trigger_seed_idx

    


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
    word_entity_dependency_vector[entity_idx] = 0

    temp_queue.put(entity_idx)
    while not temp_queue.empty():
        idx = temp_queue.get()
        for _, from_idx, to_idx in dependency_tree:
            from_idx -= 1
            to_idx -= 1
            if all([from_idx == idx, to_idx != -1, word_entity_dependency_vector[to_idx] == 0]):
                word_entity_dependency_vector[to_idx] = word_entity_dependency_vector[idx] + 1
                temp_queue.put(to_idx)
            if all([to_idx == idx, from_idx != -1, word_entity_dependency_vector[from_idx] == 0]):
                word_entity_dependency_vector[from_idx] = word_entity_dependency_vector[idx] + 1
                temp_queue.put(from_idx)
    return word_entity_dependency_vector


def get_syntactic_distance_vector(word_list, dependency_tree, entity1_idx, entity2_idx):
    '''
    计算每个单词的Syntactic Distance, 形成一个向量
    '''
    word_entity_dependency_vector1 = get_word_entity_dependency_vector(word_list, dependency_tree, entity1_idx)
    word_entity_dependency_vector2 = get_word_entity_dependency_vector(word_list, dependency_tree, entity2_idx)
    # print(word_entity_dependency_vector1 * word_entity_dependency_vector2)
    return np.sqrt(word_entity_dependency_vector1 * word_entity_dependency_vector2) / max(
        list(word_entity_dependency_vector1) + list(word_entity_dependency_vector2)
    )


def get_pos_vector(postag_list):
    postag_len = len(postag_list)
    pos_vector = np.zeros(postag_len)
    pos_mark = dict(
        {'IN': 0.1},
        **{vb: 0.1 for vb in ['VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN']},
        **{nn: 0.12 for nn in ['NN', 'NNS', 'NNP', 'NNPS']},
        **{rb: 0.3 for rb in ['RB', 'RBR', 'RBS']},
        **{jj: 0.13 for jj in ['JJ', 'JJR', 'JJS']}
    )
    for i in range(postag_len):
        pos_vector[i] = pos_mark.get(postag_list[i][1].upper(), 1)
    return pos_vector
