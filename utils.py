import numpy as np
from activation_force import calculate_activation_force


def get_trigger_words_accuracy(data_list):
    micro_count = 0
    macro_count = 0
    data_len = len(data_list)
    for data in data_list:
        if ' '.join([str(idx) for idx in data.trigger_list]) == data.trigger_words:
            micro_count += 1
        # if set(data.trigger_list).issubset(set([int(idx) for idx in data.trigger_words.split()])):
        #     macro_count += 1
        temp_trigger_words = set([int(x) for x in data.trigger_words.split()])
        for trigger in data.trigger_list:
            if trigger in temp_trigger_words:
                macro_count += 1
                break
            
    return micro_count / data_len * 100, macro_count / data_len * 100


def calculate_accuracy(data_list, kind='micro'):
    micro_count = 0
    macro_count = 0
    seed_micro_count = 0
    seed_macro_count = 0
    data_count = len(data_list)
    for data in data_list:
        # 统计提取出关系触发词与标记完全一致的个数
        if ' '.join([str(idx) for idx in data.trigger_list]) == data.trigger_words:
            micro_count += 1
        
        # 统计提取出关系触发词种子正确个数
        if hasattr(data, 'postag_list'):
            postag_list = getattr(data, 'postag_list')
            seed_is_vb = postag_list[data.trigger_seed[1]][1].startswith('VB')
            trigger_has_vb = any([postag_list[int(x)][1].startswith('VB') for x in data.trigger_words.split()])
            seed_is_nn = postag_list[data.trigger_seed[1]][1].startswith('NN')
            trigger_has_nn = any([postag_list[int(x)][1].startswith('NN') for x in data.trigger_words.split()])
            seed_is_jj = postag_list[data.trigger_seed[1]][1].startswith('JJ')
            trigger_has_jj = any([postag_list[int(x)][1].startswith('JJ') for x in data.trigger_words.split()])

            if any([
                all([seed_is_vb, trigger_has_vb]),
                all([seed_is_nn, trigger_has_nn]),
                all([seed_is_jj, trigger_has_jj]),
                not any([
                   seed_is_vb, trigger_has_vb,
                   seed_is_nn, trigger_has_nn,
                   seed_is_jj, trigger_has_jj
                ])
            ]):
                if str(data.trigger_seed[1]) in data.trigger_words.split():
                    seed_micro_count += 1
        
        # 统计提取出的关系触发词种子包含在标记的个数
        temp_trigger_words = set([int(x) for x in data.trigger_words.split()])
        if data.trigger_seed[1] in temp_trigger_words:
            seed_macro_count += 1
        
        # 统计提取出的关系触发词中部分出现在标记中的个数
        for trigger in data.trigger_list:
            if trigger in temp_trigger_words:
                macro_count += 1
                break
    accuracy = {
        'micro': micro_count / data_count * 100,
        'macro': macro_count / data_count * 100,
        'seed_micro': seed_micro_count / data_count * 100,
        'seed_macro': seed_macro_count / data_count * 100 
    }
    return accuracy.get(kind, None)


def get_trigger_idx_list_by_waf(word_list, trigger_seed, entity1_idx, entity2_idx, data_list, word_frequency_dict, postag_list=None):
    '''
    根据Trigger Seed与Word Activation Force获取完整的关系触发词
    '''
    trigger_seed_idx = trigger_seed[1]
    data_list_len = len(data_list)
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
    trigger_list = [trigger_seed_idx]
    for i in range(0, max([entity1_idx, entity2_idx, trigger_seed_idx])):
        if i == entity1_idx or i == entity2_idx or i == trigger_seed_idx:
            continue
        if postag_list and postag_list[i][1] not in postag_set:
            continue
        if i < trigger_seed_idx:
            waf = calculate_activation_force(word_list[i], trigger_seed[0], data_list, word_frequency_dict)
        else:
            waf = calculate_activation_force(trigger_seed[0], word_list[i], data_list, word_frequency_dict)
        if waf * data_list_len >= 2:
            trigger_list.append(i)
    trigger_list.sort()
    return trigger_list



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
    
    

def get_relation_trigger_seed(word_list, postag_list, dependency_tree, entity1_idx, entity2_idx, beta=0.5):
    '''
    获取触发词种子
    '''
    # 计算序列距离, 得到向量
    order_distance_vector = get_order_distance_vector(word_list, entity1_idx, entity2_idx)
    # 计算依存距离, 得到向量
    syntactic_distance_vector = get_syntactic_distance_vector(word_list, dependency_tree, entity1_idx, entity2_idx)
    # 计算词性得分, 得到向量
    pos_vector = get_pos_vector(postag_list)
    # 计算每个单词的评分, 评分越低的越可能成为中心触发词
    score_vector = (beta * syntactic_distance_vector + (1 - beta) * order_distance_vector) * pos_vector
    # print(score_vector)
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
