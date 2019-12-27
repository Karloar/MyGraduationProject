

def calculate_activation_force(word_i, word_j, data_list, word_frequency_dict, min_dis=5):
    '''
    计算 activation_force
    '''
    ij_frequency, ij_dist = get_ij_frequency_and_dist(word_i, word_j, data_list, min_dis)
    i_frequency, j_frequency = word_frequency_dict[word_i], word_frequency_dict[word_j]
    if ij_dist == 0:
        return 0
    return (ij_frequency / i_frequency) * (ij_frequency / j_frequency) / (ij_dist * ij_dist)


def get_word_frequency_dict(data_list):
    word_frequency_dict = dict()
    for data in data_list:
        for word in data.word_list:
            word_frequency_dict[word] = word_frequency_dict.get(word, 0) + 1
    return word_frequency_dict


def get_ij_frequency_and_dist(word_i, word_j, data_list, min_dis):
    '''
    计算两个单词同时出现的频率与平均距离
    '''
    total_ij_frequency = 0
    total_ij_dist = 0
    total_count = 0
    for data in data_list:
        ij_frequency, ij_dist, count = get_ij_frequency_and_dist_from_word_list(
            word_i,
            word_j,
            data.word_list,
            min_dis
        )
        total_ij_frequency += ij_frequency
        total_ij_dist += ij_dist
        total_count += count
    return total_ij_frequency, total_ij_dist / total_count
        

def get_word_index_list(word, word_list):
    '''
    从列表中找到某个词的所有索引
    '''
    idx_list = []
    begin = 0
    while True:
        try:
            idx_list.append(word_list.index(word, begin))
            begin = idx_list[-1] + 1
        except ValueError:
            break
    return idx_list


def get_ij_frequency_and_dist_from_word_list(word_i, word_j, word_list, min_dis):
    '''
    根据一个句子的词列表, 计算单词i与单词j在某个距离内的共现词频、距离和、出现次数
    '''
    word_i_idx_list = get_word_index_list(word_i, word_list)
    word_j_idx_list = get_word_index_list(word_j, word_list)
    if not word_i_idx_list or not word_j_idx_list:
        return 0, 0, 0
    ij_frequency = 0
    ij_dist = 0
    count = 0
    i_idx = 0
    j_idx = 0
    # ==============================
    # 这里需要修改
    # ==============================
    while i_idx < len(word_i_idx_list) and j_idx < len(word_j_idx_list):
        if word_i_idx_list[i_idx] > word_j_idx_list[j_idx]:
            j_idx += 1
        else:
            ij_frequency += 1
            ij_dist += word_j_idx_list[j_idx] - word_i_idx_list[i_idx]
            count += 1
            i_idx += 1
    return ij_frequency, ij_dist, count

    