from mytools.word_preprocessing import MultyProcessWordPreprocessing


class ActivationForce:

    def __init__(self, data_list, word_frequency_dict=None, ip_port_list=None):
        self._data_list = data_list
        self.word_frequency = word_frequency_dict
        if not word_frequency_dict:
            self.word_frequency = get_word_frequency_dict(self._data_list, ip_port_list)
    

    def calculate_activation_force(self, word_i, word_j, min_dis):
        ij_frequency, ij_dist = get_ij_frequency_and_dist(word_i, word_j, self._data_list, min_dis)
        i_frequency, j_frequency = self.word_frequency[word_i], self.word_frequency[word_j]
        if ij_dist == 0:
            return 0
        return (ij_frequency / i_frequency) * (ij_frequency / j_frequency) / (ij_dist * ij_dist)



def get_word_frequency_dict(data_list, ip_port_list):
    word_frequency_dict = dict()
    if not hasattr(data_list[0], 'word_list'):
        if not ip_port_list:
            raise Exception('需要传递ip与端口号列表')
        mpwp = MultyProcessWordPreprocessing(ip_port_list, data_list)
        data_list = mpwp.get_processed_data()
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
    if total_count == 0:
        return total_ij_frequency, 0
    return total_ij_frequency, total_ij_dist / total_count
        

def get_word_index_list(word, word_list):
    '''
    从列表中找到某个词的所有索引
    '''
    return [index for (index, value) in enumerate(word_list) if value.lower() == word.lower()]


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
    for word_i_idx in word_i_idx_list:
        for word_j_idx in word_j_idx_list:
            if word_i_idx < word_j_idx and word_j_idx - word_i_idx <= min_dis:
                ij_frequency += 1
                ij_dist += word_j_idx - word_i_idx
                count += 1
    return ij_frequency, ij_dist, count

    