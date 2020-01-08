
class RelationTrigger:

    def __init__(self, word_list, postag_list, entity1_idx, entity2_idx, trigger_seed, waf, min_dis=5, epsilon=0.003):
        self._word_list = word_list
        self._postag_list = postag_list
        self._entity1_idx = entity1_idx
        self._entity2_idx = entity2_idx
        self._trigger_seed = trigger_seed
        self._waf = waf
        self._min_dis = min_dis
        self._epsilon = epsilon
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

    def get_relation_trigger_words(self):
        trigger_seed_word, trigger_seed_idx = self._trigger_seed
        end_idx = min([len(self._word_list), trigger_seed_idx + self._min_dis + 1])
        waf_value = []
        for i in range(trigger_seed_idx, end_idx):
            if self.waf_value_is_zero(i):
                waf_value.append(0)
            else:
                waf_value.append(self._waf.calculate_activation_force(trigger_seed_word, self._word_list[i], self._min_dis))
        
        relation_trigger_word_idx = [trigger_seed_idx]
        for i in range(len(waf_value)):
            if waf_value[i] > self._epsilon:
                relation_trigger_word_idx.append(trigger_seed_idx + i)
        
        # relation_trigger_word_idx.sort()

        return relation_trigger_word_idx

    def waf_value_is_zero(self, idx):
        if idx == self._entity2_idx or idx == self._entity1_idx:
            return True
        if self._postag_list[idx][1] not in self._trigger_words_postag_set:
            return True
        return False