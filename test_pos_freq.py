import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)


stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'

data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))


freq_dict = {'NN': 0, 'VB': 0, 'JJ': 0, 'IN': 0, 'RB': 0, 'TO': 0, 'RP': 0}
single_freq_dict = {}


with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    for data in data_list:
        sent, e1_idx, e2_idx = data.sent, data.entity1_idx, data.entity2_idx
        word_list = nlp.word_tokenize(sent)
        postag_list = nlp.pos_tag(sent)
        # print(data.id, postag_list)
        for x in data.trigger_words.split():
            freq_dict[postag_list[int(x)][1][:2]] += 1
        if len(data.trigger_words.split()) == 1:
            idx = int(data.trigger_words.split()[0])
            temp_freq = single_freq_dict.get(postag_list[idx][1][:2], 0)
            single_freq_dict[postag_list[idx][1][:2]] = temp_freq + 1
        

print(freq_dict)
print(single_freq_dict)
