import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from utils import get_relation_trigger_seed


stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'

data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))



with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    for data in data_list:
        sent, e1_idx, e2_idx = data.sent, data.entity1_idx, data.entity2_idx

        word_list = nlp.word_tokenize(sent)
        postag_list = nlp.pos_tag(sent)
        dependency_tree = nlp.dependency_parse(sent)
        data.trigger_seed = get_relation_trigger_seed(word_list, postag_list, dependency_tree, e1_idx, e2_idx, beta=0.8)
        data.word_list = word_list

count = 0
for data in data_list:
    # print(data.relation_trigger_center, '----------', data.trigger_words.split())
    if data.trigger_seed[1] in [int(x) for x in data.trigger_words.split()]:
        count += 1

print('准确率：', '{}%'.format(str(count / len(data_list) * 100)))

# for data in data_list:
#     if data.relation_trigger_center[1]  not in [int(x) for x in data.trigger_words.split()]:
#         print('---------------------------')
#         print(data.id, data.sent)
#         print(data.entity1_idx, data.entity1)
#         print(data.entity2_idx, data.entity2)
#         print(data.relation_trigger_center)
#         print(data.trigger_words)
#         print('---------------------------')

