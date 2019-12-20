import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from activation_force import get_word_frequency_dict
from utils import get_relation_trigger_seed, get_trigger_idx_list_by_waf




stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
# 259 22
data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))

with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    for data in data_list:
        data.word_list = nlp.word_tokenize(data.sent)

data = SemEval2010Data.objects.get(pk=334)

word_frequency_dict = get_word_frequency_dict(data_list)

sent, entity1, entity2 = data.sent, data.entity1, data.entity2

with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    word_list = nlp.word_tokenize(sent)
    postag_list = nlp.pos_tag(sent)
    dependency_tree = nlp.dependency_parse(sent)
    entity1_idx = data.entity1_idx
    entity2_idx = data.entity2_idx
    print(word_list)
    print('----------------------')
    print(postag_list)
    print('----------------------')
    print(dependency_tree)
    print('----------------------')
    trigger_seed = get_relation_trigger_seed(word_list, postag_list, dependency_tree, entity1_idx, entity2_idx)
    print(trigger_seed)
    print('----------------------')
    get_trigger_idx_list_by_waf(word_list, trigger_seed, entity1_idx, entity2_idx, data_list, word_frequency_dict, postag_list)

