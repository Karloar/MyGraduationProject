import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from activation_force import get_word_frequency_dict
from utils import (
    get_relation_trigger_seed, get_trigger_idx_list_by_waf,
    get_trigger_words_accuracy
)




stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
# 259 22
data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))

with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    for data in data_list:
        data.word_list = nlp.word_tokenize(data.sent)
        data.postag_list = nlp.pos_tag(data.sent)
        data.dependency_tree = nlp.dependency_parse(data.sent)


word_frequency_dict = get_word_frequency_dict(data_list)


for data in data_list:
    trigger_seed = get_relation_trigger_seed(
        data.word_list, data.postag_list, data.dependency_tree, data.entity1_idx, data.entity2_idx
    )
    data.trigger_list = get_trigger_idx_list_by_waf(
        data.word_list, trigger_seed, data.entity1_idx, data.entity2_idx, data_list, word_frequency_dict, data.postag_list
    )


micro_accuracy, macro_accuracy = get_trigger_words_accuracy(data_list)
print('微准确率：', micro_accuracy)
print('宏准确率：', macro_accuracy)


