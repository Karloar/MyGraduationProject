import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
# from activation_force import get_word_frequency_dict
# from utils import get_relation_trigger_seed, get_trigger_idx_list_by_waf
from mytools.trigger_seed import TriggerSeedExtraction
from mytools.activation_force2 import ActivationForce



stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
# 259 22
# data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))
data_list = SemEval2010Data.objects.all()
data_list2 = SemEval2010Data.objects.filter(~Q(trigger_words=''))

with StanfordCoreNLP('http://127.0.0.1', 9000, logging_level=logging.WARNING) as nlp:
    for data in data_list:
        data.word_list = nlp.word_tokenize(data.sent)

data = SemEval2010Data.objects.get(pk=386)

af = ActivationForce(data_list, ip_port_list=[('127.0.0.1', 9000)])
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
    re = TriggerSeedExtraction(word_list, dependency_tree, postag_list, entity1_idx, entity2_idx, beta=0.7)
    # trigger_seed = get_relation_trigger_seed(word_list, postag_list, dependency_tree, entity1_idx, entity2_idx)
    trigger_seed = re.get_relation_trigger_seed()
    print(trigger_seed)
    for i in range(trigger_seed[1]):
        print(word_list[i], af.calculate_activation_force(word_list[i], trigger_seed[0], 2), af.word_frequency[word_list[i]])
    for i in range(trigger_seed[1]+1, len(word_list)):
        print(word_list[i], af.calculate_activation_force(trigger_seed[0], word_list[i], 2), af.word_frequency[word_list[i]])
    print(af.word_frequency[trigger_seed[0]])
    
