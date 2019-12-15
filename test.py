import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from utils import get_relation_trigger_center, get_entity_idx


stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
# 259 22
data = SemEval2010Data.objects.get(pk=399)

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
    relation_trigger_center = get_relation_trigger_center(word_list, postag_list, dependency_tree, entity1_idx, entity2_idx)
    print(relation_trigger_center)
