import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from mytools.trigger_seed import TriggerSeedExtraction



stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
# 259 22 399 577 163 173 203 366 369 616 
# 613 644 869 877 883 902
# 944 945
data = SemEval2010Data.objects.get(pk=1081)

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
    trigger_seed = TriggerSeedExtraction(
        word_list,
        dependency_tree,
        postag_list,
        entity1_idx,
        entity2_idx,
        beta=0.7
    ).get_relation_trigger_seed()
    print(trigger_seed)
    
