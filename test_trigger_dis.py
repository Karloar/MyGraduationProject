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
s = 0
c = 0
for data in data_list:
    trigger_idx = [int(x) for x in data.trigger_words.split()]
    if len(trigger_idx) > 1:
        s += trigger_idx[-1] - trigger_idx[0]
        c += 1

print(s / c)

