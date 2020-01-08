import logging
import init_django_env
from time import sleep
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from mytools.word_repocess import MultyProcessWordProcess


stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'

data_list = SemEval2010Data.objects.all()


mwp = MultyProcessWordProcess([('127.0.0.1', 9000), ('127.0.0.1', 9000)], data_list)
processed_data_list = mwp.get_processed_data()
print(processed_data_list[2000].word_list)
