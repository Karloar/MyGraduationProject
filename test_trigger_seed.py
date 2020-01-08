import logging
import init_django_env
from stanfordcorenlp import StanfordCoreNLP
from django.db.models import Q
from data_process.models import (
    SemEval2010Data, SemEval2010Relation
)
from mytools.trigger_seed import TriggerSeedExtraction
from mytools.activation_force import ActivationForce
from mytools.relation_trigger import RelationTrigger
from mytools.word_preprocessing import MultyProcessWordPreprocessing
from utils import calculate_accuracy, print_running_time




stanford_path = r'/Users/wanglei/Documents/programs/other/stanford-corenlp-full-2018-02-27'
stanfor_server_list = [('127.0.0.1', 9000)]
# 259 22
data_list = SemEval2010Data.objects.filter(~Q(trigger_words=''))
total_data_list = SemEval2010Data.objects.all()

preprocessed_data_list = MultyProcessWordPreprocessing(stanfor_server_list, data_list).get_processed_data()
preprocessed_total_data_list = MultyProcessWordPreprocessing(stanfor_server_list, total_data_list).get_processed_data()

waf = ActivationForce(preprocessed_total_data_list)


@print_running_time
def main():
    for data in data_list:
        data.trigger_seed = TriggerSeedExtraction(
            data.word_list,
            data.dependency_tree,
            data.postag_list,
            data.entity1_idx,
            data.entity2_idx,
            beta=0.7
        ).get_relation_trigger_seed()
      

    (
        seed_micro_accuracy,
        seed_macro_accuracy,
        seed_first_accuracy,
    ) = calculate_accuracy(data_list, kind_list=['seed_micro', 'seed_macro', 'seed_first'])

    print('触发词种子微准确率：', seed_micro_accuracy)
    print('触发词种子宏准确率：', seed_macro_accuracy)
    print('触发词种子为第一触发词概率：', seed_first_accuracy)


if __name__ == "__main__":
    main()
