import threading
from stanfordcorenlp import StanfordCoreNLP


class WordPreprocessing(threading.Thread):

    def __init__(self, ip, port, data_list):
        super().__init__()
        self._ip = ip
        self._port = port
        self.data_list = data_list

    def run(self):
        with StanfordCoreNLP('http://{}'.format(self._ip), self._port) as nlp:
            for data in self.data_list:
                data.word_list = nlp.word_tokenize(data.sent)
                data.postag_list = nlp.pos_tag(data.sent)
                data.dependency_tree = nlp.dependency_parse(data.sent)


class MultyProcessWordPreprocessing:
    
    def __init__(self, ip_port_list, data_list):
        self._thread_num = len(ip_port_list)
        self._ip_port_list = ip_port_list
        self._data_list = data_list
        self._data_num = len(data_list)

    def get_processed_data(self):
        per_thread_num = int(self._data_num / self._thread_num + 0.5)
        wp_threads = []
        for i in range(self._thread_num):
            ip, port = self._ip_port_list[i]
            data_begin = i * per_thread_num
            data_end = min([(i + 1) * per_thread_num, self._data_num])
            wp_thread = WordPreprocessing(ip, port, self._data_list[data_begin:data_end])
            wp_threads.append(wp_thread)
        for wp_thread in wp_threads:
            wp_thread.setDaemon(True)
            wp_thread.start()
        for wp_thread in wp_threads:
            wp_thread.join()
        data_list = []
        for wp_thread in wp_threads:
            data_list.extend(wp_thread.data_list)
        return data_list
