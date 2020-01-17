import os
from sqlalchemy.orm import sessionmaker
from stanfordcorenlp import StanfordCoreNLP
from models import Reuters10, engine


base_dir = r'/Users/wanglei/Downloads/Reuters10'
Session = sessionmaker(bind=engine)


def get_file_dirs(base_dir):
    '''
    得到包含数据文件的所有文件夹目录
    '''
    file_dirs = []
    for file_dir in os.listdir(base_dir):
        file_abs_dir = os.path.join(base_dir, file_dir)
        if os.path.isdir(file_abs_dir):
            file_dirs.append(file_abs_dir)
    return file_dirs


def get_files(file_dir):
    '''
    从文件夹目录中获取所有数据文件的绝对路径
    '''
    files = []
    for file in os.listdir(file_dir):
        if file.lower() != 'list.txt':
            files.append(os.path.join(file_dir, file))
    return files


def file_process(file_path):
    '''
    读取数据文件，拆分成句子
    '''
    data_lines = []
    begin = False
    with open(file_path, 'r') as f:
        for line in f.readlines()[:-2]:
            line = line.strip()
            if line == '<TEXT>':
                begin = True
                continue
            if line == '</TEXT>':
                break
            if begin:
                data_lines.append(line)
    data = ' '.join(data_lines)
    return split_sentence(data)


def split_sentence(data):
    sents = []
    sent = []
    for i in range(len(data)):
        sent.append(data[i])
        if i < len(data)-1 and data[i] == '.' and data[i+1] == ' ':
            sents.append(''.join(sent).strip())
            sent = []
    return sents


def get_entity_list(ner_list, postag_list):
    '''
    根据NER列表与pos列表，获取标点符号分隔后的候选实体索引列表
    '''
    entity_list = [i for i in range(len(ner_list)) if ner_list[i][1] != 'O' and postag_list[i][1].startswith('NN')]
    punc_list = [i for i in range(len(postag_list)) if postag_list[i][1] in (',', ';', '\'')]
    if not punc_list:
        return [entity_list]
    entities = []
    entity_set = set(entity_list)
    punc_set = set(punc_list)
    temp = []
    for i in range(max(entity_list + punc_list)):
        if i in entity_set:
            temp.append(i)
        elif i in punc_set:
            if temp:
                entities.append(temp)
                temp = []
    if temp:
        entities.append(temp)

    return entities


def get_entities(entity_list, postag_list, min_dis=3, max_dis=12):
    
    for entities in entity_list:
        entity_set = set(entities)
        for entity_idx in entities:
            for dis in range(min_dis, max_dis+1):
                if entity_idx + dis in entity_set:
                    return entity_idx, entity_idx + dis, True
    return -1, -1, False


if __name__ == "__main__":
    count = 0
    session = Session()
    with StanfordCoreNLP('http://127.0.0.1', 9000) as nlp:
        file_dirs = get_file_dirs(base_dir)
        for file_dir in file_dirs:
            files = get_files(file_dir)
            for file in files:
                sents = file_process(file)
                for sent in sents:
                    word_list = nlp.word_tokenize(sent)
                    postag_list = nlp.pos_tag(sent)
                    
                    if len(word_list) < 10 and len(word_list) > 50:
                        continue
                    ner_list = nlp.ner(sent)
                    entity_list = get_entity_list(ner_list, postag_list)
                    entity1_idx, entity2_idx, status = get_entities(entity_list, postag_list)
                    if not status:
                        continue
                    reuter = Reuters10(
                        sent=sent,
                        entity1_idx=entity1_idx,
                        entity2_idx=entity2_idx,
                        entity1=word_list[entity1_idx],
                        entity2=word_list[entity2_idx]
                    )
                    try:
                        session.add(reuter)
                        session.commit()
                        count += 1
                    except Exception as e:
                        print(e)
                    if count % 100 == 0:
                        print('{} sents have comitted!'.format(str(count)))
    try:
        session.commit()
    except Exception as e:
        print(e)
    print(count)
    session.close()
    
    
