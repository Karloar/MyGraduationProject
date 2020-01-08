import numpy as np


def calculate_accuracy(data_list, kind_list=['micro']):
    micro_count = 0
    macro_count = 0
    seed_micro_count = 0
    seed_macro_count = 0
    seed_first_count = 0
    data_count = len(data_list)
    for data in data_list:
        # 统计提取出关系触发词与标记完全一致的个数
        if hasattr(data, 'trigger_list') and ' '.join([str(idx) for idx in data.trigger_list]) == data.trigger_words:
            micro_count += 1
        if int(data.trigger_words.split()[0]) == data.trigger_seed[1]:
            seed_first_count += 1
        
        # 统计提取出关系触发词种子正确个数
        if hasattr(data, 'postag_list'):
            postag_list = getattr(data, 'postag_list')
            seed_is_vb = postag_list[data.trigger_seed[1]][1].startswith('VB')
            trigger_has_vb = any([postag_list[int(x)][1].startswith('VB') for x in data.trigger_words.split()])
            seed_is_nn = postag_list[data.trigger_seed[1]][1].startswith('NN')
            trigger_has_nn = any([postag_list[int(x)][1].startswith('NN') for x in data.trigger_words.split()])
            seed_is_jj = postag_list[data.trigger_seed[1]][1].startswith('JJ')
            trigger_has_jj = any([postag_list[int(x)][1].startswith('JJ') for x in data.trigger_words.split()])

            if any([
                all([seed_is_vb, trigger_has_vb]),
                all([seed_is_nn, trigger_has_nn]),
                all([seed_is_jj, trigger_has_jj]),
                not any([
                   seed_is_vb, trigger_has_vb,
                   seed_is_nn, trigger_has_nn,
                   seed_is_jj, trigger_has_jj
                ])
            ]):
                if str(data.trigger_seed[1]) in data.trigger_words.split():
                    seed_micro_count += 1
        
        # 统计提取出的关系触发词种子包含在标记的个数
        temp_trigger_words = set([int(x) for x in data.trigger_words.split()])
        if data.trigger_seed[1] in temp_trigger_words:
            seed_macro_count += 1
        
        # 统计提取出的关系触发词中部分出现在标记中的个数
        if hasattr(data, 'trigger_list'):
            for trigger in data.trigger_list:
                if trigger in temp_trigger_words:
                    macro_count += 1
                    break
    accuracy = {
        'micro': micro_count / data_count * 100,
        'macro': macro_count / data_count * 100,
        'seed_micro': seed_micro_count / data_count * 100,
        'seed_macro': seed_macro_count / data_count * 100,
        'seed_first': seed_first_count / data_count * 100,
    }
    return tuple(accuracy[kind] for kind in kind_list)


def print_running_time(*args, **kwargs):
    '''
    在程序运行结束后显示运行时间，用@print_running_time修饰在函数上。
    可选参数：
    :param  show_func_name  True / False
    :param  message     自定义输出信息，在需要输出时间的地方用{:f},
    '''
    from time import time
    if len(args) == 1 and len(kwargs) == 0:
        def _func(*fcargs, **fckwargs):
            tic = time()
            func = args[0]
            return_val = func(*fcargs, **fckwargs)
            toc = time()
            print("程序运行时间：{:f} 秒。".format(toc - tic))
            return return_val
        return _func

    if len(args) == 0 and len(kwargs) != 0:
        def _func(func):
            def __func(*fcargs, **fckwargs):
                tic = time()
                return_val = func(*fcargs, **fckwargs)
                toc = time()
                func_name = '程序'
                if "message" in kwargs:
                    print(kwargs["message"].format(toc - tic))
                else:
                    if 'show_func_name' in kwargs and kwargs['show_func_name']:
                        func_name = "函数 " + func.__name__ + " "
                    print("{:s}运行时间：{:f} 秒。".format(func_name, toc - tic))
                return return_val
            return __func
        return _func
