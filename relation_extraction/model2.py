import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding,
    Dropout,
    Bidirectional,
    LSTM,
    Layer,
    Dense
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from utils import MyWord2VecPKL


class BiLstmAttr:

    def __init__(self, config, model_file=None):

        self.model_file = model_file

        # 是否使用关系触发词特征
        self.use_trigger_words = config.get('use_trigger_words', False)
        
        # 句子的最大长度
        self.max_words = config.get('max_words')
        # 单词的向量维度
        self.word_dim = config.get('word_dim')
        # 词汇表长度
        self.word_num = config.get('word_num')
        # 类别个数
        self.n_classes = config.get('n_classes')
        # embedding矩阵
        self.embedding_matrix = config.get('embedding_matrix')
        # dropout比率
        self.dropout = config.get('dropout', 0.5)
        # 学习率
        self.learning_rate = config.get('learning_rate', 0.001)
        # batch size
        self.batch_size = config.get('batch_size', 200)
        # 迭代次数
        self.epoches = config.get('epoches', 50)
        # 是否显示训练过程
        self.verbose = config.get('verbose', True)
        # 准确率精度
        self.digits = config.get('digits', 4)

        # 创建模型
        self.model = Sequential()

        # Embedding层
        self.model.add(Embedding(
            input_dim=self.word_num,
            output_dim=self.word_dim,
            input_length=self.max_words,
            trainable=False,
            weights=[self.embedding_matrix],
            name='embedding'
        ))

        # 第一层Dropout
        self.model.add(Dropout(self.dropout, name='dropout_1'))
        # 双向LSTM
        self.model.add(Bidirectional(LSTM(self.word_dim, return_sequences=True), merge_mode='sum'))
        # 第二层Dropout
        self.model.add(Dropout(self.dropout, name='dropout_2'))
        # Attention层
        self.model.add(AttentionLayer(name='attention_layer'))
        # 第三层Dropout
        self.model.add(Dropout(self.dropout, name='dropout_3'))
        # 分类层
        self.model.add(Dense(self.n_classes, activation='softmax'))
        # 编译
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"], )

    def fit(self, X_train, y_train):
        import os
        if not self.model_file or (self.model_file and not os.path.exists(self.model_file)):
            self.model.fit(X_train, y_train, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epoches)
            self.train_score = self.model.evaluate(X_train, y_train)
            if self.model_file:
                self.model.save(self.model_file)
        else:
            # custom_objects {'Layer Name Class': LayerNameClass}
            self.model = load_model(self.model_file, custom_objects={'AttentionLayer': AttentionLayer})

    def predict(self, X_test, y_test=None):
        self.y_pred = self.model.predict(X_test)
        if y_test is not None:
            self.test_score = self.model.evaluate(X_test, y_test)
            self.classify_report = metrics.classification_report(y_test, np.argmax(self.y_pred, axis=1), digits=self.digits)
    



class AttentionLayerExample(Layer):

    def __init__(self, attention_size=None, *args, **kwargs):
        self.attention_size = attention_size
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
    
    def build(self, input_shape):
        # input_shape (M, max_words, word_dim)
        assert len(input_shape) == 3
        # max_words
        self.time_steps = input_shape[1]
        # word_dim
        self.hidden_size = input_shape[2]

        if not self.attention_size:
            self.attention_size = self.hidden_size
        # self.w (word_dim, word_dim)
        self.W = self.add_weight(name='att_weight', shape=(self.hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        # self.b (word_dim, )
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        # self.V (word_dim, )
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs (None, max_words, word_dim)
        # self.V (word_dim, 1)
        self.V = K.reshape(self.V, (-1, 1))
        
        # H (None, max_words, word_dim)
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # Score (None, max_words, 1)
        score = K.softmax(K.dot(H, self.V), axis=1)
        # outputs (None, word_dim)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



class AttentionLayer(Layer):

    def __init__(self, attention_size=None, use_trigger_words=False, *args, **kwargs):
        self.attention_size = attention_size
        self.use_trigger_words = use_trigger_words
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        config['use_trigger_words'] = self.use_trigger_words
        return config
    
    def build(self, input_shape):
        # input_shape (M, max_words, word_dim)
        assert len(input_shape) == 3
        # max_words
        self.time_steps = input_shape[1]
        # word_dim
        self.hidden_size = input_shape[2]

        if not self.attention_size:
            self.attention_size = self.hidden_size
        # self.w (word_dim, word_dim)
        self.W = self.add_weight(name='att_weight', shape=(self.hidden_size, 1),
                                initializer='uniform', trainable=True)
        # self.trigger_words (None, max_words, 1)
        self.trigger_words = Input(shape=(self.time_steps, 1), name='trigger_words')
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs (None, max_words, word_dim)
        # self.M (None, max_words, word_dim)
        self.M = K.tanh(inputs)
        # self.alpha (None, max_words, 1)
        self.alpha = K.softmax(K.dot(self.M, self.W), axis=1)

        if self.use_trigger_words:
            self.alpha *= self.trigger_words

        # self.r (None, word_dim)
        self.r = K.sum(self.alpha * inputs, axis=1)
        return K.tanh(self.r)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


if __name__ == "__main__":
    myword2vecpkl = MyWord2VecPKL.getMyWord2vecPKL()
    config = {
        'max_words': 100,
        'word_num': myword2vecpkl.word_num,
        'word_dim': myword2vecpkl.word_dim,
        'embedding_matrix': myword2vecpkl.embedding_matrix,
        'n_classes': 10,
        'use_trigger_words': True
    }
    bilstmattr = BiLstmAttr(config)
