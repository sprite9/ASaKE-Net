
class Configargs():
    def __init__(self):
        self.num_rel = 13  #关系数量
        self.train_file = "./dataset/AlSiAlloyData/train_data.json"
        #self.dev_file = "./dataset/AlSiAlloyData/typeData/normal.json"
        #self.dev_file = "./dataset/AlSiAlloyData/numberData/number_4.json"
        self.dev_file = "./dataset/AlSiAlloyData/dev_data.json"
        self.schema_fn = "./dataset/AlSiAlloyData/schema.json"
        #self.bert_path = "./bert-base-chinese"
        self.bert_path = "./chinese_roberta_L-12_H-768"

        self.step = 175
        self.tags = "./dataset/tag2id.json"
        self.bert_dim = 768
        self.tag_size = 4
        self.batch_size = 4
        self.max_len = 195   #最大输入序列长度
        self.learning_rate = 1e-5
        self.epochs = 200
        self.checkpoint = "checkpoint/type/model_roberta.pt"  #保存模型位置
        self.dev_result = "dev_result/type/result.json"
        self.dropout_prob = 0.1
        self.entity_pair_dropout = 0.2
        self.bidirectional=True

        self.dataset = "AlSiAlloyData"
        self.log = "log/roberta/{}_bert_word(2)_all.log".format(self.dataset)


        self.max_word_num = 1     #最大词组数量
        self.data_path = "./dataset/AlSiAlloyData/word_v1"   #构建此表保存位置
        self.overwrite=False   #是否保存到指定路径
        self.max_scan_num = 4000000    #
        self.pretrain_embed_path = './word_embedding/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt'
        self.add_layer = 1   #在Bert的第几层融合词向量
        self.eps = 1.0e-08   #权重衰减
        self.warm_up_ratio = 0.1
