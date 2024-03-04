
class Configargs():
    def __init__(self):
        self.num_rel = 13  #关系数量
        self.train_file = "./dataset/ASaIED/train_data.json"
        self.dev_file = "./dataset/ASaIED/dev_data.json"
        self.schema_fn = "./dataset/ASaIED/schema.json"

        self.bert_path = "./chinese_roberta_L-12_H-768"
        self.tags = "./dataset/tag2id.json"
        self.bert_dim = 768
        self.tag_size = 4
        self.batch_size = 4
        self.steps = 175
        self.max_len = 195   #最大输入序列长度
        self.learning_rate = 1e-5
        self.epochs = 200
        self.checkpoint = "checkpoint/ASaRE-Net_self.pt"  #保存模型位置
        self.dev_result = "dev_result/result.json"
        self.test_result = "dev_result/test.json"
        self.dropout_prob = 0.1
        self.entity_pair_dropout = 0.2

        self.dataset = "ASaIE"
        self.log = "log/{}_bert_log_test.log".format(self.dataset)

        self.max_word_num = 1     #最大词组数量
        self.data_path = "./dataset/ASaIED/word_v1"   #构建此表保存位置
        self.overwrite=False   #是否保存到指定路径,第一次构建词典时，需设置为true
        self.max_scan_num = 4000000    #
        self.pretrain_embed_path = './word_embedding/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt'
        self.add_layer = 1   #在Bert的第几层融合词向量
        self.eps = 1.0e-08
        self.warm_up_ratio = 0.1
