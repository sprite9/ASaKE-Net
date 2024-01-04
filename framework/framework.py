import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader import REDataset, collate_fn
from models.models import ASaKE
from logger.logger import Logger
from processors.processor import LEBertProcessor
from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup
import torchvision


class Framework():
    def __init__(self, config):
        self.config = config
        with open(self.config.tags, "r", encoding="utf-8") as f:
            self.tag2id = json.load(f)[1]
        with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
            self.id2rel = json.load(fs)[1]
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path,do_lower_case=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log = Logger(self.config.log)

    def build_processor(self):
        processor = LEBertProcessor(self.config, self.tokenizer)  # 构建数据集词表，并获取对应词向量
        # 在bert中融入词向量
        bert_config = BertConfig.from_pretrained(self.config.bert_path)
        bert_config.add_layer = self.config.add_layer
        bert_config.word_vocab_size = processor.word_embedding.shape[0]
        bert_config.word_embed_dim = processor.word_embedding.shape[1]  # 2162 200

        return processor,bert_config


    def train(self):
        def cal_loss(predict, target, mask):
            loss_ = self.loss_function(predict, target)
            loss = torch.sum(loss_ * mask) / torch.sum(mask)
            return loss

        #
        processor,bert_config=self.build_processor()
        #初始化模型
        model = ASaKE(bert_config)  # 加载模型
        model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))

        ##############

        train_dataset = REDataset(self.config, self.config.train_file,processor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)

        dev_dataset = REDataset(self.config, self.config.dev_file,processor)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_fn)

        #t_total = (len(train_dataloader) // self.config.batch_size) * self.config.epochs
        #model = ASaKE(self.config).to(self.device)   #加载模型
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,eps=self.config.eps)
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warm_up_ratio * t_total, num_training_steps=t_total)

        global_step = 0
        global_loss = 0
        best_epoch = 0

        best_f1_score = 0
        best_recall = 0
        best_precision = 0

        for epoch in range(self.config.epochs):
            print("[{}/{}]".format(epoch+1, self.config.epochs))
            for data in tqdm(train_dataloader):
                output = model(data).to(self.device)
                optimizer.zero_grad()
                loss = cal_loss(output, data["matrix"].to(self.device), data["loss_mask"].to(self.device))
                global_loss += loss.item()
                loss.backward()
                optimizer.step()
               # scheduler.step()

                global_step += 1
                if global_step  % self.config.steps == 0:
                    self.log.logger.info("epoch: {} global_step: {:5.4f} global_loss: {:5.4f}".format(epoch+1, global_step, global_loss))
                    global_loss = 0
                    #print("epoch: {} global_step: {} global_loss: {:5.4f}".format(epoch + 1, global_step + 1, global_loss))

            if (epoch + 1) % 5 == 0:
                precision, recall, f1_score, predict = self.evaluate(dev_dataloader, model)
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_recall = recall
                    best_precision = precision
                    best_epoch = epoch + 1
                    print("save model ......")
                    self.log.logger.info("save model......")
                    torch.save(model.state_dict(), self.config.checkpoint)
                    json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                    print("epoch:{} best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(epoch+1, best_epoch, best_recall, best_precision, best_f1_score))
                    self.log.logger.info("epoch:{} best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(epoch+1, best_epoch, best_recall, best_precision, best_f1_score))

        print("best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(best_epoch, best_recall, best_precision, best_f1_score))
        self.log.logger.info("best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(best_epoch, best_recall, best_precision, best_f1_score))


    def evaluate(self, dataloader, model):
        print("eval mode......")
        self.log.logger.info("save model......")
        model.eval()
        predict_num, gold_num, correct_num = 0, 0, 0
        predict = []
        def to_ret(data):
            ret = []
            for i in data:
                ret.append(tuple(i))
            return tuple(ret)

        with torch.no_grad():
            for data in tqdm(dataloader):
                # [num_rel, seq_len, seq_len]
                pred_triple_matrix = model(data, train=False).cpu()[0]
                number_rel, seq_lens, seq_lens = pred_triple_matrix.shape
                relations, heads, tails = np.where(pred_triple_matrix > 0)

                token = data["token"][0]
                gold = data["triple"][0]
                pair_numbers = len(relations)
                predict_triple = []
                if pair_numbers > 0:
                    for i in range(pair_numbers):
                        r_index = relations[i]
                        h_start_idx = heads[i]
                        t_start_idx = tails[i]
                        if pred_triple_matrix[r_index][h_start_idx][t_start_idx] == self.tag2id["HB-TB"] and i + 1 < pair_numbers:
                            t_end_idx = tails[i + 1]
                            if pred_triple_matrix[r_index][h_start_idx][t_end_idx] == self.tag2id["HB-TE"]:
                                for h_end_index in range(h_start_idx, seq_lens):
                                    if pred_triple_matrix[r_index][h_end_index][t_end_idx] == self.tag2id["HE-TE"]:

                                        subject_head, subject_tail = h_start_idx, h_end_index
                                        object_head, object_tail = t_start_idx, t_end_idx
                                        subject = ''.join(token[subject_head: subject_tail + 1])
                                        object = ''.join(token[object_head: object_tail + 1])
                                        relation = self.id2rel[str(int(r_index))]
                                        if len(subject) > 0 and len(object) > 0:
                                            predict_triple.append((subject, relation, object))
                                        break
                gold = to_ret(gold)
                predict_triple = to_ret(predict_triple)
                gold_num += len(gold)
                predict_num += len(predict_triple)
                correct_num += len(set(gold) & set(predict_triple))
                lack = set(gold) - set(predict_triple)
                new = set(predict_triple) - set(gold)
                predict.append({"text": data["sentence"][0], "gold": gold, "predict": predict_triple,
                                "lack": list(lack), "new": list(new)})

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        print("predict_num: {} gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))
        self.log.logger.info("predict_num: {} gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))
        model.train()
        return precision, recall, f1_score, predict

    def test(self):
        processor, bert_config = self.build_processor()
        dev_dataset = REDataset(self.config, self.config.dev_file, processor)
        dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn, pin_memory=True)
        #模型加载
        print("load model......")
        model = ASaKE(bert_config)  # 加载模型
        #model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))
        model.load_state_dict(torch.load(self.config.checkpoint, map_location=self.device))
        model.to(self.device)
        precision, recall, f1_score, predict = self.evaluate(dev_dataloader,model)
        with open(self.config.test_result, "w", encoding="utf-8") as f:
            json.dump(predict, f, indent=4, ensure_ascii=False)
        print("test result!!!")
        print("precision:{:5.4f}, recall:{:5.4f}, f1_score:{:5.4f}".format(precision, recall, f1_score))


