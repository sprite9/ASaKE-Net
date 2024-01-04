import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer,BertConfig
import numpy as np
from processors.processor import LEBertProcessor


def find_idx(token, target):
    target_length = len(target)
    for k, v in enumerate(token):
        if token[k: k + target_length] == target:
            return k
    return -1


class REDataset(Dataset):
    def __init__(self, config, file,processor, is_test=False):
        self.config = config
        with open(file, "r", encoding="utf-8") as f:
            self.data = json.load(f, strict=False)
        with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
            self.rel2id = json.load(fs, strict=False)[0]
        with open(self.config.tags) as ft:
            self.tag2id = json.load(ft, strict=False)[1]
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.is_test = is_test
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json_data = self.data[idx]
        sentence = ins_json_data["text"]
        triple = ins_json_data["spo_list"]
        token = ['[CLS]'] + list(sentence)[:self.config.max_len] + ['[SEP]']
        token_len = len(token)
        token2id = self.tokenizer.convert_tokens_to_ids(token)
        input_ids = np.array(token2id)
        mask = [0] * token_len
        mask = np.array(mask) + 1
        mask_len = len(mask)
        loss_mask = np.ones((mask_len, mask_len))


#
        char_index2words = self.get_char2words(list(sentence))
        word_ids_list = []
        word_pad_id = self.processor.word_vocab.convert_token_to_id('[PAD]')
        for words in char_index2words:
            words = words[:self.config.max_word_num]
            word_ids = self.processor.word_vocab.convert_tokens_to_ids(words)
            word_pad_num = self.config.max_word_num - len(words)
            word_ids = word_ids + [word_pad_id] * word_pad_num
            word_ids_list.append(word_ids)
        # 开头和结尾进行padding
        word_ids_list = [[word_pad_id] * self.config.max_word_num] + word_ids_list + [[word_pad_id] * self.config.max_word_num]

        if len(input_ids) > self.config.max_len:
            input_ids = input_ids[: self.config.max_len]
            word_ids_list = word_ids_list[: self.config.max_len]
        assert len(input_ids)  == len(word_ids_list)

        if not self.is_test:
            s2po = {}
            for spo in triple:
                triple_tuple = (list(spo[0]), spo[1], list(spo[2]))
                sub_head = find_idx(token, triple_tuple[0])
                obj_head = find_idx(token, triple_tuple[2])
                if sub_head != -1 and obj_head != -1:
                    sub = (sub_head, sub_head + len(triple_tuple[0]) - 1)
                    obj = (obj_head, obj_head + len(triple_tuple[2]) - 1, self.rel2id[triple_tuple[1]])
                    if sub not in s2po:
                        s2po[sub] = []
                    s2po[sub].append(obj)

            if len(s2po) > 0:

                matrix = np.zeros((self.config.num_rel, token_len, token_len))
                for sub in s2po:
                    sub_head = sub[0]
                    sub_tail = sub[1]
                    for obj in s2po.get((sub_head, sub_tail), []):
                        obj_head, obj_tail, rel = obj
                        matrix[rel][sub_head][obj_head] = self.tag2id["HB-TB"]
                        matrix[rel][sub_head][obj_tail] = self.tag2id["HB-TE"]
                        matrix[rel][sub_tail][obj_tail] = self.tag2id["HE-TE"]
                return sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask,word_ids_list
            else:
                return None
        else:
            # token2id = self.tokenizer.convert_tokens_to_ids(token)
            # input_ids = np.array(token2id)
            # mask = [0] * token_len
            # mask = np.array(mask) + 1
            # mask_len = len(mask)
            # loss_mask = np.ones((mask_len, mask_len))
            matrix = np.zeros((self.config.num_rel, token_len, token_len))
            return sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask,word_ids_list

    def get_char2words(self, text):
        """
        获取每个汉字，对应的单词列表
        :param text:
        :return:
        """
        text_len = len(text)
        char_index2words = [[] for _ in range(text_len)]
        for idx in range(text_len):
            sub_sent = text[idx:idx + self.processor.trie_tree.max_depth]  # speed using max depth
            words = self.processor.trie_tree.enumerateMatch(sub_sent)  # 找到以text[idx]开头的所有单词
            for word in words:
                start_pos = idx
                end_pos = idx + len(word)
                for i in range(start_pos, end_pos):
                    char_index2words[i].append(word)
        # todo 截断
        # for i, words in enumerate(char_index2words):
        #     char_index2words[i] = char_index2words[i][:self.max_word_num]
        return char_index2words


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[4], reverse=True)
    sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask,word_ids_list = zip(*batch)
    # word_ids_list = np.array(word_ids_list)
    # word_mask = (word_ids_list != 0).astype(int)


    cur_batch = len(batch)
    max_token_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_token_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_batch, max_token_len).zero_()
    batch_loss_mask = torch.LongTensor(cur_batch, 1, max_token_len, max_token_len).zero_()
    batch_matrix = torch.LongTensor(cur_batch, 13, max_token_len, max_token_len).zero_()
    batch_word_ids = torch.LongTensor(cur_batch,max_token_len,3).zero_()
    batch_word_mask = torch.LongTensor(cur_batch, max_token_len, 3).zero_()
    for i in range(cur_batch):
        word_ids = torch.LongTensor(word_ids_list[i])
        word_mask = (word_ids != 0).long()

        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_attention_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_loss_mask[i, 0, :token_len[i], :token_len[i]].copy_(torch.from_numpy(loss_mask[i]))
        batch_matrix[i, :, :token_len[i], :token_len[i]].copy_(torch.from_numpy(matrix[i]))
        batch_word_ids[i, :token_len[i]]=word_ids
        batch_word_mask[i, :token_len[i]]=word_mask



    return {"sentence": sentence,
            "token": token,
            "triple": triple,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "matrix": batch_matrix,
            "word_ids":batch_word_ids,
            "word_mask":batch_word_mask
            }

