import os
from loguru import logger
import torch
from tqdm import trange, tqdm
import numpy as np
from utils.utils import write_pickle, load_pickle
from utils.utils import load_lines, write_lines
from processors.trie_tree import Trie
import json
from processors.vocab import Vocabulary
from os.path import join

from transformers import BertTokenizer


#构建词表
class Processor(object):
    def __init__(self, config):
        self.data_path = config.data_path
        self.overwrite = config.overwrite
        self.train_file = config.train_file
        self.dev_file = config.dev_file

class LEBertProcessor(Processor):
    def __init__(self, config, tokenizer):
        super(LEBertProcessor, self).__init__(config)
        self.train_file = config.train_file
        self.dev_file = config.dev_file
        self.data_path = config.data_path
        self.overwrite = config.overwrite
        self.tokenizer = tokenizer
        data_files = [self.train_file, self.dev_file]
        self.word_embedding, self.word_vocab, self.trie_tree = self.init(
            config.pretrain_embed_path,config.max_scan_num, data_files,self.overwrite
        )

    def init(self, pretrain_embed_path,  max_scan_num, data_files, overwrite):
        word_embed_path = join(self.data_path, 'word_embedding.pkl')
        word_vocab_path = join(self.data_path, 'word_vocab.pkl')
        word_vocab_path_ = join(self.data_path, 'word_vocab.txt')
        trie_tree_path = join(self.data_path, 'trie_tree.pkl')

        if overwrite or not os.path.exists(word_embed_path) or not os.path.exists(word_vocab_path):
            # 加载词向量
            word_embed_dict, word_list, word_embed_dim = self.load_word_embedding(pretrain_embed_path, max_scan_num)
            # 构建字典树
            trie_tree = self.build_trie_tree(word_list, trie_tree_path)
            # 找到数据集中的所有单词
            corpus_words = self.get_words_from_corpus(data_files, word_vocab_path_, trie_tree)
            # 初始化模型的词向量
            model_word_embedding, word_vocab, embed_dim = self.init_model_word_embedding(corpus_words, word_embed_dict, word_embed_path, word_vocab_path)
        else:
            model_word_embedding = load_pickle(word_embed_path)
            word_vocab = load_pickle(word_vocab_path)
            trie_tree = load_pickle(trie_tree_path)
        return model_word_embedding, word_vocab, trie_tree

    @classmethod
    def load_word_embedding(cls, word_embed_path, max_scan_num):
        """
        todo 存在许多单字的，考虑是否去掉
        加载前max_scan_num个词向量, 并且返回词表
        :return:
        """
        logger.info('loading word embedding from pretrain')
        word_embed_dict = dict()
        word_list = list()

        with open(word_embed_path, 'r', encoding='utf8') as f:
            for idx, line in tqdm(enumerate(f)):
                # 只扫描前max_scan_num个词向量
                if idx > max_scan_num:
                    break
                items = line.strip().split()
                if idx == 0:
                    assert len(items) == 2
                    num_embed, word_embed_dim = items
                    num_embed, word_embed_dim = int(num_embed), int(word_embed_dim)
                else:
                    assert len(items) == word_embed_dim + 1
                    word = items[0]
                    embedding = np.empty([1, word_embed_dim])
                    embedding[:] = items[1:]
                    word_embed_dict[word] = embedding
                    word_list.append(word)
        logger.info('word_embed_dim:{}'.format(word_embed_dim))
        logger.info('size of word_embed_dict:{}'.format(len(word_embed_dict)))
        logger.info('size of word_list:{}'.format(len(word_list)))

        return word_embed_dict, word_list, word_embed_dim

    @classmethod
    def build_trie_tree(cls, word_list, save_path):
        """
        # todo 是否不将单字加入字典树中
        构建字典树
        :return:
        """
        logger.info('building trie tree')
        trie_tree = Trie()
        for word in word_list:
            trie_tree.insert(word)
        write_pickle(trie_tree, save_path)
        return trie_tree

    @classmethod
    def get_words_from_corpus(cls, files, save_file, trie_tree):
        """
        找出文件中所有匹配的单词
        :param files:
        :return:
        """
        logger.info('getting words from corpus')
        all_matched_words = set()
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                datas = json.load(f, strict=False)
                for index, data in enumerate(datas):
                    text = data['text']
                    text = list(text)
                    matched_words = cls.get_words_from_text(text, trie_tree)
                    _ = [all_matched_words.add(word) for word in matched_words]

        all_matched_words = list(all_matched_words)
        all_matched_words = sorted(all_matched_words)
        write_lines(all_matched_words, save_file)
        return all_matched_words

    @classmethod
    def get_words_from_text(cls, text, trie_tree):
        """
        找出text中所有的单词
        :param text:
        :param trie_tree:
        :return:
        """
        length = len(text)
        matched_words_set = set()   # 存储匹配到的单词
        for idx in range(length):
            sub_text = text[idx:idx + trie_tree.max_depth]
            words = trie_tree.enumerateMatch(sub_text)

            _ = [matched_words_set.add(word) for word in words]
        matched_words_set = list(matched_words_set)
        matched_words_set = sorted(matched_words_set)
        return matched_words_set

    def init_model_word_embedding(self, corpus_words, word_embed_dict, save_embed_path, save_word_vocab_path):
        logger.info('initializing model word embedding')
        # 构建单词和id的映射
        word_vocab = Vocabulary(corpus_words, vocab_type='word')
        # embed_dim = len(word_embed_dict.items()[1].size)
        embed_dim = next(iter(word_embed_dict.values())).size

        scale = np.sqrt(3.0 / embed_dim)
        model_word_embedding = np.empty([word_vocab.size, embed_dim])

        matched = 0
        not_matched = 0

        for idx, word in enumerate(word_vocab.idx2token):
            if word in word_embed_dict:
                model_word_embedding[idx, :] = word_embed_dict[word]
                matched += 1
            else:
                model_word_embedding[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                not_matched += 1

        logger.info('num of match:{}, num of not_match:{}'.format(matched, not_matched))
        write_pickle(model_word_embedding, save_embed_path)
        write_pickle(word_vocab, save_word_vocab_path)

        return model_word_embedding, word_vocab, embed_dim

    #
