import torch
import torch.nn as nn
from torch.autograd import Variable
from models.lebert import LEBertModel
from config.config import Configargs

class ASaRE(nn.Module):
    def __init__(self,config):
        super(ASaRE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configargs = Configargs()
        self.use_cuda=True
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim).to(self.device)
        self.bert = LEBertModel.from_pretrained(pretrained_model_name_or_path=self.configargs.bert_path,config=config).to(self.device)
        #self.bert = BertModel.from_pretrained(self.configargs.bert_path)
        # Multihead attention:
        self.mha = nn.MultiheadAttention(self.configargs.bert_dim, num_heads=13).to(self.device)
        self.linear1 = nn.Linear(self.configargs.bert_dim * 2, self.configargs.bert_dim).to(self.device)
        self.relation_linear = nn.Linear(self.configargs.bert_dim * 3, self.configargs.num_rel * self.configargs.tag_size).to(self.device)
        self.project_matrix = nn.Linear(self.configargs.bert_dim * 2, self.configargs.bert_dim * 3).to(self.device)
        self.dropout = nn.Dropout(0.2).to(self.device)
        self.dropout_2 = nn.Dropout(0.1).to(self.device)
        self.activation = nn.ReLU().to(self.device)


    def get_encoded_text(self, input_ids, mask,word_ids,word_mask):
        word_embeddings=self.word_embeddings(word_ids)
        bert_encoded_text = self.bert(input_ids=input_ids, attention_mask=mask,  word_embeddings=word_embeddings, word_mask=word_mask)
        return bert_encoded_text

    # def rand_init_hidden(self, batch_size):
    #     if self.use_cuda:
    #         return Variable(
    #             torch.randn(2 * self.configargs.rnn_layers, batch_size, self.configargs.hidden_dim)).cuda(), Variable(
    #             torch.randn(2 * self.configargs.rnn_layers, batch_size, self.configargs.hidden_dim)).cuda()
    #     else:
    #         return Variable(
    #             torch.randn(2 * self.configargs.rnn_layers, batch_size, self.configargs.hidden_dim)), Variable(
    #             torch.randn(2 * self.configargs.rnn_layers, batch_size, self.configargs.hidden_dim))

    def get_triple_score(self, sequence_output,train):
        batch_size, seq_len, bert_dim = sequence_output.size()
        #pooler_output = pooler_output.unsqueeze(dim=1).expand(batch_size, seq_len, bert_dim)

        sequence_output, attn_output_weights1 = self.mha(sequence_output, sequence_output, sequence_output)

        head_rep = sequence_output.unsqueeze(dim=2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size,seq_len * seq_len,bert_dim)
        tail_rep = sequence_output.repeat(1, seq_len, 1)
        # [batch_size, seq_len*seq_len, bert_dim * 2]
        entity_pair = torch.cat([head_rep, tail_rep], dim=-1)

        # [batch_size, seq_len*seq_len, bert_dim * 3]
        entity_pair = self.project_matrix(entity_pair)
        entity_pair = self.dropout_2(entity_pair)
        entity_pair = self.activation(entity_pair)

        # [batch_size, seq_len*seq_len, num_rel*tag_size]
        matrix_socre = self.relation_linear(entity_pair).reshape(batch_size, seq_len, seq_len, self.configargs.num_rel, self.configargs.tag_size)
        if train:
            return matrix_socre.permute(0, 4, 3, 1, 2)
        else:
            return matrix_socre.argmax(dim=-1).permute(0, 3, 1, 2)

    def get_triple_score_test(self, sequence_output,train):
        batch_size, seq_len, bert_dim = sequence_output.size()
        head_rep = sequence_output.unsqueeze(dim=2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size,seq_len * seq_len,bert_dim)
        tail_rep = sequence_output.repeat(1, seq_len, 1)
        # [batch_size, seq_len*seq_len, bert_dim * 2]
        entity_pair = torch.cat([head_rep, tail_rep], dim=-1)

        # [batch_size, seq_len*seq_len, bert_dim * 3]
        entity_pair = self.project_matrix(entity_pair)
        entity_pair = self.dropout_2(entity_pair)
        entity_pair = self.activation(entity_pair)

        # [batch_size, seq_len*seq_len, num_rel*tag_size]
        matrix_socre = self.relation_linear(entity_pair).reshape(batch_size, seq_len, seq_len, self.configargs.num_rel, self.configargs.tag_size)
        if train:
            return matrix_socre.permute(0, 4, 3, 1, 2)
        else:
            return matrix_socre.argmax(dim=-1).permute(0, 3, 1, 2)

    def forward(self, data, train=True):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        word_ids=data["word_ids"].to(self.device)
        word_mask=data["word_mask"].to(self.device)
        bert_encoded_text = self.get_encoded_text(input_ids, attention_mask,word_ids,word_mask)
        sequence_output = self.dropout(bert_encoded_text[0])
        #pooler_output = bert_encoded_text[1]  #
        matrix_score = self.get_triple_score_test(sequence_output,train)
        return matrix_score