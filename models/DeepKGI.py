import torch
import torch.nn as nn
from .CLGF import CLGF_GNNDrug
from .GNN_cell import GNN_cell
import torch.nn.functional as F
from collections import Counter
import re
import json
import os


class BPETokenizer:

    def __init__(self, vocab_size=1000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

    def _get_stats(self, tokens):
        # 计算token对的频率
        pairs = Counter()
        for token in tokens:
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_vocab(self, tokens, best_pair):
        # 合并最佳token对
        new_tokens = []
        for token in tokens:
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                    new_token.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_tokens.append(new_token)
        return new_tokens

    def train(self, smiles_list):
        """训练BPE分词器"""
        vocab = Counter()
        for smiles in smiles_list:
            vocab.update(list(smiles))
        for token in self.special_tokens:
            vocab[token] = 0

        self.vocab = {char: i + len(self.special_tokens) for i, char in enumerate(vocab)}
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        tokens = [list(smiles) for smiles in smiles_list]
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self._get_stats(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            tokens = self._merge_vocab(tokens, best_pair)

            # 更新词汇表
            new_token = best_pair[0] + best_pair[1]
            new_index = len(self.vocab)
            self.vocab[new_token] = new_index
            self.inverse_vocab[new_index] = new_token
            self.merges[best_pair] = new_token

    def tokenize(self, smiles):
        """将SMILES字符串token化"""
        tokens = list(smiles)

        for pair, merge in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, smiles):
        tokens = self.tokenize(smiles)

        # 添加开始和结束标记
        tokens = ["<bos>"] + tokens + ["<eos>"]

        ids = [self.vocab.get(token, self.special_tokens["<unk>"]) for token in tokens]

        # 填充或截断到固定长度
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        elif len(ids) < self.max_length:
            ids = ids + [self.special_tokens["<pad>"]] * (self.max_length - len(ids))

        return ids

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length
            }, f, indent=2)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.vocab = {k: int(v) for k, v in data['vocab'].items()}
            self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
            self.merges = data['merges']
            self.special_tokens = {k: int(v) for k, v in data['special_tokens'].items()}
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']


class DrugSequenceEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence):
        # sequence: (batch_size, seq_length)
        embedded = self.embedding(sequence)  # (batch_size, seq_length, embedding_dim)

        lengths = (sequence != 0).sum(dim=1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, hn = self.gru(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        last_output = output[torch.arange(output.size(0)), lengths - 1]

        return self.fc(last_output)  # (batch_size, hidden_dim)


class DeepKGI(nn.Module):
    def __init__(self, cluster_predefine, args, drug_smiles_list=None, bpe_model_path=None):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.num_feature = args.num_feature
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

        # BPE分词器
        self.bpe_tokenizer = BPETokenizer(vocab_size=1000, max_length=100)

        # 加载或训练BPE分词器
        if bpe_model_path and os.path.exists(bpe_model_path):
            print(f"加载预训练的BPE分词器: {bpe_model_path}")
            self.bpe_tokenizer.load(bpe_model_path)
        elif drug_smiles_list:
            print("训练新的BPE分词器...")
            self.bpe_tokenizer.train(drug_smiles_list)
            if bpe_model_path:
                self.bpe_tokenizer.save(bpe_model_path)
                print(f"BPE分词器已保存至: {bpe_model_path}")
        else:
            raise ValueError("需要提供drug_smiles_list或bpe_model_path")

        self.drug_sequence_encoder = DrugSequenceEncoder(
            vocab_size=len(self.bpe_tokenizer.vocab),
            embedding_dim=64,
            hidden_dim=64,
            num_layers=1
        )

        self.GNN_drug = CLGF_GNNDrug(self.dim_drug)

        self.drug_feature_fusion = nn.Sequential(
            nn.Linear((self.dim_drug * 9) + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio)
        )

        self.GNN_cell = GNN_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)
        print("Final_node", self.GNN_cell.final_node)
        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(128, 1),
        )

        # 6. 特征交互层
        self.linear1 = nn.Linear(128, 64, bias=False)
        self.linear2 = nn.Linear(64, 128, bias=False)

    def encode_smiles(self, smiles_list):
        """编码SMILES字符串列表为张量"""
        return torch.tensor([self.bpe_tokenizer.encode(smiles) for smiles in smiles_list])

    def forward(self, drug, cell):
        # drug: 包含图数据和SMILES字符串的元组
        drug_graph, drug_smiles = drug

        x_drug_graph = self.GNN_drug(drug_graph)

        drug_sequences = self.encode_smiles(drug_smiles).to(x_drug_graph.device)
        x_drug_seq = self.drug_sequence_encoder(drug_sequences)

        x_drug = torch.cat([x_drug_graph, x_drug_seq], dim=1)
        x_drug = self.drug_feature_fusion(x_drug)

        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        x_mid = self.linear1(x_drug)
        x_ = self.linear2(x_mid)

        x = torch.cat([
            x_cell * x_,
            x_cell + x_,
            x_,
            x_mid,
            x_drug,
            x_cell,
            x_cell + x_drug,
            x_cell * x_drug
        ], dim=-1)

        return self.regression(x)