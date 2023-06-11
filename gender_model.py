import json
import torch

import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from module import LSTMEncoder


class GenderModel(torch.nn.Module):
    def __init__(self, config_path, embedding_path):
        super(GenderModel, self).__init__()
        self.config = self._get_params(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_path)
        self.BertModel = AutoModel.from_pretrained(embedding_path)
        embed_out_dim = list(self.BertModel.parameters())[-1].shape[0]
        self.bi_lstm = LSTMEncoder(
            input_size=embed_out_dim,
            rnn_size=self.config["rnn_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bidirectional=self.config["Bidirectional"],
        )
        self.linear = torch.nn.Linear(self.config["rnn_size"] * 2, 2)

    def forward(self, input_text):
        tokens = []
        for word in input_text:
            token_ids = self.tokenizer.encode(word, add_special_tokens=True)
            tokens.append(token_ids)

        max_length = max(len(sequence) for sequence in tokens)
        token_lengths = torch.tensor([max_length for _ in tokens])
        padded_list = [
            sequence + [0] * (max_length - len(sequence)) for sequence in tokens
        ]
        input_ids = torch.tensor(padded_list)
        embedding = self.BertModel(input_ids)

        output, last_output = self.bi_lstm(embedding.last_hidden_state, token_lengths)
        logits = self.linear(output)
        logits = torch.mean(logits, dim=1)
        logits = F.softmax(logits, dim=1)
        return logits

    @staticmethod
    def _get_params(json_file):
        with open(json_file, "r") as file:
            jdata = json.load(file)
        return jdata


if __name__ == "__main__":
    model = GenderModel("./config.json", "bertmodel")
