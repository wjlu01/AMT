#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings
        self.encoder = bert.encoder
        if self.args.model!="AMT":
            self.pooler = bert.pooler

    def forward(self, txt, mask, segment):
        bsz = txt.size(0)
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2) #[batch, 1, 1, len]
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        txt_embed_out = self.txt_embeddings(txt, segment)

        encoded_layers = self.encoder(
            txt_embed_out, extended_attention_mask, output_all_encoded_layers=False
        ) #len=1

        return encoded_layers[-1] if self.args.model=="AMT" else self.pooler(encoded_layers[-1])


class BertClf(nn.Module):
    # deal OTGT
    
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        return self.clf(x)
