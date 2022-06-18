#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] 

        self.max_seq_len = args.max_seq_len - args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)


    def read_image(self,index,name):
        try:
            image = Image.open(
                    os.path.join(self.data_dir, self.data[index][name])
                ).convert("RGB")
        except:
            image = Image.fromarray(128 * np.ones((299, 299, 3), dtype=np.uint8))
            # print(index,name,self.data[index][name])
        image = self.transforms(image)
        return image


    def __getitem__(self, index):

        ## load label
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index][self.args.label]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index][self.args.label])]
            )

        ## load image
        image=self.read_image(index,"img")

        ## load text
        sentence1 = (
            self.tokenizer(self.data[index]["text"])[
                : (self.args.max_seq_len - 2)
            ]+
            ["[SEP]"]
        ) #[CLS] word token
        segment1 = torch.ones(len(sentence1))
        sentence1 = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence1
            ]
        ) 

        ## load text&caption
        sent1 = self.tokenizer(self.data[index]["text"])
        sent2 = self.tokenizer(self.data[index]["caption"])
        truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
        sentence2 = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
        segment2 = torch.cat(
            [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
        )
        sentence2 = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence2
            ]
        ) 
        
        txt_indexs=len(sent1)+2

        return sentence1, segment1, sentence2, segment2, txt_indexs, image, label
