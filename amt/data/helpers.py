#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from data.dataset import JsonlDataset
from data.vocab import Vocab


def get_transforms(args):
    if args.dataset=="bloomberg":
        sz=256
    else:
        sz=299
    return transforms.Compose(
        [
            transforms.Resize(sz),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(args,path):
    label_freqs = Counter()
    data_labels = [json.loads(line)[args.label] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)
    labels=list(label_freqs.keys())
    labels.sort()
    return labels, label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()

    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch, args):  # batch : sentence, segment, image, label, caption_image

    ## deal image
    img_tensor = torch.stack([row[-2] for row in batch]) # [4, 3, 224, 224]

    ## deal label
    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[-1] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[-1] for row in batch]).long()

    # deal text
    lens = [len(row[0]) for row in batch] 
    bsz, max_seq_len= len(batch), max(lens)

    mask_tensor1 = torch.zeros(bsz, max_seq_len).long()
    text_tensor1 = torch.zeros(bsz, max_seq_len).long()
    segment_tensor1 = torch.zeros(bsz, max_seq_len).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor1[i_batch, :length] = tokens
        segment_tensor1[i_batch, :length] = segment
        mask_tensor1[i_batch, :length] = 1

    # deal text&caption
    lens = [len(row[2]) for row in batch] 
    bsz, max_seq_len= len(batch), max(lens)

    mask_tensor2 = torch.zeros(bsz, max_seq_len).long()
    text_tensor2 = torch.zeros(bsz, max_seq_len).long()
    segment_tensor2 = torch.zeros(bsz, max_seq_len).long()
    segment_tensor3 = torch.zeros(bsz, max_seq_len).long()
    
    txt_indexs_tensor = torch.zeros(bsz,2).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[2:4]
        text_tensor2[i_batch, :length] = tokens
        segment_tensor2[i_batch, :length] = segment
        segment_tensor3[i_batch, :length] = segment+1
        mask_tensor2[i_batch, :length] = 1
        
        txt_index = input_row[4]
        txt_indexs_tensor[i_batch,0]=txt_index
        txt_indexs_tensor[i_batch,1]=length-txt_index
        
    

    return text_tensor1, segment_tensor1, mask_tensor1,text_tensor2, segment_tensor2, segment_tensor3, mask_tensor2, txt_indexs_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(
            args,os.path.join(args.data_path, args.dataset, "train.jsonl")
        )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "train.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "dev.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.dataset, "test.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )


    test = {
        "test": test_loader
        }

    return train_loader, val_loader, test
