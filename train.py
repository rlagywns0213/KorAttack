# This file is the pipeline of fine-tuing
# Author: Hongcheng Gao, Yangyi Chen
# Date: 2022-10-18 

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from custom_dataset import packDataset_util
import torch.nn as nn
import transformers
import pandas as pd
import os
from tqdm import tqdm

def load_data(file_path,data_type):
    if 'nsmc' in file_path:
        data = pd.read_csv(file_path+data_type, sep='\t')
        data = data.dropna() #결측치 제거 (5개 샘플)
        data.reset_index(drop=True, inplace=True)
    p_data = []
    if 'hate-speech' in file_path:
        data = pd.read_csv(file_path+data_type, sep='\t')
    for i in range(len(data)):
        if 'korean-hate-speech' in file_path:
            if data['contain_gender_bias'][i]:
                p_data.append((data['comments'][i], 1))
            else:
                p_data.append((data['comments'][i], 0))
        elif 'nsmc' in file_path:
            p_data.append((data['document'][i], data['label'][i]))       
    return p_data

def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks).logits
            _, flag = torch.max(output, dim=1)
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        return acc

def train():
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in tqdm(train_loader):
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks).logits
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            dev_acc = evaluaion(test_loader)
            print('finish evaluation, acc: {}/{}'.format(dev_acc, best_acc))
            if dev_acc > best_acc:
                best_acc = dev_acc
                model_path = "model_2023/kobert_nsmc"
                torch.save(model, model_path)
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    test_acc = evaluaion(test_loader)
    print('*' * 89)
    print('finish all, test acc: {}'.format(test_acc))
    
    torch.save(model, model_path+'_last')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, default= 'data/nsmc/'#'data/korean-hate-speech/' #'data/nsmc/'
    )
    parser.add_argument(
        '--bert_type', type=str, default= 'monologg/kobert'# 'monologg/kobert' #'beomi/kcbert-base'
    )
    parser.add_argument(
        '--labels', type=int, default=2
    )


    args = parser.parse_args()

    data_path = args.data_path
    bert_type = args.bert_type
    labels = args.labels
    EPOCHS = 15

    if 'kobert' in bert_type:
        from tokenization_kobert import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(bert_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
    
    model = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=labels)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
    # model = model.cuda()
    
    # orig_train = load_data(data_path,"labeled/train.tsv")
    # orig_test = load_data(data_path,"labeled/dev.tsv")

    #nsmc
    orig_train = load_data(data_path,"ratings_train.txt")
    orig_test = load_data(data_path,"ratings_test.txt")

    pack_util = packDataset_util(bert_type)
    train_loader = pack_util.get_loader(orig_train, shuffle=True, batch_size=256)
    test_loader = pack_util.get_loader(orig_test, shuffle=False, batch_size=256)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0 * len(train_loader), num_training_steps=EPOCHS * len(train_loader))

    train()

