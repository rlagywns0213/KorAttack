import nltk
# nltk.download('stopwords')
import argparse
from ast import arg
import datasets
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from custom_dataset import packDataset_util
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from attack_util import *
from tqdm import tqdm
from nltk.corpus import stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import random
random.seed(714)
import os
import copy
base_path = os.path.abspath('.')

f = open("stopwords.txt", 'r')
filter_words = []
while True:
    line = f.readline()
    filter_words.append(line.split('\n')[0])
    if not line: break
f.close()
filter_words = set(filter_words) # 596개 https://www.ranks.nl/stopwords/korean

def load_data(file_path,data_type):
    if 'nsmc' in file_path:
        data = pd.read_csv(file_path+data_type, sep='\t')
        data = data.dropna() #결측치 제거 (5개 샘플)
        data.reset_index(drop=True, inplace=True)
    else:
        data = pd.read_csv("/home/rlagywns0213/22_hj/attack/koradv/data/korean-hate-speech/labeled/"+data_type, sep='\t')
        
    p_data = []
    for i in range(len(data)):
        if 'korean-hate-speech' in file_path:
            if data['contain_gender_bias'][i]:
                p_data.append((data['comments'][i], 1))
            else:
                p_data.append((data['comments'][i], 0))
        elif 'nsmc' in file_path:
            p_data.append((data['document'][i], data['label'][i]))       
    return p_data

def get_output_label(sentence,tokenizer,model):
    if pd.isna(sentence):
        return -1
    tokenized_sent = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    input_ids, attention_mask = tokenized_sent['input_ids'], tokenized_sent['attention_mask']
    # print(len(input_ids[0]))
    if torch.cuda.is_available():
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
    orig_probs = model(input_ids, attention_mask).logits.squeeze()
    return  orig_probs

def correct_print(asn,total):
    print("Attack Success")
    print('ASR: ',asn / total)


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words

def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length, args):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    seqs = seqs.cuda()
    eval_data = TensorDataset(seqs)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        bs = masked_input.size(0)
        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    return import_scores

def attack(model,orig_test,tokenizer,ATTACK_ITER,task_name, args=None):
    stopwords_list = stopwords.words('english')
    correct = 0
    correct_samples=[]
    # for sentence, label in tqdm(orig_test):
    #     orig_probs = get_output_label(sentence,tokenizer,model)
    #     orig_probs = torch.softmax(orig_probs, -1)
    #     orig_label = torch.argmax(orig_probs)

    #     if orig_label == label:
    #         correct += 1.
    #         # if label==1:
    #         if len(sentence.split(' ')) ==1:
    #             continue
    #         correct_samples.append((sentence,label))

    # print(correct_samples)
    # import pickle
    # with open('pickle_dataset/kcbert_hate.pickle', 'wb') as f:
    #     pickle.dump(correct_samples, f)
    
    # with open('pickle_dataset/kcbert_hate.pickle','rb') as f:
    #     correct_samples = pickle.load(f)
    
    # print('Orig Acc: ', correct / len(orig_test))

    
    # if len(correct_samples)>1000:
    #     random.seed(0)
    #     correct_samples = random.sample(correct_samples, 1000) #전체 데이터셋 중 1000개만 뽑음
    
    import pickle    
    if args.data_name == 'korean-hate-speech':
        with open('pickle_dataset/kcbert_hate_1000.pickle','rb') as f:
            correct_samples = pickle.load(f)
        
    elif args.data_name =='nsmc':
        with open('pickle_dataset/kcbert_nsmc_1000.pickle','rb') as f:
            correct_samples = pickle.load(f)
    
    # import pickle
    if (args.split_token == True) & (args.sub_char == True):
        action_list = [split_token, sub_char]
    elif args.split_token == True:
        action_list = [split_token]
    elif args.sub_char ==True:
        action_list =[sub_char]
    elif args.insert ==True:
        action_list =[insert_irrelevant]
    if (args.split_token == True) & (args.sub_char == True) & (args.insert == True):
        action_list =[split_token, sub_char, insert_irrelevant]
    
    if (args.split_token == True) & (args.sub_char == False) & (args.insert == True):
        action_list = [split_token, insert_irrelevant]
    
    if (args.split_token == False) & (args.sub_char == True) & (args.insert == True):
        action_list = [sub_char, insert_irrelevant]
        
    # action_list = [sub_char,insert_space, insert_irrelevant]#sub_char]#swap_char] #, sub_char]  # insert_space, insert_period
    asn = 0
    total=0
    total_attempt = 0
    success_list=[]
    querytime=0
    print(ATTACK_ITER)
    #한국어 nsmc
    sentence_index = 0
    one_word = 0
    max_iter = 5
    success_1_num = 0
    for sentence, label in tqdm(correct_samples):    
        querytime_each=0
        changed = 0
        sentence_index +=1
        
        total=total+1
        sentence = sentence.lower()
        orig_sentence=sentence
        import time
        start_time = time.time()
        orig_probs = get_output_label(sentence,tokenizer,model)
        orig_probs = torch.softmax(orig_probs, -1)
        orig_label = torch.argmax(orig_probs)
        current_prob = orig_probs.max()
        if orig_label == label:
            total_attempt += 1
            # Attack algorithm flow
            flag = False
            cache_pos_li = []
            batch_size = 32
            max_length = 300

            # while cur_iter < ATTACK_ITER:
            words_li = sentence.split(' ')
            final_words = copy.deepcopy(words_li)
            if len(words_li) ==1:
                # 한단어 우선 pass
                one_word+=1
                continue
            important_scores = get_important_scores(words_li, model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length, args)
            querytime  += int(len(words_li))
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            attempt_list = []

            for top_index in list_of_index:
                # if changed > int(0.4 * (len(words_li))):
                #     break  # exceed
                tgt_word = final_words[top_index[0]]
                candidate = tgt_word
                best_gap = 0.0
                cur_iter = 0
                changed += 1
                while cur_iter < ATTACK_ITER:

                    if tgt_word in filter_words:
                        candidate = tgt_word
                        break

                    action = random.choice(action_list) # 1개를 선택
                    new_word = action(tgt_word)
                    
                    # if new_word is None and action == sub_char:
                    #     cache_pos_li.remove(word_pos)
                    #     continue
                    if new_word is None:
                        cur_iter+=1
                        continue

                    final_words[top_index[0]] = new_word
                    # words_li = words_li[:word_pos[i]] + [new_word] + words_li[word_pos[i]+1:]

                    sentence = ' '.join(final_words)
                    
                    attempt_list.append((tgt_word, new_word))
                    if len(sentence) >300: #길이가 300자 이상 넘어가서 inference 불가한 경우
                        break
                    new_probs = get_output_label(sentence,tokenizer,model)
                    new_probs = torch.softmax(new_probs, -1)
                    new_label = torch.argmax(new_probs)
                    querytime = querytime+1
                    querytime_each=querytime_each+1
                    if new_label != label: #action으로 attack에 성공하는 경우
                        asn+=1
                        # print("----")
                        correct_print(asn,total_attempt)
                        success_list.append([4, sentence_index, attempt_list, orig_sentence,int(orig_label), sentence, int(new_label), querytime_each,changed, round(time.time() - start_time,2)])
                        # success_list.append([orig_sentence,sentence,querytime_each])
                        flag=True
                        
                        break
                    else: #action으로 Attack에 실패하는 경우=> 가장 많이 떨어뜨리는 단어로 저장
                        gap = orig_probs[orig_label] - new_probs[orig_label]
                        if gap > best_gap:
                            best_gap = gap
                            candidate = new_word
                        cur_iter+=1

                if flag==True:
                    break
                else: #5번해도 실패한 경우 candidate로 단어 교체
                    final_words[top_index[0]] = candidate
        # rec_times=cur+rec_times
        else:
            # rec_times=cur+rec_times
            continue
        
    print("query_time:", querytime)
    print("attack num: ", asn)
    print("total_attempt:",total_attempt)
    print("one_word:", one_word)
    name = ['success','sentence_index','record_attempt','origin','orig_label','attack','new_label','querytime','changed','time']
    save_data = pd.DataFrame(columns=name,data=success_list)
    save_data.to_csv(base_path+'/output/nsmc_subchar'+str(args.sub_char)+'_splittoken'+str(args.split_token)+'_insertirrelevant'+str(args.insert)+'max_iter'+str(ATTACK_ITER)+'.csv')
    print('ASR: ',asn / total_attempt)

def pipe(dataset_name,attack_iter, args=None):

    bert_type = "beomi/kcbert-base"
    # bert_type = "monologg/kobert"
     
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    if args.data_name == 'korean-hate-speech':
        data_path = 'data/korean-hate-speech/labeled/'
        orig_test = load_data(dataset_name,"dev.tsv")
        model = torch.load('/home/rlagywns0213/22_hj/attack/koradv/model_2023/kcbert_hate_speech_94.90%')
        
    elif args.data_name =='nsmc':
        data_path = 'data/nsmc/'
        orig_test = load_data(data_path,"ratings_test.txt")
        model = torch.load('/home/rlagywns0213/22_hj/attack/koradv/model_2023/kcbert_nsmc_89.93%')
        
    model_path = "model_2023/"+dataset_name
    # model = AutoModelForSequenceClassification.from_pretrained('/home/rlagywns0213/22_hj/attack/koradv/model_2023/nsmc')
    # model = torch.load('model_2023/kobert_nsmc')
    
    if torch.cuda.is_available():
        model = model.cuda()
        # model = nn.DataParallel(model.cuda())

    ATTACK_ITER = attack_iter
    attack(model,orig_test,tokenizer,ATTACK_ITER,dataset_name, args)
    print("-"*10+dataset_name+"-"*10)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='nsmc', type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--iter', default=15, type=int)
    parser.add_argument('--split_token',default=False, action='store_true')
    parser.add_argument('--sub_char', default=False,action='store_true')
    parser.add_argument('--insert', default=False,action='store_true')
    args = parser.parse_args()
    pipe(args.data_name,args.iter, args)

