import logging
import os
import json
from typing import List
import importlib
import re
from collections import deque

import numpy as np
import torch

from utils.encoder import Encoder

from DataAugment.EditAbstract import Edition
from DataAugment.normEdit import normEdition
from DataAugment.randEdit import RDEdition
from DataAugment.expandEdit import expandEdition
from DataAugment.MixedEdit import MixedEdition
from DataAugment.dtypeEdit import dtypeEdition


def linear_table_in(table):
    '''
    get processed linear table for gpt
    '''
    res = ""
    for ind, row in enumerate(table):
        res += (" row " + str(ind) + " : ")
        res += " ; ".join(row)

    return res.strip()


class OrgDataMng(object):
    """
    Set of original data .
    """

    def __init__(self, data_path) -> None:
        # skip into original data
        if 'original_data' in os.listdir(data_path):
            data_path = os.path.join(data_path, 'original_data')

        self.train = json.load(
            open(os.path.join(data_path, 'train.json'), 'r'))
        self.valid = json.load(
            open(os.path.join(data_path, 'valid.json'), 'r'))
        self.test = json.load(open(os.path.join(data_path, 'test.json'), 'r'))


class DataLoader:
    def __init__(self, data, encoder, batch_size, for_train, man_text_len=50, man_input_len=300, man_table_len=200, eos=50256, empty=28920, all_task=True, edit_strategy: str = None, edit_prob: float=None) -> None:
        self.data: List = data
        self.for_train = for_train
        self.all_task = all_task

        if self.for_train:
            # EDIT_class = importlib.import_module(edit_strategy)

            # # strip data if needed
            # self.editor: Edition = EDIT_class()
            if edit_strategy == 'rep':
                self.editor = normEdition(prob=edit_prob)
            elif edit_strategy == 'rand':
                self.editor = RDEdition(prob=edit_prob)
            elif edit_strategy == 'expand':
                self.editor = expandEdition(prob=edit_prob)
            elif edit_strategy == 'mix':
                self.editor = MixedEdition(prob=edit_prob)
            elif edit_strategy == 'dtype':
                self.editor = dtypeEdition(prob=edit_prob)
            else:
                logging.warn('edit strategy %s not understood' % edit_strategy)
                self.editor = Edition()
            assert isinstance(self.editor, Edition)
            self.data = self.editor.load(self.data)
            print(self.editor.edit(self.data[0]))

        # self.edit_strategy = edit_strategy


        # for training mode , settings

        # self.data = self.arg_data

        self.encoder: Encoder = encoder
        self.batch_size = batch_size

        self.man_text_len = man_text_len
        self.man_input_len = man_input_len
        self.man_table_len = man_table_len
        self.eos = eos
        self.empty = empty

        self.shuffle = for_train

        self.data_size = len(self.data)
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1

        self.count = 0
        
        # how many groups to select for classification loss

        if self.shuffle:
            self.shuffle_all_data()

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.num_batches:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        if self.shuffle:
            self.shuffle_all_data()

    def shuffle_all_data(self):
        """
        Shuffle all data
        Returns:
            None
        """
        data_size = len(self.data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.data = np.array(self.data)[shuffle_indices].tolist()
        return

    def get_zipped_batch(self, data):
        """
        Get zipped batch data of given start and end index
        """
        return zip(
            data['text'],
            data['topic'],
            data['logic'],
            data['header'],
            data['table']
        )

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1

        # generate batch data
        batch_in = self.data[start_index: end_index]

        if self.for_train:
            batch_in = self.editor.group_edit(batch_in)

        # if self.for_train:
        #     batch_in = self.editor.group_edit(batch_in)

        # group by category
        cats_list = {
            'topic': [self.encoder.encode(d['topic'])[0] for d in batch_in],
            'text': [self.encoder.encode(d['sent'])[0] for d in batch_in],
            'logic': [self.encoder.encode(d['logic_str'])[0] for d in batch_in],
            'header': [self.encoder.encode(' ; '.join(d['table_header']))[0] for d in batch_in],
            'table': [self.encoder.encode(linear_table_in(d['table_cont']))[0] for d in batch_in],
        }

        if self.all_task:
            # prepare input for level task
            level_task_in = []
            level_task_out = []

            chain_task_in, chain_task_out = [], []

            for d in batch_in:
                logic = d['logic_str'][:-6]
                # seperate '= true'
                tokens: list = [t for t in re.split(r'{|}|;| ', logic) if t != '']
                tok = np.random.choice(tokens)
                raw_tokens: list = logic.split(' ')
                inx, outy = "analyse token ' %s ': " % tok, None
                pos = raw_tokens.index(tok)
                raw_token_len = len(raw_tokens)
                # type
                if raw_tokens[pos + 1] == '{':
                    # operator
                    domain_t = []
                    i = pos + 2
                    level = 0
                    for i in range(pos+2, raw_token_len):
                        if raw_tokens[i] == '{':
                            level += 1
                        elif raw_tokens[i] == '}':
                            level -= 1
                            if level < 0:
                                break
                        elif level == 0 and raw_tokens[i] != ';':
                            domain_t.append(raw_tokens[i])
                    outy = "token ' %s ' is operator, occupies: %s" % (tok, ', '.join(domain_t))
                else:
                    # common value
                    pre_oper = None
                    i = 0
                    stack = deque()
                    for i in range(raw_token_len):
                        if raw_tokens[i] == '{':
                            stack.append(i)
                        elif raw_tokens[i] == '}':
                            stack.pop()
                        elif i == pos:
                            pre_oper = stack.pop()
                            break
                    outy = "token ' %s ' is argument, occupied by: %s" % (tok, raw_tokens[pre_oper - 1])
                level_task_in.append(inx)
                level_task_out.append(outy)

                # for chain path interpret
                tok = np.random.choice(tokens, 2)
                tok_a, tok_b = tok[0], tok[1]
                pos_a = raw_tokens.index(tok_a)
                pos_b = raw_tokens.index(tok_b)
                path_a, path_b = [tok_a], [tok_b]
                inx = "find path from ' %s ' to ' %s ': " % (tok_a, tok_b)
                pre_oper = deque()
                pre_oper.append(-1)
                pa = []
                levels, lev = [], 0
                for i in range(raw_token_len):
                    if raw_tokens[i] == '{':
                        pre_oper.append(i - 1)
                        lev += 1
                    elif raw_tokens[i] == '}':
                        pre_oper.pop()
                        lev -= 1
                    fa = pre_oper.pop()
                    pa.append(fa)
                    pre_oper.append(fa)
                    levels.append(lev)
                while pos_a != pos_b:
                    if levels[pos_a] > levels[pos_b]:
                        pos_a = pa[pos_a]
                        path_a.append(raw_tokens[pos_a])
                    else:
                        pos_b = pa[pos_b]
                        path_b.append(raw_tokens[pos_b])
                path_b.pop()
                outy = ' - '.join(path_a + list(reversed(path_b)))
                chain_task_in.append(inx)
                chain_task_out.append(outy)
                
            max_level_in_len = max([len(self.encoder.encode(sam)[0]) for sam in level_task_in])
            max_level_out_len = max([len(self.encoder.encode(sam)[0]) for sam in level_task_out])

            max_chain_in_len = max([len(self.encoder.encode(sam)[0]) for sam in chain_task_in])
            max_chain_out_len = max([len(self.encoder.encode(sam)[0]) for sam in chain_task_out])

        max_topic_len = max([
            len(sample) for sample in cats_list['topic']
        ])
        max_text_len = max([
            len(sample) for sample in cats_list['text']
        ])

        max_logic_len = max([
            len(sample) for sample in cats_list['logic']
        ])

        # max_header_len = max([
        #     len(sample) for sample in cats_list['header']
        # ])
        # max_table_len = max([
        #     len(sample) for sample in cats_list['table']
        # ])
        
        max_input_len = min(max_topic_len + max_logic_len + 10, self.man_input_len)
        if self.all_task:
            max_level_in_len += max_logic_len + 10
            max_chain_in_len += max_logic_len + 10


        batch_data = {
            'enc_in': [],
            'enc_len': [],
            'dec_len': [],
            'dec_out': [],
            'gpt_context': [],
            'attention_mask': [],

            'level_enc_in': [],
            'level_dec_out': [],
            'level_dec_len': [],
            'level_attention_mask': [],
            
            'chain_enc_in': [],
            'chain_dec_out': [],
            'chain_dec_len': [],
            'chain_attention_mask': [],
        }

        description, _ = self.encoder.encode('summary logic: ')

        left_, _ = self.encoder.encode(' {')
        right_, _ = self.encoder.encode(' }')
        pad_, _ = self.encoder.encode(' ;')
        left_ = left_[0]
        right_ = right_[0]
        pad_ = pad_[0]

        for uuu, (text, topic, logic, header, table) in enumerate(self.get_zipped_batch(cats_list)):

            # target text
            text_len = len(text)
            gold = text + [self.eos] * (max_text_len - text_len + 1)
            text = text + [self.eos] * (max_text_len - text_len)

            if self.all_task:
                level_gold, _ = self.encoder.encode(level_task_out[uuu])
                level_dec_len = len(level_gold)
                level_gold = level_gold + [self.eos] * (max_level_out_len - len(level_gold) + 1)

                chain_gold, _ = self.encoder.encode(chain_task_out[uuu])
                chain_dec_len = len(chain_gold)
                chain_gold = chain_gold + [self.eos] * (max_chain_out_len - len(chain_gold) + 1)

            # OOM
            if max_text_len > self.man_text_len:
                gold = gold[:self.man_text_len + 1]
                text = text[:self.man_text_len]
                text_len = min(text_len, self.man_text_len)

            # inout
            period, _ = self.encoder.encode(' . ')
            per_len = len(period)

            # full
            input = description + topic + period[:]
            
            input += logic
            # not_pad += [1] * len(logic)
            if self.all_task:
                level_input, _ = self.encoder.encode(level_task_in[uuu])
                ll = len(level_input)
                level_input = level_input + period[:] + logic
                fill_len = max_level_in_len - len(level_input)
                level_input = [self.empty] * fill_len + level_input
                level_start_pos = fill_len + ll + per_len

                chain_input, _ = self.encoder.encode(chain_task_in[uuu])
                ll = len(chain_input)
                chain_input = chain_input + period[:] + logic
                fill_len = max_chain_in_len - len(chain_input)
                chain_input = [self.empty] * fill_len + chain_input
                chain_start_pos = fill_len + ll + per_len

            # input = input[:self.man_input_len]
            # not_pad = not_pad[:self.man_input_len]
            input_len = len(input)
            input = [self.empty] * (max_input_len - input_len) + input
            start_pos = max_input_len - input_len + len(description) + len(topic) + per_len

            # for summarization. ref to gpt2 paper
            gpt_context = " . description: "

            gpt_context, _ = self.encoder.encode(gpt_context)

            '''
            ######################
            calculate attention of logical form
            ######################
            '''
            
            # calculate token distance
            segs = [[]]

            left_count, left_num = [], 0
            right_count, right_num = [], 0
            # pad_count, pad_num = [], 0

            for i, t in enumerate(logic):
                if t == left_:
                    left_num += 1
                    segs.append([i])
                    segs.append([])
                elif t == right_:
                    right_num += 1
                    segs.append([i])
                    segs.append([])
                elif t == pad_:
                    # pad_num += 1
                    segs.append([i])
                    segs.append([])
                else:
                    segs[-1].append(i)
                left_count.append(left_num)
                right_count.append(right_num)
                # pad_count.append(pad_num)
            
            segs = [seg for seg in segs if len(seg) > 0]

            cur_seg_id = 0
            cur_pos = 0
            seg_len = len(segs)
            operator_id = set()
            while True:
                if cur_seg_id + 1 < seg_len:
                    if logic[segs[cur_seg_id + 1][0]] == left_ and logic[cur_pos] not in (left_, right_, pad_):
                        for t in segs[cur_seg_id]:
                            operator_id.add(t)
                else:
                    break
                cur_seg_id += 1
                cur_pos = segs[cur_seg_id][0]
            # now `operator_id` contains all id that belong to operator
            cur_pos = 0
            cur_seg_id = 0
            logic_len = len(logic)

            relation_d = np.ones((logic_len, logic_len))
            # default setting: all token has the closest relation ship
            while True:
                if cur_seg_id + 1 < seg_len:
                    # token cur_pos is a start of operator
                    if cur_pos in operator_id:
                        l, r = 0, 0
                        never = False
                        for j in range(cur_pos + len(segs[cur_seg_id]), logic_len):
                            if logic[j] not in (left_, right_, pad_):
                                # procedure
                                if never:
                                    # already out of bound of token cur_pos
                                    relation_d[cur_pos: cur_pos + len(segs[cur_seg_id]), j] = 0
                                    relation_d[j, cur_pos: cur_pos + len(segs[cur_seg_id])] = 0
                                    # break
                                else:
                                    # in the bound of cur_pos operator, continue depth equals to number of `{` minus number of `}`
                                    assert l > r
                                    relation_d[cur_pos: cur_pos + len(segs[cur_seg_id]), j] = 1 if l - r == 1 else 0
                                    relation_d[j, cur_pos: cur_pos + len(segs[cur_seg_id])] = 1 if l - r == 1 else 0
                            elif logic[j] == left_:
                                l += 1
                            elif logic[j] == right_:
                                r += 1
                                if r >= l:
                                    # out of bound
                                    never = True
                    else:
                        # not operator token, only work in token itself
                        for j in range(cur_pos + len(segs[cur_seg_id]), logic_len):
                            if logic[j] not in (left_, right_, pad_):
                                relation_d[cur_pos: cur_pos + len(segs[cur_seg_id]), j] = 0
                                relation_d[j, cur_pos: cur_pos + len(segs[cur_seg_id])] = 0
                        
                else:
                    break
                cur_pos = segs[cur_seg_id + 1][0]
                cur_seg_id += 1
            
            '''
            #################
            set attention to input tensors
            #################            
            '''
            all_relation = np.ones((len(input) + len(gpt_context), len(input) + len(gpt_context)))
            all_relation[start_pos: start_pos + logic_len, start_pos: start_pos + logic_len] = relation_d
            
            if self.all_task:
                level_attention_mask = np.ones((len(level_input) + len(gpt_context), len(level_input) + len(gpt_context)))
                level_attention_mask[level_start_pos: level_start_pos + logic_len, level_start_pos: level_start_pos + logic_len] = relation_d
                
                chain_attention_mask = np.ones((len(chain_input) + len(gpt_context), len(chain_input) + len(gpt_context)))
                chain_attention_mask[chain_start_pos: chain_start_pos + logic_len, chain_start_pos: chain_start_pos + logic_len] = relation_d


            batch_data['enc_in'].append(input)
            batch_data['enc_len'].append(input_len)
            # batch_data['dec_in'].append(text)  # summary
            batch_data['dec_len'].append(text_len)  # summary len
            batch_data['dec_out'].append(gold)  # padded summary
            batch_data['attention_mask'].append(all_relation)
            batch_data['gpt_context'].append(gpt_context)
            # batch_data['drelation'].append(all_relation)
            # batch_data['not_pad'].append(not_pad)
            if self.all_task:
                batch_data['level_enc_in'].append(level_input)
                batch_data['level_dec_out'].append(level_gold)
                batch_data['level_dec_len'].append(level_dec_len)
                batch_data['level_attention_mask'].append(level_attention_mask)

                batch_data['chain_enc_in'].append(chain_input)
                batch_data['chain_dec_out'].append(chain_gold)
                batch_data['chain_dec_len'].append(chain_dec_len)
                batch_data['chain_attention_mask'].append(chain_attention_mask)
            
            if self.for_train and self.count == 1:
                print(self.encoder.decode(input))
                print(self.encoder.decode(gold))
                
                print(self.encoder.decode(level_input))
                print(self.encoder.decode(level_gold))

                print(self.encoder.decode(chain_input))
                print(self.encoder.decode(chain_gold))

        
        for key in batch_data:
            # if key == 'drelation' or key == 'not_pad':
            #     batch_data[key] = torch.FloatTensor(batch_data[key]).cuda()
            # else:
            batch_data[key] = torch.LongTensor(batch_data[key]).cuda()
        return batch_data
