"""
Edit data columns into other string .
No edit topic
"""

from collections import defaultdict
from copy import copy
import os
from typing import Dict, Iterable, List, Optional
import numpy as np
import string
import json

from tqdm import tqdm

from DataAugment.EditAbstract import Edition

class dtypeEdition(Edition):
    def __init__(self, prob=1) -> None:
        super().__init__()
        self.prob = prob

    def load(self, data: List[Dict]) -> List[Dict]:
        print(len(data))

        # count all columns
        columns_set = set([])
        for sample in data:
            headers = sample['table_header']
            for column in headers:
                columns_set = columns_set.union([token for token in column.split(' ') if self.check_replacable(token)])
        
        # self.global_dictionary = list(columns_set)
        
        # """
        self.global_dictionary = {
            'time': [],
            'number': [],
            'str': []
        }
        self.token2type = {}
        with open('DataAugment/time_headers.txt', 'r') as f:
            self.global_dictionary['time'] = f.read().split('\n')
            for t in self.global_dictionary['time']:
                self.token2type[t] = 'time'
        with open('DataAugment/number_headers.txt', 'r') as f:
            self.global_dictionary['number'] = f.read().split('\n')
            for t in self.global_dictionary['number']:
                self.token2type[t] = 'number'
        with open('DataAugment/str_headers.txt', 'r') as f:
            self.global_dictionary['str'] = f.read().split('\n')
            for t in self.global_dictionary['str']:
                self.token2type[t] = 'str'
        
        self.rep_dictionary = {
            'number': self.global_dictionary['number'],
            'str': self.global_dictionary['str'],
            'time': None
        }
        with open('DataAugment/time_headers2rep.txt', 'r') as f:
            self.rep_dictionary['time'] = f.read().split('\n')
        # """
        
        sample_new = []
        for sample in data:
            topic : str = sample['topic']
            text : str = sample['sent']
            header : str = sample['table_header']
            logic : str = sample['logic_str']
            # tokens = set([token for token in logic.split(' ') if token.isalnum()])
            # flatten header
            header_tokens = []
            for header_str in header:
                header_tokens += header_str.split(' ')
            # the column token that appear in logic form but not in text should not be replaced

            # column tokens appeared in logic form
            header_form_tokens = set(logic.split(' ')).intersection(header_tokens)
            # but not in text
            tokens2remove = header_form_tokens.difference(text.split(' '))
            if len(tokens2remove) > 0:
                continue
            sample_new.append(sample)
        
        return sample_new

    def edit(self, org_sample: Dict, edit_topic: bool = False) -> Optional[Dict]:
        # extract atributes
        # sample contains 'topic', 'sent', 'table_header'
        sample = copy(org_sample)
        topic : str = sample['topic']
        text : str = sample['sent']
        header : str = sample['table_header']
        logic : str = sample['logic_str']
        # tokens = set([token for token in logic.split(' ') if token.isalnum()])
        # flatten header
        header_tokens = []
        for header_str in header:
            header_tokens += header_str.split(' ')
        tokens = set([token for token in header_tokens if self.check_replacable(token)])

        # the column token that appear in logic form but not in text should not be replaced

        # column tokens appeared in logic form
        header_form_tokens = set(logic.split(' ')).intersection(header_tokens)
        # but not in text
        tokens2remove = header_form_tokens.difference(text.split(' '))
        if len(tokens2remove) > 1:
            return sample
        # remove those tokens
        tokens = tokens.difference(tokens2remove)
        
        assert edit_topic is False
        # consider topic tokens
        # if edit_topic:
        #     tokens = tokens.union([token for token in topic.split(' ') if self.check_replacable(token)])
        
        # replace
        tokens = list(tokens)
        drop = np.random.rand(len(tokens))
        tokens = [token for idx, token in enumerate(tokens) if drop[idx] < self.prob]

        # rd_tokens = np.random.choice(self.global_dictionary, len(tokens))
        rep_map = {}
        for tok in tokens:
            dtype = self.token2type[tok]
            rep_map[tok] = np.random.choice(self.rep_dictionary[dtype])
            
            # else:
            # rep_map[tok] = np.random.choice(self.global_dictionary)
        
        # edit topic
        if edit_topic:
            turb_topic = ' '.join([
                rep_map.get(token, token) for token in topic.split(' ')
            ])
        else:
            turb_topic = topic

        # edit text
        turb_sent = ' '.join([
            rep_map.get(token, token) for token in text.split(' ')
        ])

        # edit logic
        turb_logic = ' '.join([
            rep_map.get(token, token) for token in logic.split(' ')
        ])

        # edit header
        turb_header = []
        for header_str in header:
            turb_column = ' '.join([
                rep_map.get(token, token) for token in header_str.split(' ')
            ])
            turb_header.append(turb_column)
        
        sample['sent'] = turb_sent
        sample['topic'] = turb_topic
        sample['table_header'] = turb_header
        sample['logic_str'] = turb_logic

        return sample

    def check_replacable(self, s:str) -> bool:
        return s.isalnum() and not s.isnumeric()


'''

import argparse

if __name__ == '__main__':
    # s = generate_rd_string()
    # print(s)

    np.random.seed(17)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str)
    parser.add_argument('--prob', type=float, default=0.7)
    args = parser.parse_args()
    train : List = json.load(open(os.path.join(args.data_folder, 'train.json'), 'r'))
    print(len(train))

    # count all columns
    columns_set = set([])
    for sample in train:
        headers = sample['table_header']
        for column in headers:
            columns_set = columns_set.union([token for token in column.split(' ') if check_replacable(token)])
    
    global_dictionary = np.array(list(columns_set))

    turbed_train = []
    for sa in tqdm(train):
        d = edit_sample(sa, global_dictionary, args.prob)
        if d is None:
            # turbed_train.append(sa)
            continue
        else:
            turbed_train.append(d)
    print(len(turbed_train))
    # turbed_train += train
    json.dump(train, open(os.path.join(args.data_folder, 'train_backup.json'), 'w'))
    json.dump(turbed_train, open(os.path.join(args.data_folder, 'train.json'), 'w'))

'''