"""
Edit data columns into other string .
"""

from copy import copy
import os
from typing import Dict, List, Optional
import numpy as np
import string
import json

from tqdm import tqdm

from DataAugment.EditAbstract import Edition

class RDEdition(Edition):
    """
    Use random string to replace column
    """
    def __init__(self, prob=0.3) -> None:
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
    
    def check_replacable(self, s:str) -> bool:
        return s.isalnum() and not s.isnumeric()

    def generate_rd_string(self, length:Optional[int]=None) -> string:
        if length is None:
            length = np.random.randint(5, 11)
        rand_char_ids = np.random.randint(0, 26, (length, ))
        rand_chas = map(lambda x : chr(ord('a') + x), rand_char_ids)
        return ''.join(list(rand_chas))
    
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
        if len(tokens2remove) > 0:
            return sample
        # remove those tokens
        tokens = tokens.difference(tokens2remove)

        # consider topic tokens
        if edit_topic:
            tokens = tokens.union([token for token in topic.split(' ') if self.check_replacable(token)])
        
        # replace
        tokens = list(tokens)
        drop = np.random.rand(len(tokens))
        tokens = [token for idx, token in enumerate(tokens) if drop[idx] < self.prob]
        rep_map = {
            token: self.generate_rd_string() for token in tokens
        }
        
        # edit topic
        # if edit_topic:
        turb_topic = ' '.join([
            rep_map.get(token, token) for token in topic.split(' ')
        ])
        # else:
        #     turb_topic = topic

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
        
        # if turb_sent == sample['sent']:
        #     return None
        # if 'attendance' in turb_sent:
        #     print(turb_sent)
        sample['sent'] = turb_sent
        sample['topic'] = turb_topic
        sample['table_header'] = turb_header
        sample['logic_str'] = turb_logic

        return sample



'''
def check_replacable(s:str) -> bool:
    return s.isalnum() and not s.isnumeric()

def generate_rd_string(length:Optional[int]=None) -> string:
    if length is None:
        length = np.random.randint(5, 11)
    rand_char_ids = np.random.randint(0, 26, (length, ))
    rand_chas = map(lambda x : chr(ord('a') + x), rand_char_ids)
    return ''.join(list(rand_chas))

def edit_sample(org_sample: Dict, rp_prob : float, edit_topic: bool = True) -> Optional[Dict]:
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
    tokens = set([token for token in header_tokens if check_replacable(token)])

    # the column token that appear in logic form but not in text should not be replaced

    # column tokens appeared in logic form
    header_form_tokens = set(logic.split(' ')).intersection(header_tokens)
    # but not in text
    tokens2remove = header_form_tokens.difference(text.split(' '))
    if len(tokens2remove) > 0:
        return None
    # remove those tokens
    tokens = tokens.difference(tokens2remove)

    # consider topic tokens
    if edit_topic:
        tokens = tokens.union([token for token in topic.split(' ') if check_replacable(token)])
    
    # replace
    tokens = list(tokens)
    drop = np.random.rand(len(tokens))
    tokens = [token for idx, token in enumerate(tokens) if drop[idx] < rp_prob]
    rep_map = {
        token: generate_rd_string() for token in tokens
    }
    
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
    
    if turb_sent == sample['sent']:
        return None
    
    sample['sent'] = turb_sent
    sample['topic'] = turb_topic
    sample['table_header'] = turb_header
    sample['logic_str'] = turb_logic

    return sample


import argparse

if __name__ == '__main__':
    # s = generate_rd_string()
    # print(s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str)
    parser.add_argument('--prob', type=float, default=0.7)
    args = parser.parse_args()
    train : List = json.load(open(os.path.join(args.data_folder, 'train.json'), 'r'))
    print(len(train))
    turbed_train = []
    for sa in tqdm(train):
        d = edit_sample(sa, args.prob)
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