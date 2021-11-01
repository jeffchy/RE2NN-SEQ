import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_project_root)

import pandas as pd
import re
from typing import List, Optional, Tuple
from pydash import flow
from pydash.arrays import uniq, without, compact, flatten
from automata_tools import get_word_to_index
from src_seq.tools.reader import MIT_BIO_READER, CONLL03_BIO_READER, SNIPS_BIO_READER
import pickle


ATIS_BIO_PATH = os.path.join(_project_root, 'data', 'ATIS-BIO')
ATIS_ZH_BIO_PATH = os.path.join(_project_root, 'data', 'ATIS-ZH-BIO')
SNIPS_PATH = os.path.join(_project_root, 'data', 'SNIPS-BIO')


def load_rule(filePath: str):
    """
    Load rule in pd.DataFrame that use Tag name as index, each column contains a rule or None

    ### Example
    location                          None                                  None
    Cuisine                           None                                  None
    Price                             None                     (([0-9]*)|vmost)*
    Rating                            None                     (open|closew* ){0
    Hours            ( night| dinner| l...                                  None
    Amenity                           None                                  None
    Restaurant_Name                   None                                  None
    """
    ruleOfTags = dict()
    with open(filePath, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        currentTag: Optional[str] = None
        currentRules: List[str] = []
        for line in lines:
            # if this line is a tag name like "[Cuisine]"
            if re.match(r'^\[[a-z_A-Z_\+]+\]$', line):
                tagName = line[1:-1]
                # if we are going to load a new set of rules of a tag
                if tagName != currentTag:
                    # if we were parsing previous set of rules, store them into dict
                    if currentTag != None:
                        ruleOfTags[currentTag] = compact(currentRules)
                        currentRules = []
                    currentTag = tagName
            else:  # parsing line that contains a rule and a example split by //
                rule = line.split('//')[0]
                currentRules.append(rule.strip())
    # add rules of last tag
    if currentTag != None:
        ruleOfTags[currentTag] = compact(currentRules)

    return pd.DataFrame.from_dict(ruleOfTags, orient='index')

def load_rule_slot(filePath: str):
    """
    Load rule in pd.DataFrame that use Tag name as index, each column contains a rule or None

    ### Example
    location                          None                                  None
    Cuisine                           None                                  None
    Price                             None                     (([0-9]*)|vmost)*
    Rating                            None                     (open|closew* ){0
    Hours            ( night| dinner| l...                                  None
    Amenity                           None                                  None
    Restaurant_Name                   None                                  None
    """

    with open(filePath, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        currentRules = []
        for line in lines:
            rule = line.split('//')[0].strip()
            if rule:
                currentRules.append(rule)

    return currentRules


def load_SNIPS_dataset():

    with open(os.path.join(SNIPS_PATH, 'train.txt'), 'r', encoding='utf8') as f:
        texts_train, tags_train = SNIPS_BIO_READER(f)
    with open(os.path.join(SNIPS_PATH, 'dev.txt'), 'r', encoding='utf8') as f:
        texts_dev, tags_dev = SNIPS_BIO_READER(f)
    with open(os.path.join(SNIPS_PATH, 'test.txt'), 'r', encoding='utf8') as f:
        texts_test, tags_test = SNIPS_BIO_READER(f)

    df_train = pd.DataFrame(zip(texts_train, tags_train))
    df_train = df_train.sample(frac=1).reset_index(drop=True).rename(columns={0: 'text', 1: 'tags'})
    df_test = pd.DataFrame(zip(texts_test, tags_test))
    df_test = df_test.sample(frac=1).reset_index(drop=True).rename(columns={0: 'text', 1: 'tags'})
    df_dev = pd.DataFrame(zip(texts_dev, tags_dev))
    df_dev = df_dev.sample(frac=1).reset_index(drop=True).rename(columns={0: 'text', 1: 'tags'})

    df_train['mode'] = 'train'
    df_test['mode'] = 'test'
    df_dev['mode'] = 'dev'
    df = pd.concat([df_train, df_dev, df_test], ignore_index=True)

    return {
        'data': df,
    }


def load_BIO_dataset(dataset_name):
    import numpy as np


    if dataset_name == 'ATIS-ZH-BIO':
        train_fname = 'zh.atis.train.bio'
        test_fname = 'zh.atis.test.bio'
        dir_path = ATIS_ZH_BIO_PATH
    else:
        raise NotImplementedError()

    with open(os.path.join(dir_path, train_fname), 'r', encoding='utf8') as f:
        texts_train, tags_train = MIT_BIO_READER(f)
    with open(os.path.join(dir_path, test_fname), 'r', encoding='utf8') as f:
        texts_test, tags_test = MIT_BIO_READER(f)

    df = pd.DataFrame(zip(texts_train, tags_train))
    # Shuffle order
    df = df.sample(frac=1).reset_index(drop=True).rename(columns={0: 'text', 1: 'tags'})
    df_train, df_dev = np.split(df, [int(len(df)*0.8)], axis=0)


    df_test = pd.DataFrame(zip(texts_test, tags_test))
    df_test = df_test.sample(frac=1).reset_index(drop=True).rename(columns={0: 'text', 1: 'tags'})

    df_train['mode'] = 'train'
    df_test['mode'] = 'test'
    df_dev['mode'] = 'dev'
    df = pd.concat([df_train, df_dev, df_test], ignore_index=True)

    return {
        'data': df,
    }

def load_data_ATIS(mode, DATA_DIR = '../data/ATIS'):
    query, slots, intent = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format(mode)), 'rb'))
    return query, slots, intent


def load_dict_ATIS(DATA_DIR = '../data/ATIS'):
    t2i, s2i, in2i, i2t, i2s, i2in, dicts = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format('dicts')), 'rb'))
    i2s = {k:v.lower() for k, v in i2s.items()}
    s2i = {k.lower():v for k, v in s2i.items()}
    i2t = {k:v.lower() for k, v in i2t.items()}
    t2i = {k.lower():v for k, v in t2i.items()}
    return t2i, i2t, s2i, i2s, dicts


if __name__ == "__main__":
    pass