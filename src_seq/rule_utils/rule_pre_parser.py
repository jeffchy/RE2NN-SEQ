import time
import re
from typing import Set, Dict, Optional, List, cast, Tuple
from pydash.arrays import uniq, without, compact, flatten
import pandas as pd
import argparse

pattern_define_var = re.compile(r'@(\w+)@=(\(.+\))')
pattern_use_var = re.compile(r'@(\w+)<:>((\w|-|\.)+)@')


class ParsingError(Exception):
    pass

def remove_multiple_lines(lines: List[str]):
    new_lines = []
    a_line = ""
    for l in lines:
        res = l.split('\\')
        # sanity check and line merging
        if len(res) == 1: # no \
            a_line += l
            new_lines.append(a_line)
            a_line = ""
        elif len(res) == 2:
            if res[1] == '':
                a_line += res[0]
            else:
                raise ParsingError("should have no content after \\")
        else:
            raise ParsingError("should have at most 1 \\ per line")

    return new_lines


def scanning_variables_in_line(line):

    res = re.match(pattern_define_var, line)
    if res: # match
        var_name = res.group(1)
        entites = res.group(2)
        if  (('@' not in var_name) and
            ('@' not in entites)):
            return True, var_name, entites
        else:
            raise ParsingError("should only have 1 defining sentence per line (2 @s)")

    return False, None, None


def generate_replace_text(entities, label_name, scheme):
    content = entities[1:-1]
    if (('(' in content) or (')' in content)):
        raise ParsingError("should not have parathesis in the variable defining entities")
    else:
        entity_seq = content.split('|')
        words = []
        if scheme == 'BIO':
            if ('I-' not in label_name) and ('B-' not in label_name):
                for ent in entity_seq:
                    word_list = ent.strip().split()
                    if len(word_list) == 0:
                        raise ParsingError(" | should have content on left and right side, error entity: {}".format(content))
                    elif len(word_list) == 1:
                        word_text = word_list[0] + "<:>" + "B-" + label_name + ' '
                    else:
                        word_text = word_list[0] + "<:>B-" + label_name + ' '
                        for i in range(1, len(word_list)):
                            word_text += (word_list[i] + "<:>I-" + label_name + ' ')
                    words.append(word_text)
                replace_text = "( "+ "|".join(words) + ")"
            else:
                for ent in entity_seq:
                    word_list = ent.strip().split()
                    if len(word_list) == 0:
                        raise ParsingError(" | should have content on left and right side, error entity: {}".format(content))
                    else:
                        word_text = word_list[0] + "<:>" + label_name + ' '
                        for i in range(1, len(word_list)):
                            word_text += (word_list[i] + "<:>" + label_name + ' ')
                    words.append(word_text)
                replace_text = "( "+ "|".join(words) + ")"
        else:
            raise NotImplementedError()

        return replace_text



def replace_variables(lines, scheme='BIO'):
    assert scheme in ['BIO', 'BMES']
    var_set = {}
    replaced_lines = []
    for line in lines:
        have_var_defining, var_name, entities = scanning_variables_in_line(line)
        if have_var_defining: # sentence defining a variable
            var_set[var_name] = entities
        else:
            span_infos = []
            for res in re.finditer(pattern_use_var, line):
                var_name = res.group(1)
                label_name = res.group(2)

                if var_name not in var_set:
                    raise ParsingError("{} Variable Undefined".format(var_name))
                else:
                    entities = var_set[var_name]
                    replace_text = generate_replace_text(entities, label_name, scheme)

                    span_infos.append({
                        'span': res.span(),
                        'replace_text': replace_text
                    })

            if not span_infos:
                replaced_lines.append(line)
            else:
                left = 0
                new_line = ""
                for i in span_infos:
                    right = i['span'][0]
                    seg = line[left: right]
                    new_line += ( seg + i['replace_text'])
                    left = i['span'][1]
                new_line += line[left:]
                replaced_lines.append(new_line)

    return replaced_lines

def O_compolete(lines):
    from pyparsing import Word, alphanums, OneOrMore
    from pyparsing import pyparsing_unicode as PU
    specialWords = Word("()|?+*")
    normalWords = Word(alphanums + "'`_-.<>:{},$%&" + PU.Chinese.printables)
    leafWords = specialWords | normalWords

    new_lines = []
    for line in lines:

        words = OneOrMore(leafWords).parseString(line).asList()
        new_line = []
        for word in words:
            if ('<:>' not in word) and \
                not re.search(r'\*|\?|\{|\}|\+|\||\(|\)', word): # cancel out special variables
                new_word = word + '<:>O'
                new_line.append(new_word)
            else:
                new_line.append(word)

        new_line = ' '.join(new_line)
        new_lines.append(new_line)

    return new_lines





def rule_pre_parser(filePath: str):
    """
    :param filePath:
    :return: pre parsed rules
    Functions:
    1. remove comments //
    2. remove multiline operator \
    2. define dictionary variableS
        Grammar: @all_restaurants@=(KFC | berger king | bergerking) (one per one line)
            => (KFC<:>B-Restaurant_Name|berger<:>B-Restaurant_Name king<:>I-Restaurant_Name|bergerking<:>B-Restaurant_Name)
            BIO/BMES scheme
            in the rule: use @all_restaurants<:>Restaurant_Name@
    4. complete the O

    """
    print("Parsing File: {}".format(filePath))

    with open(filePath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

        print("REMOVE COMMENTS AND EMPTY LINE ...... ")
        removed_lines = []
        for line in lines:
            l = line.split('//')[0]
            l = l.strip()
            if l: removed_lines.append(l)

        print("REMOVE MULTILINE OPERATOR ...... ")
        merged_lines = remove_multiple_lines(removed_lines)

        print("REPLACE VARIABLES ...... ")
        replaced_lines = replace_variables(merged_lines)

        print("REMOVE MULTILINE OPERATOR ...... ")
        complete_lines = O_compolete(replaced_lines)


    with open(filePath+'.parsed', 'w', encoding='utf-8') as f:
        for line in complete_lines:
            f.write(line + '\n')
            print(line)

    return complete_lines

def load_slot_rule_from_file(rule_file: str):
    with open(rule_file, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        currentRules = []
        for line in lines:
            rule = line.split('//')[0].strip()
            if rule:
                currentRules.append(rule)

    return currentRules

def load_slot_rule_from_lines(rule_lines: List[str]):
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

    currentRules = []
    for line in rule_lines:
        rule = line.split('//')[0].strip()
        if rule:
            currentRules.append(rule)

    return currentRules

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule_path', type=str, help='rule path')
    args = parser.parse_args()
    rule_pre_parser(args.rule_path)