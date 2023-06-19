from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
import numpy as np
import string
import re
from collections import Counter
import pickle
import pandas as pd

# Tree_index = 0
terminal_mark = 0

def get_content(address):
    with open(address,'r',encoding='utf-8') as f:
        contents = f.readlines()
        f.close()

    return contents

def scan_sourceTree(list_,tree,Tree_outline,p_s_list):
    if Tree_outline:
        for x in tree:
            scan_sourceTree(list_, x, False,p_s_list)
            return
    dict_ = {}
    # print(type(tree))
    global  terminal_mark
    if str(tree) == tree:
        terminal_mark += 1
        # print(type(tree),tree)
        p_s_list.append(len(list_))
        dict_['mark'] = 'TERMINAL'
        dict_['label'] = tree
        dict_['t_m'] = terminal_mark
        list_.append(dict_)
    else:
        p_s_list.append(len(list_))
        label_ = tree.label()
        dict_['label'] = label_
        dict_['mark'] = 'NOT_TERMINAL'
        dict_['s_list'] = []
        list_.append(dict_)
        for x in tree:
            scan_sourceTree(list_,x,False,dict_['s_list'])

def scan_listTree(node,listTree):
    list_ = []
    if node['mark'] == 'TERMINAL':
        # print(node['t_m'])
        return [node['t_m']]
    else:
        # print("node:",node)
        for ele in node['s_list']:
            # print(listTree[ele])
            temp_l = scan_listTree(listTree[ele],listTree)
            list_.extend(temp_l)
        node['terminals_id'] = list_
        return list_

def check_Tree(listTree):
    scan_listTree(listTree[0],listTree)

cut_type = '_jiebaAndLTP'
kinds = 'sampleSentence'
# ch_address = './Alignment/test.src'
# en_address = './Alignment/test.tgt'
ch_address = './alignment/'+kinds+'/'+'test_split'+cut_type+'.src'
en_address = './alignment/'+kinds+'/'+'test.tgt'

ch_tran = get_content(ch_address)
en_tran = get_content(en_address)

index = 0
all_list_ = []
while index < len(en_tran)/3:
    index_1 = index
    index_2 = int(index_1 + len(en_tran)/3)
    index_3 = int(index_1 + (len(en_tran)/3)*2)
    # if en_tran[index_1] != en_tran[index_2] or en_tran[index_1] != en_tran[index_3] or en_tran[index_2] != en_tran[index_3]:
    #     print(index,en_tran)
    dict_ = {}
    dict_['origin'] = en_tran[index_1]
    dict_['baidu'] = ch_tran[index_1]
    dict_['bing'] = ch_tran[index_2]
    dict_['google'] = ch_tran[index_3]
    all_list_.append(dict_)
    index += 1
# print(all_list_)

# initialize a constituency parser
eng_parser = CoreNLPParser('http://localhost:9000')
for ele in all_list_:
    tree_list = []
    source_tree = eng_parser.raw_parse(ele['origin'])
    # ele['tree'] = source_tree
    temp_list = []
    Tree_outline = True
    # for x in source_tree:
    # global terminal_mark
    terminal_mark = 0
    scan_sourceTree(temp_list,source_tree,Tree_outline,[])
    # print(temp_list)
    ele['tree'] = temp_list

with open("./alignment/"+kinds+"/pkl_"+kinds+"_zhen"+cut_type+".dat", 'wb') as f:
    pickle.dump(all_list_, f)

# for ele in all_list_:
#     list_Tree = ele['tree']
#     temp_list = []
#     check_Tree(list_Tree)
#     # print('x')
#
# with open("./alignment/pkl_delta_200_zhen.dat", 'wb') as f:
#     pickle.dump(all_list_, f)

