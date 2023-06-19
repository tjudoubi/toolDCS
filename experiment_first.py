# from pyltp import Segmentor
import re
from nltk.translate.bleu_score import sentence_bleu
from bert_score import scoreRe
from bert_score import score
from simcse import SimCSE
from nltk.translate.chrf_score import sentence_chrf
import os
import numpy as np
import Chinese2vec
# from moverscore_v2 import get_idf_dict, word_mover_score
# from collections import defaultdict
import time

from nltk.corpus import wordnet
from SynonymsRep import SynonymsReplacer

from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from pyltp import Segmentor


#######################导入ChinesewordVector#####################
fv = Chinese2vec.fast2vec()
# word2vec_path_ = "D:\\BaiduNetdiskDownload\\sgns.renmin.bigram-char.bz2"
word2vec_path_ = "./sgns.baidubaike.bigram-char.bz2"
# word2vec_path_ = "D:\\BaiduNetdiskDownload\\merge_sgns_bigram_char300.txt.bz2"
fv.load_word2vec_format(word2vec_path=word2vec_path_)

########################加载哈工大LTP分词工具###########################################
LTP_DATA_DIR = './ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# 实例化
segmentor = Segmentor()
# 加载分词模型
segmentor.load(cws_model_path)
replacer = SynonymsReplacer(synonyms_file_path='./tools/HIT-IRLab-Synonyms.txt',
                                cws_model=segmentor)
######################################################################################


cut_type = "_jiebaAndLTP"
kinds = ""

# cut_type = ""
threshold = 0.75
date = "2022_8_4"

######################################导入simcse模型##########################
model_save_path = "./simcse-chinese-roberta-wwm-ext"
# model_save_path = "D:\\ARTICAL\\chinese_L-12_H-768_A-12"
model = SentenceTransformer(model_save_path)

metric_type = cut_type+"_"+date+"_threshold_"+str(threshold)+"_"+kinds
# model_simcse = SimCSE(model_save_path)
# model = SentenceTransformer()
############################################################################
stop_wordlist = [line.strip() for line in open('hit_stopword.txt',encoding='UTF-8').readlines()]

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·《》—"""

##############导入词对齐列表###################
import pandas
ch_align_list = pandas.read_pickle("./alignment/"+kinds+'/'+kinds+"_align_three_200_zhen"+cut_type)


##############导入句子压缩列表 建字典###################
with open('./sent_comp/'+kinds+'/'+'test.cln.strip.sent',encoding='utf-8') as f:
    sent_list_ = f.readlines()
# with open('./sent_comp/test_result_greedy.mydata - 副本.sents',encoding='utf-8') as f:
with open('./sent_comp/'+kinds+'/'+'slahan_wo_syn_0.sents', encoding='utf-8') as f:
    compression_list_ = f.readlines()
    for i in range(0,len(compression_list_)):
        compression_list_[i] = compression_list_[i].replace('<s>','').replace('</s>','').replace(' \n','').split()
        compression_list_[i] = [int(ele_) for ele_ in compression_list_[i]]

dict_comp = {}
ch_align_dict = {}

for i in range(0,len(compression_list_)):
    dict_comp[sent_list_[i]] = compression_list_[i]
    ch_align_dict[sent_list_[i]] = ch_align_list[i]
#########################################################

######################################导入源句和其对应的句法树及翻译字典####################################
import pandas
sent_tree_list_ = pandas.read_pickle('./alignment/'+kinds+'/'+'/pkl_'+kinds+'_zhen'+cut_type+'.dat')
print(sent_tree_list_)
###############################################################################################

###########################向英文句法树添加新的属性，其对应的是当前节点对应的该分支的叶子节点#############
def scan_Tree(tree_node,node_list,dict_sent_comp,ch_dict):
    if tree_node['mark'] == 'NOT_TERMINAL':
        s_len = len(tree_node['s_list'])
        new_s_list = []
        for node_ele in tree_node['s_list']:
            s_flag = scan_Tree(node_list[node_ele],node_list,dict_sent_comp,ch_dict)
            if s_flag == False:
                # print('s')
                # tree_node['s_list'].remove(node_ele)
                s_len -= 1
            else:
                new_s_list.append(node_ele)
        tree_node['s_list'] = new_s_list
        if s_len == 0:
            return False
        else:
            return True
    elif tree_node['mark'] == 'TERMINAL':

        id = tree_node['t_m']-1
        if dict_sent_comp[id] == 1:
            print(id)
            return False
        elif dict_sent_comp[id] == 0:
            return True
###############################################################################################
#####################将英文句法树叶子节点英文词与对应的中文index对应上##################
def clarity_travel(tree_node,node_list,ch_dict):
    # global travel_string, baidu_adjunct_set, bing_adjunct_set, google_adjunct_set
    if tree_node['mark'] == 'NOT_TERMINAL':
        for node_ele in tree_node['s_list']:
            clarity_travel(node_list[node_ele],node_list,ch_dict)
    elif tree_node['mark'] == 'TERMINAL':
        # travel_string = travel_string + tree_node['label'] + ' '
        tree_node['baidu_id'] = ch_dict['baidu'][tree_node['t_m'] - 1]
        tree_node['bing_id'] = ch_dict['bing'][tree_node['t_m'] - 1]
        tree_node['google_id'] = ch_dict['google'][tree_node['t_m'] - 1]


        print(tree_node['label'])
####################################################################################


# def travel(tree_node,node_list,ch_dict):
#     global travel_string, baidu_adjunct_set, bing_adjunct_set, google_adjunct_set
#     if tree_node['mark'] == 'NOT_TERMINAL':
#         for node_ele in tree_node['s_list']:
#             travel(node_list[node_ele],node_list,ch_dict)
#     elif tree_node['mark'] == 'TERMINAL':
#         print(tree_node['label'])
#
# def getMainString(sent,comp_list,ch_dict):
#     global main_string, baidu_main_set, bing_main_set, google_main_set
#     list_ = sent.split()
#     for i in range(0,len(comp_list)):
#         if comp_list[i] == 1:
#             main_string = main_string + list_[i] + ' '
#             for ele in ch_dict['baidu'][i]:
#                 baidu_main_set.add(ele)
#             for ele in ch_dict['bing'][i]:
#                 bing_main_set.add(ele)
#             for ele in ch_dict['google'][i]:
#                 google_main_set.add(ele)



for sent_ele in sent_list_:
    for tree_ele in sent_tree_list_:
        if sent_ele == tree_ele['origin']:
            clarity_travel(tree_ele['tree'][0],tree_ele['tree'],ch_align_dict[sent_ele])
            print(sent_ele)
            scan_Tree(tree_ele['tree'][0],tree_ele['tree'],dict_comp[sent_ele],ch_align_dict[sent_ele])
print("x")
# #########################################################################################
# ##########################align-repair###################################################
# for tree_ele in sent_tree_list_:
#     def get_nodeTran(id_list, tran_list):
#         temp = ""
#         for ele in id_list:
#             temp = temp + tran_list[ele] + " "
#         return temp[:-1]
#     def get_exit(stringA,stringB):
#         AtempB = ""
#         BtempA = ""
#         listA = stringA.split()
#         listB = stringB.split()
#         noSplitB = stringB.replace(' ','')
#         noSplitA = stringA.replace(' ','')
#         mood = ['的','了','的','了','么','呢','吧','啊']
#         for ele in listA:
#             if ele not in noSplitB:
#                 AtempB = AtempB + ele
#         public_flag = False
#         for ele in listA:
#             if ele in noSplitB:
#                 public_flag = True
#         if public_flag == False and stringB != "":
#             AtempB = ""
#
#         for ele in listB:
#             if ele not in noSplitA:
#                 BtempA = BtempA + ele
#         public_flag = False
#         for ele in listB:
#             if ele in noSplitA:
#                 public_flag = True
#         if public_flag == False and stringA != "":
#             BtempA = ""
#         return AtempB,BtempA
#
#     def repair_align(node_align,sub_a,sub_b,tranlist):
#         # if len(node_align) > 0:
#         set_temp = set()
#         for i in range(0,len(tranlist)):
#             if tranlist[i] in sub_a:
#                 set_temp.add(i)
#             if tranlist[i] in sub_b:
#                 set_temp.add(i)
#         for ele in set_temp:
#             if ele not in node_align:
#                 node_align.append(ele)
#         node_align.sort()
#         # return
#
#     for node in tree_ele['tree']:
#         baidu_tran = tree_ele['baidu'].replace('\n', '').split()
#         google_tran = tree_ele['google'].replace('\n', '').split()
#         bing_tran = tree_ele['bing'].replace('\n', '').split()
#         if node['mark'] == 'TERMINAL' and (node['label'] not in punctuation):
#             baidu_id_list = node['baidu_id']
#             google_id_list = node['google_id']
#             bing_id_list = node['bing_id']
#             baidu_node = get_nodeTran(baidu_id_list,baidu_tran)
#             google_node = get_nodeTran(google_id_list, google_tran)
#             bing_node = get_nodeTran(bing_id_list, bing_tran)
#             # print(node['label'],": ",baidu_tran,google_tran,bing_tran)
#             # if baidu_tran != google_tran:
#             bai_f_goo,goo_f_bai = get_exit(baidu_node,google_node)
#             bai_f_bin,bin_f_bai = get_exit(baidu_node,bing_node)
#             bin_f_goo,goo_f_bin = get_exit(bing_node,google_node)
#             repair_align(node['baidu_id'], goo_f_bai, bin_f_bai, baidu_tran)
#             repair_align(node['google_id'], bai_f_goo, bin_f_goo, google_tran)
#             repair_align(node['bing_id'], goo_f_bin, bai_f_bin, bing_tran)
# ###########################align-repair####################################################
# ###########################################################################################

def cut_adjunct_other_Clauses(tree_node,first_list,node_list,Sflag,flag,adjunct_temp_lists):
    # global adjunct_temp_lists
    if tree_node['mark'] == 'NOT_TERMINAL':
        # if (tree_node['label'] == 'S' or tree_node['label'] == 'SBAR' or tree_node['label'] == 'SBARQ' or tree_node['label'] == 'SINV' or tree_node['label'] == 'SQ' or tree_node['label'] == 'PP') and Sflag != 1 and flag == 0:
        if (tree_node['label'] == 'SBAR' or tree_node['label'] == 'SBARQ' or tree_node[
            'label'] == 'SINV' or tree_node['label'] == 'SQ' ):
            second_list = []
            flag = 1
            for node_ele in tree_node['s_list']:
                cut_adjunct_other_Clauses(node_list[node_ele],second_list,node_list,0,flag,adjunct_temp_lists)
            adjunct_temp_lists.append(second_list)
        else:
            for node_ele in tree_node['s_list']:
                cut_adjunct_other_Clauses(node_list[node_ele], first_list, node_list,0,flag,adjunct_temp_lists)
    elif tree_node['mark'] == 'TERMINAL':
        first_list.append(tree_node)
        print(tree_node['label'])

def cut_adjunct_S_Clauses(tree_node,first_list,node_list,Sflag,flag,adjunct_temp_lists):
    # global adjunct_temp_lists
    if tree_node['mark'] == 'NOT_TERMINAL':
        # if (tree_node['label'] == 'S' or tree_node['label'] == 'SBAR' or tree_node['label'] == 'SBARQ' or tree_node['label'] == 'SINV' or tree_node['label'] == 'SQ' or tree_node['label'] == 'PP') and Sflag != 1 and flag == 0:
        if tree_node['label'] == 'S' :
            second_list = []
            flag = 1
            for node_ele in tree_node['s_list']:
                cut_adjunct_S_Clauses(node_list[node_ele],second_list,node_list,0,flag,adjunct_temp_lists)
            adjunct_temp_lists.append(second_list)
        else:
            for node_ele in tree_node['s_list']:
                cut_adjunct_S_Clauses(node_list[node_ele], first_list, node_list,0,flag,adjunct_temp_lists)
    elif tree_node['mark'] == 'TERMINAL':
        first_list.append(tree_node)
        print(tree_node['label'])

def get_cut_adjunct(tree_node,node_list,clause_flag):
    adjunct_temp_lists = []
    first_list = []
    if clause_flag == True:
        cut_adjunct_other_Clauses(tree_node,first_list, node_list, 1, 0,adjunct_temp_lists)
    else:
        cut_adjunct_S_Clauses(tree_node,first_list, node_list, 1, 0,adjunct_temp_lists)
    adjunct_temp_lists.append(first_list)
    return adjunct_temp_lists


def add_main_list(adjunct_list,comp_list,tree):

    def merge_list(list_l):
        temp_ = []
        if len(list_l) > 0:
            temp_.append(list_l[0])
        for i in range(1, len(list_l)):
            if len(list_l[i]) <= 0:
                # all_list_[len(list_)-1].append(temp_list_[i][0])
                temp_[len(temp_) - 1].extend(list_l[i])
            elif len(list_l[i]) != 0:
                temp_.append(list_l[i])
        return temp_

    # main_string = ""
    mainId_list = []
    for i in range(0, len(comp_list)):
        if comp_list[i] == 1:
            mainId_list.append(i)
    mainId_list.sort()
    main_list = []
    for id in mainId_list:
        for ele in tree:
            if ele['mark'] == 'TERMINAL' and ele['t_m'] == id+1 and (ele['label'] not in punctuation):
                main_list.append(ele)

    temp_list_ = []
    for list_single in adjunct_list:
        # temp_list_.append(list_single)
        temp = []
        for ele in list_single:
            if ele['label'] not in punctuation:
                temp.append(ele)
        temp_list_.append(temp)

    temp_list_ = merge_list(temp_list_)
    temp_list_.insert(0,main_list)
    all_list_ = merge_list(temp_list_)

    return all_list_

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

layer_label = 12
reg = r"[(a-z)|(A-Z)]+[0-9]*[a-zA-Z]*"
reg1 = r"[(a-z)|(A-Z)]+[0-9]*[a-zA-Z]*"
reg2 = r"[0-9]+[.]*[0-9]*"



def getString(list_):
    s1 = ""
    for ele in list_:
        s1 += ele
    dicts = {i: '' for i in punctuation}
    punc_table = str.maketrans(dicts)
    new_s = s1.translate(punc_table)

    new_s = re.sub(r'[（]([a-zA-Z0-9]+[\s]*[a-zA-Z0-9]*)*[）]', '', new_s)
    return new_s



def isEnglish(word):
    flag = False
    for ele in word:
        if (ele <= 'z' and ele >= 'a') or (ele <= 'Z' and ele >= 'A'):
            flag = True
        else:
            return False
    return flag

def replace_ch(dict_ele,ch_id,tree,type_stringA,type_stringB,another_list):
    treeList = tree['tree']
    Btrans = tree[type_stringB].strip().split()
    for ele in treeList:
        if ele['mark']=='TERMINAL':
            if ch_id in ele[type_stringA+'_id']:
                ch_id_another_list = ele[type_stringB+'_id']
                for one_l in another_list:
                    if one_l['self_id'][0] in ch_id_another_list:
                        one_l['flag'] = True
                        one_l['other_id'] = ele[type_stringB+'_id']
                        one_l['other'] = dict_ele['self']
    return


def remove_punc(string):
    temp_str = ""
    mood = ['的', '了', '么', '呢', '吧', '啊']
    for ele in string:
        if ele not in punctuation :
            temp_str = temp_str + ele
    return temp_str

#########get_pair函数用来进行 英文NER（命名实体）对应的中文 进行英文的替换####################################################
def get_pair(Alist,Blist,type_stringA,type_stringB,tree):
    Atrans_list = tree[type_stringA].strip().split()
    Btrans_list = tree[type_stringB].strip().split()
    compare_listA = []
    compare_listB = []
    for i in range(0,len(Alist)):
        A_ch_two_list = []
        B_ch_two_list = []
        for ele in Alist[i]:
            dict_ = {'self':Atrans_list[ele],'other':Atrans_list[ele],'flag':False,'self_id':[ele],'other_id':[ele]}
            A_ch_two_list.append(dict_)
        for ele in Blist[i]:
            dict_ = {'self': Btrans_list[ele],'other':Btrans_list[ele], 'flag': False,'self_id':[ele],'other_id':[ele]}
            B_ch_two_list.append(dict_)
        # print(A_ch_two_list)
        for j in range(0,len(A_ch_two_list)):
            if isEnglish(A_ch_two_list[j]['self']):
                # A_ch_two_list[j]['flag'] = True
                replace_ch(A_ch_two_list[j],Alist[i][j],tree,type_stringA,type_stringB,B_ch_two_list)
                # replace_ch(A_ch_two_list[j], Alist[i][j], tree, type_stringA, type_stringB)
        for j in range(0, len(B_ch_two_list)):
            if isEnglish(B_ch_two_list[j]['self']):
                # print("B:",B_ch_two_list[j]['self'])
                # B_ch_two_list[j]['flag'] = True
                replace_ch(B_ch_two_list[j], Blist[i][j], tree, type_stringB, type_stringA,A_ch_two_list)
                # replace_ch(B_ch_two_list[j], Blist[i][j], tree, type_stringB, type_stringA)
        single_listA = []
        single_listB = []
        eng_listA = []
        eng_listB = []
        transA = ""
        transB = ""
        for ele in A_ch_two_list:
            if ele['flag'] == False and ele['self_id'] not in single_listA:
                transA = transA + ele['self']
                single_listA.append(ele['self_id'])
            elif ele['flag'] == True and ele['other_id'] not in eng_listA:
                transA = transA + ele['other']
                eng_listA.append(ele['other_id'])

        for ele in B_ch_two_list:
            if ele['flag'] == False and ele['self_id'] not in single_listB:
                transB = transB + ele['self']
                single_listB.append(ele['self_id'])
            elif ele['flag'] == True and ele['other_id'] not in eng_listB:
                transB = transB + ele['other']
                eng_listB.append(ele['other_id'])
        compare_listA.append(remove_punc(transA))
        compare_listB.append(remove_punc(transB))
    # print(compare_listB,compare_listA)
    return compare_listB,compare_listA
###################################################################################################

def normal(strings):
    new_s = ""
    for ele in strings:
        if ele != '的':
            new_s = new_s + ele
    return new_s

########################word alignment repair  可以暂时不看################################
def get_matchedId(tree_list,type_string):
    temp_set = set()
    for ele in tree_list:
        if ele['mark'] == 'TERMINAL':
            for id in ele[type_string+'_id']:
                temp_set.add(id)
    temp_list = list(temp_set)
    temp_list.sort()
    return temp_list

def cmp_func(eleA):
    length = len(eleA['trans'])
    if length > 0:
        if eleA['trans'][length-1] == eleA['end'] and length > 1:
            return eleA['trans'][length-2]-eleA['trans'][0]
        else:
            return eleA['trans'][length-1]-eleA['trans'][0]
    else:
        return length

def check_proper_A(dict_trans,id,end):
    index = 0
    while index <= end:
        if id > dict_trans['trans'][index] and (index + 1 <= end) and id < dict_trans['trans'][index + 1]:
            # delta = dict_trans['trans'][index + 1]-dict_trans['trans'][index]
            delta = min(id-dict_trans['trans'][index],dict_trans['trans'][index + 1]-id)
            return True, delta
        index += 1
    return False, 100000

def check_proper_B(dict_trans,id,end):
    if end >= 0 and id < dict_trans['trans'][0]:
        # delta = dict_trans['trans'][index + 1]-dict_trans['trans'][index]
        delta = dict_trans['trans'][0] - id
        return True, delta
    elif end >= 0 and id > dict_trans['trans'][end]:
        delta = id - dict_trans['trans'][end]
        return True, delta

    return False, 100000

def checkMissWord(tree,dict_trans_list,type_string):
    chinese = tree[type_string].replace('\n','').split()
    for ele in dict_trans_list:
        ele['end'] = len(chinese)-1
    matchedId_list = get_matchedId(tree['tree'],type_string)
    miss_list = []
    for i in range(0,len(chinese)):
        if (i not in matchedId_list) and (chinese[i] not in punctuation):
            miss_list.append(i)
    result_list = []
    if len(miss_list) > 0:
        dict_trans_list.sort(key=cmp_func)
        temp_list = []
        kt_ele = dict()
        for ele in dict_trans_list:
            if ele['key'] != 0:
                temp_list.append(ele)
            else:
                kt_ele = ele
        temp_list.append(kt_ele)
        dict_trans_list = temp_list
        # flag = False
        miss_miss_list = []
        for id in miss_list:
            flag = False
            lop_i = -1
            delta_min = 1000000
            for i in range(0,len(dict_trans_list)):
                dict_trans = dict_trans_list[i]
                length = len(dict_trans['trans'])
                if length != 0 and dict_trans['trans'][length-1] != dict_trans['end']:
                    flag, delta_temp = check_proper_A(dict_trans,id,length-1)
                else:
                    flag, delta_temp = check_proper_A(dict_trans,id,length-2)
                if flag == True:
                    if delta_temp <= delta_min:
                        delta_min = delta_temp
                        lop_i = i
                        # print("x")
                # if flag == True:
                #     # mlmk = id
                #     dict_trans['trans'].append(id)
                #     dict_trans['trans'].sort()
                #     break
            if lop_i != -1:
                dict_trans_list[lop_i]['trans'].append(id)
                dict_trans_list[lop_i]['trans'].sort()
            else:
                miss_miss_list.append(id)

        print(miss_miss_list)

        for id in miss_miss_list:
            flag = False
            lop_i = -1
            delta_min = 1000000
            for i in range(0,len(dict_trans_list)):
                dict_trans = dict_trans_list[i]
                length = len(dict_trans['trans'])
                if length != 0 and dict_trans['trans'][length-1] != dict_trans['end']:
                    flag, delta_temp = check_proper_B(dict_trans,id,length-1)
                else:
                    flag, delta_temp = check_proper_B(dict_trans,id,length-2)
                if flag == True:
                    if delta_temp <= delta_min:
                        delta_min = delta_temp
                        lop_i = i
                        # print("x")
                # if flag == True:
                #     # mlmk = id
                #     dict_trans['trans'].append(id)
                #     dict_trans['trans'].sort()
                #     break
            if lop_i != -1:
                dict_trans_list[lop_i]['trans'].append(id)
                dict_trans_list[lop_i]['trans'].sort()


    dict_trans_list.sort(key=lambda x: x['key'])
    for ele in dict_trans_list:
        result_list.append(ele['trans'])

    return result_list
###################################################################################################



def get_trans_list(adjunct_list,flag_,tree,comp_list,ch_dict):
    trans_list = []
    type_string = ""
    if flag_ == 1: ##baidu
        type_string = "baidu"
    elif flag_ == 2:
        type_string = "google"
    elif flag_ == 3:
        type_string = "bing"

    type_id = type_string + "_id"
    tran_ch_string = tree[type_string].replace('\n','').split()
    for list_ele in adjunct_list:
        temp_set = set()
        for ele in list_ele:
            for tran_index in ele[type_id]:
                if tran_ch_string[tran_index] not in punctuation:
                    temp_set.add(tran_index)
        temp_list = list(temp_set)
        temp_list.sort()
        trans_list.append(temp_list)

    ########################word alignment repair  对词对齐中可能出现的词语遗漏进行了修正，可以暂时忽略###############
    dict_trans_list = []
    for i in range(0,len(trans_list)):
        temp_dict = {}
        temp_dict['key'] = i
        temp_dict['trans'] = trans_list[i]
        dict_trans_list.append(temp_dict)

    trans_list = checkMissWord(tree,dict_trans_list,type_string)
    #############################################################################################################

    return trans_list


def get_engNode_string(list_):
    temp = ""
    for ele in list_:
        temp = temp + ele['label'] + " "
    return temp

def transAndResult(d1,score_list,type_A,type_B,tree,lab,pl,main_list):
    strings = ""
    strings = strings + type_A+": "+ tree[type_A]+type_B+": "+ tree[type_B]
    flag = 'same'
    for i in range(0,len(d1[0])):
        cut_score_str = ""
        if score_list[i] < threshold:
            synony_ = synony_check(d1[0][i], d1[1][i]) ####同义词词林检查
            cut_score = caclu_bert_bertscore(d1[0][i], d1[1][i])#########
            if synony_ == False and cut_score < 0.45:
                    # cut_score = 1.0
                cut_score_str = ";-;" + str(cut_score)
                flag = 'different'
            else:
                cut_score = 1.0
                cut_score_str = ";-;" + str(cut_score)
            # flag = 'different'
        eng_string = get_engNode_string(main_list[i])
        strings = strings + "      " + eng_string +'\n'
        strings = strings +"      " + d1[0][i] + ";-;" + d1[1][i] + ";-;"+str(score_list[i])+ cut_score_str +'\n'

    lb_temp = ""
    for ele in lab:
        if (pl[0] in ele) and (pl[2] in ele):
            lb_temp = ele
    # strings = strings + "      " + flag + lb_temp[1] + lb_temp + '\n'+'\n'
    strings = strings + "      " + flag + '\n' + '\n'
    return strings

###############check_real_different############################
import jieba
########jiebaAndLTP###############
def jiebaAndLTP(ch_str_re):
    temp_s = jieba.cut(ch_str_re, cut_all=False, HMM=True)
    list_1 = list(temp_s)
    list_2 = []
    for ele in list_1:
        seg_list = segmentor.segment(ele)
        for ele2 in seg_list:
            list_2.append(ele2)
    return list_2

##################################

def clear_stopwords(list_):
    result = []
    for word in list_:
        if word not in stop_wordlist:
            result.append(word)
    return result

def check_in_fv(cut_):
    list_ = []
    for ele in cut_:
        if ele in fv.model.index_to_key:
            list_.append(ele)
        else:
            topn = fv.most_similar(ele, topn=2,min_n = 1, max_n = 3)
            if len(topn) > 0:
                list_.append(topn[0][0])
            else:
                list_.append(" ")
    return list_

def get_fv_score(cut_piar_1,cut_pair_2):
    F_list = []
    for i in range(0,len(cut_piar_1)):
        F_score = fv.similarity(cut_piar_1[i],cut_pair_2[i])
        F_list.append(F_score)
    return F_list

def calcu_cutpair(cut1,cut2):
    cut_1 = check_in_fv(cut1)
    cut_2 = check_in_fv(cut2)
    row_len,column_len = len(cut_1),len(cut_2)
    cut_pair_1 = []
    cut_pair_2 = []
    for ele1 in cut_1:
        for ele2 in cut_2:
            cut_pair_1.append(ele1)
            cut_pair_2.append(ele2)
    # P, R, F1 = scoreRe.score(cut_pair_1, cut_pair_2, model_type="D:\\ARTICAL\\bert-base-chinese", verbose=True,num_layers=8)
    F1 = get_fv_score(cut_pair_1, cut_pair_2)
    score_list = [float(ele) for ele in F1]
    weight_matrix = np.zeros((row_len,column_len),dtype=float)
    print('x')
    for i in range(0,row_len):
        for j in range(0,column_len):
            weight_matrix[i][j] = score_list[i*column_len+j]
    y = 1000
    for i in range(0,row_len):
        x = 0
        for j in range(0,column_len):
            x = max(x,weight_matrix[i][j])
        y = min(y,x)

    for j in range(0,column_len):
        x = 0
        for i in range(0,row_len):
            x = max(x,weight_matrix[i][j])
        y = min(y,x)
    return y

def get_common(list_A,list_B):
    set_A = set(list_A)
    set_B = set(list_B)
    set_ = set_A & set_B
    return list(set_)


def remove_common(common_list,cut_):
    list_ = []

    for ele in cut_:
        if ele not in common_list:
            list_.append(ele)
    return list_

def caclu_bert_bertscore(str1,str2):
    import jieba
    str1 = str1.replace('\u200b',"")
    str2 = str2.replace('\u200b',"")
    cut_1 = jiebaAndLTP(str1)
    cut_2 = jiebaAndLTP(str2)
    cut_1 = clear_stopwords(cut_1)
    cut_2 = clear_stopwords(cut_2)
    common_list = get_common(cut_1,cut_2)
    cut_1 = remove_common(common_list,cut_1)
    cut_2 = remove_common(common_list,cut_2)
    if len(cut_1) == 0 and len(cut_2) == 0:
        return 1
    elif len(cut_1) == 0 or len(cut_2) == 0:
        return 0
    else:
        try:
            cut_score = calcu_cutpair(cut_1,cut_2)
        except:
            cut_score = 0.0
        return cut_score
############################################################
def synony_check(str1,str2):

    def get_nltk_sym(word):
        syn = set()
        print("oolfdd:", word)
        wordnet.synsets(word, lang='cmn')
        print('sds')
        for each in wordnet.synsets(word, lang='cmn'):
            list = each.lemma_names('cmn')
            for w in list:
                syn.add(w)
        return syn

    def get_syno(string):
        list_1 = jiebaAndLTP(string)
        moss_list = []
        for i in range(0,len(list_1)):
            w = list_1[i]
            temp_list = [w]
            syn = replacer.get_syno_sents_list(w)
            try:
                syn = list(set(syn).union(set(get_nltk_sym(w))))
            except:
                syn = list(set(syn))
            temp_list.extend(syn)
            moss_list.append(temp_list)
        return moss_list

    def get_remain_syno(syno_l,string):
        temp_string = string
        teag = 0
        for w_l in syno_l:
            for ele in w_l:
                if ele in temp_string:
                    temp_string = temp_string.replace(ele,"",1)
                    teag += 1
                    break

        for word in stop_wordlist:
            if word in temp_string:
                temp_string = temp_string.replace(word, "", 1)

        return temp_string,teag


    syno_1 = get_syno(str1)
    syno_2 = get_syno(str2)

    remain_syno1_str2,tagA = get_remain_syno(syno_1,str2)
    remain_syno2_str1,tagB = get_remain_syno(syno_2,str1)
    if (len(remain_syno1_str2) == 0 and tagA == len(syno_1)) or (len(remain_syno2_str1) == 0 and tagB == len(syno_2)):
        return True
    else:
        return False


############################################################
def catch_label_key(d1,score_list,type_A,type_B,tree):

    strings = ""
    strings = strings + type_A+": "+ tree[type_A]+type_B+": "+ tree[type_B]
    flag = 'same'
    for i in range(0,len(d1[0])):
        if score_list[i] < threshold:
            synony_ = synony_check(d1[0][i],d1[1][i])
            cut_score = caclu_bert_bertscore(d1[0][i], d1[1][i])
            if synony_ == False and cut_score < 0.45:
                flag = 'different'

    return flag


def get_match_bert_score(d1,bert_score_dicts):
    temp_list = []
    for i in range(0, len(d1[0])):
        score = bert_score_dicts[d1[0][i] + ';-;' + d1[1][i]]
        temp_list.append(score)
    return temp_list

def get_match_sbert(d1):
    temp_list = []
    for i in range(0, len(d1[0])):
        e1 = model.encode(normal(d1[0][i]))
        e2 = model.encode(normal(d1[1][i]))
        score = 1 - distance.cosine(e1, e2)
        # score = bert_score_dicts[d1[0][i] + ';-;' + d1[1][i]]
        temp_list.append(score)
    return temp_list

def write_other(d1,score_list,type_A,type_B,tree,lab,pl,four_t,main_list):
    temp_string = "#" + str(i1 + 1) + ": " + contents[0] + '\n'
    temp_string = temp_string + transAndResult(d1,score_list,type_A,type_B,tree,lab,pl,main_list) + '\n'
    with open("./compute_result" + metric_type + '_'+four_t, 'a', encoding='utf-8') as f:
        f.write(temp_string)

def calculate(d1,bert_score_dicts):
    # temp_list = get_match_bert_score(d1,bert_score_dicts)
    temp_list = get_match_sbert(d1)
    # temp_list = get_match_simcse(d1)
    return temp_list


####################检查句法结构结点中是否有SBAR、SBARQ等从句节点##################
def check_have_other_clauses(list_):
    for ele in list_:
        if ele['label'] in ['SBAR','SBARQ','SINV','SQ']:
            return True
    return False
#########################################################################

if __name__ == '__main__':
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    #########导入源句 及  人标数据##########################
    with open('./dataset/pred_dataset/'+kinds+'.tgt',encoding='utf-8') as file_obj:
        contents = file_obj.read()
        array = contents.split('\n')

    bert_score_sentence = []
    bert_score_dicts = {}

    print(bert_score_dicts)
    with open("./compute_result"+metric_type, 'w', encoding='utf-8') as f:
        f.write("")

    for i1 in range(0, len(array)):
        # i1 = 134
        print(i1,"-p-p-p-p-p-p",len(array))
        temp_list = []
        contents = array[i1].split(";-;")
        temp_list.append(contents[0])
        # print(i1)
        content_tree = dict()
        for sent_tree_ele in sent_tree_list_:
            if contents[0]+'\n' == sent_tree_ele['origin']:
                content_tree = sent_tree_ele
        clause_flag = check_have_other_clauses(content_tree['tree'])  ####################检查句法结构结点中是否有SBAR、SBARQ等从句节点
        adjunct_list = get_cut_adjunct(content_tree['tree'][1], content_tree['tree'], clause_flag)  ##########如果有SBAR等 就切分SBAR等节点，否则 切分S类型节点划分修饰部分
        main_list = add_main_list(adjunct_list,dict_comp[contents[0]+'\n'],content_tree['tree'])
        if len(contents) > 0:
            baidu_list = get_trans_list(main_list,1,content_tree,dict_comp[contents[0]+'\n'],ch_align_dict[contents[0]+'\n'])
            google_list = get_trans_list(main_list,2,content_tree,dict_comp[contents[0]+'\n'],ch_align_dict[contents[0]+'\n'])
            bing_list = get_trans_list(main_list,3,content_tree,dict_comp[contents[0]+'\n'],ch_align_dict[contents[0]+'\n'])
            print(adjunct_list)
            #########get_pair函数用来进行 英文NER（命名实体）对应的中文 进行英文的替换####################################################
            d1 = get_pair(google_list,baidu_list,"google","baidu",content_tree)
            d2 = get_pair(bing_list,google_list,"bing","google",content_tree)
            d3 = get_pair(bing_list, baidu_list, "bing", "baidu", content_tree)
            score_list1 = calculate(d1,bert_score_dicts)
            score_list2 = calculate(d2,bert_score_dicts)
            score_list3 = calculate(d3,bert_score_dicts)


            strings = "#"+str(i1+1) + ": " + contents[0]+'\n'
            strings = strings + transAndResult(d1,score_list1,"baidu","google",content_tree,"",'1 2',main_list)
            strings = strings + transAndResult(d2, score_list2, "google","bing", content_tree,"",'2 3',main_list)
            strings = strings + transAndResult(d3, score_list3, "baidu", "bing", content_tree,"",'1 3',main_list)


            # strings = strings + transAndResult(d1, score_list1, "baidu", "google", content_tree,
            #                                    "", '1 2', main_list)
            # strings = strings + transAndResult(d2, score_list2, "google", "bing", content_tree,
            #                                    "", '2 3', main_list)
            # strings = strings + transAndResult(d3, score_list3, "baidu", "bing", content_tree,
            #                                    "", '1 3', main_list)

            strings = strings + '\n'
            with open("./compute_result"+metric_type,'a',encoding='utf-8') as f:
                f.write(strings)
