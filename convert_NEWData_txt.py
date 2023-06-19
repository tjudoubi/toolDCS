import pandas as pd
import jieba
from nltk.parse import CoreNLPParser
from pyltp import Segmentor
import re
import os

from stanfordcorenlp import StanfordCoreNLP

list_en = []
list_ch = []
list_ch_split = []
pattern_re = "[（|\(][a-zA-Z|’|“|”|‘|\s|\'|\"]+[）|\)]"

cut_type = "jiebaAndLTP"

#################################################
# LTP分词模型的路径
LTP_DATA_DIR = 'D:\\ARTICAL\\ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# 实例化
segmentor = Segmentor()
# 加载分词模型
segmentor.load(cws_model_path)
##################################################

#########stanford Parser#####
stan = StanfordCoreNLP("D:/ARTICAL/TestTranslation/stanford-corenlp-full-2018-10-05/stanford-corenlp-full-2018-10-05",lang='zh')
def tokenCutStanford(ch_str_re):
    seg_results = stan.word_tokenize(ch_str_re)
    return seg_results
##############################

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

eng_parser = CoreNLPParser('http://localhost:9000')
def turn_form(sentence):
    words = eng_parser.tokenize(sentence)
    list_ = []
    for ele in words:
        list_.append(ele)
    return list_

def transform(ch_str,en_str):
    ch_str_re = re.sub(pattern_re,'',ch_str)
    # seg_list = jieba.cut(ch_str_re, cut_all=False, HMM=True)
    # seg_list = segmentor.segment(ch_str_re)
    # seg_list = thuseg.cut(ch_str_re,text=True).split()
    # seg_list = pynlpir.segment(ch_str_re, pos_tagging=False)
    # seg_list = tokenCutTrankit(ch_str_re)
    # seg_list = tokenCutStanford(ch_str_re)
    seg_list = jiebaAndLTP(ch_str_re)
    seg_ch = ' '.join(seg_list)
    seg_ch = re.sub('\s+', ' ', seg_ch)

    seg_list = turn_form(en_str)
    seg_en = ' '.join(seg_list)
    seg_en = re.sub('\s+', ' ', seg_en)

    return seg_ch,seg_en

kinds = ["sampleSentence"]
# for i in range(0,len(kinds)):
#     kinds[i] = kinds[i] + "_chinese"
for kind in kinds:
    # kind = 'tech'
    list_en = []
    list_ch = []
    list_ch_split = []
    business_baidu_ch = pd.read_pickle('NEWData/'+kind+'/'+kind+'_chinese_Baidu.dat')
    for ele in business_baidu_ch.keys():
        print(ele)
        ch_str = business_baidu_ch[ele]
        seg_ch, seg_en = transform(ch_str,ele)
        list_en.append(seg_en)
        list_ch.append(ch_str)
        list_ch_split.append(seg_ch)


    business_bing_ch = pd.read_pickle('NEWData/'+kind+'/'+kind+'_chinese_Bing.dat')
    for ele in business_bing_ch.keys():
        ch_str = business_bing_ch[ele]
        seg_ch, seg_en = transform(ch_str,ele)
        list_en.append(seg_en)
        list_ch.append(ch_str)
        list_ch_split.append(seg_ch)

    business_google_ch = pd.read_pickle('NEWData/'+kind+'/'+kind+'_chinese_Google.dat')
    for ele in business_google_ch.keys():
        ch_str = business_google_ch[ele]
        seg_ch, seg_en = transform(ch_str,ele)
        list_en.append(seg_en)
        list_ch.append(ch_str)
        list_ch_split.append(seg_ch)


    with open('./alignment/'+kind+'/test.tgt', 'w', encoding='utf-8') as f:
        for ele in list_en:
            f.write(ele+'\n')
        f.close()

    with open('./alignment/'+kind+'/test_split_'+cut_type+".src",'w',encoding='utf-8') as f:
        for ele in list_ch_split:
            f.write(ele+'\n')
        f.close()

    with open('alignment/'+kind+'/test.src', 'w', encoding='utf-8') as f:
        for ele in list_ch:
            f.write(ele+'\n')
        f.close()