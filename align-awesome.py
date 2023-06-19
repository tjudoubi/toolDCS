import pickle

align_list = []
ch = []
en = []
top_index = 0
cut_type = "_jiebaAndLTP"
kinds = 'sampleSentence'
with open('./alignment/'+kinds+'/'+ kinds + '_zhen'+cut_type+'.src-tgt.align','r',encoding='utf-8') as f:
    align_list_2 = f.readlines()
    # print(align_list)

with open('./alignment/'+kinds+'/test_split'+cut_type +'.src','r',encoding='utf-8') as f:
    ch = f.readlines()
    # print(ch)
with open('./alignment/'+kinds+'/'+'test.tgt','r',encoding='utf-8') as f:
    en = f.readlines()
#
# with open('./NEWData/alignment_src_tgt.txt','r',encoding='utf-8') as f:
#     align_list_2 = f.readlines()

top_index = len(align_list_2)

match_list = []
index = 0
ch_index = 0
import re
while ch_index < len(ch):
    # re_string = "\s+"
    # ch[ch_index] = re.sub(re_string,' ',ch[ch_index])
    ch[ch_index] = ch[ch_index].replace('   ',' ').replace('  ',' ')
    ch_index += 1

# print(ch[402])

while index < top_index:
    # tgt_src = align_list[index].replace('\n',"").split(' ')
    src_tgt = align_list_2[index].replace('\n',"").split(' ')

    tgt = en[index].replace('\n',"").split(' ')
    src = ch[index].replace('\n',"").split(' ')
    temp_dict ={}
    for i in range(0,len(tgt)):
        temp_dict[i] = set()
    # for ele in tgt_src:
    #     tgtANDsrc = ele.split('-')
    #     # print(int(tgtANDsrc[0]))
    #     temp_dict[int(tgtANDsrc[0])-1].add(int(tgtANDsrc[1])-1)
    if src_tgt == ['']:
        for i in range(0, len(tgt)):
            temp_dict[i] = list(temp_dict[i])
        match_list.append(temp_dict)
        index += 1
        continue
    for ele in src_tgt:
        srcANDtgt = ele.split('-')
        print(int(srcANDtgt[0]),srcANDtgt)
        temp_dict[int(srcANDtgt[1])].add(int(srcANDtgt[0]))
    for i in range(0,len(tgt)):
        temp_dict[i] = list(temp_dict[i])
        temp_dict[i].sort()
        print('x')
    match_list.append(temp_dict)
    index += 1

all_list_ = []
index = 0
while index < top_index/3:
    index_1 = index
    index_2 = int(index_1 + top_index/3)
    index_3 = int(index_1 + (top_index/3)*2)
    # if en_tran[index_1] != en_tran[index_2] or en_tran[index_1] != en_tran[index_3] or en_tran[index_2] != en_tran[index_3]:
    #     print(index,en_tran)
    dict_ = {}
    # dict_['origin'] = en_tran[index_1]
    dict_['baidu'] = match_list[index_1]
    dict_['bing'] = match_list[index_2]
    dict_['google'] = match_list[index_3]
    all_list_.append(dict_)
    index += 1
with open('./alignment/'+kinds+'/'+ kinds + '_align_three_200_zhen'+cut_type,'wb') as f:
    pickle.dump(all_list_,f)

index = 0
while index < top_index:
    tgt = en[index].replace('\n', "").split(' ')
    src = ch[index].replace('\n', "").split(' ')
    with open('./alignment/'+kinds+'/'+'write_Align_combine_new'+cut_type+'.txt','a',encoding='utf-8') as f:
        f.write('#'+str(index+1))
        f.write(en[index])
        f.write(ch[index])
        f.close()
    i = 0
    string = ""
    for i in range(i,len(tgt)):
        string = string + tgt[i] + '{'
        temp = ""
        for ele in match_list[index][i]:
            string = string + src[ele] + " "
        string = string + '}'
    string = string + '\n'+'\n'

    with open('./alignment/'+kinds+'/'+'write_Align_combine_new'+cut_type+'.txt', 'a', encoding='utf-8') as f:
        f.write(string)
        f.close()
    index += 1