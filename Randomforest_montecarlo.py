# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# from sklearn.externals.six import StringIO
import queue
# from sklearn.metrics.ranking import roc_auc_score
import os
import random
from decimal import *

threshold = 10e-7  
smooth = 0.001
gram = 6  # gram=3 buxing
N = 100000


data = []
target = []
rfclass = []  # 分类的数字形式
classNum = 0

Guess = queue.Queue()  # tuple:pro
GuessSort = {}
#psw_set="12306"
file = open("myspace.txt",encoding='utf-8')
testfile=open("test.txt",encoding='utf-8')
fout = open("randpick.txt", "w",encoding='utf-8')
gout = open("result.txt",'w',encoding='utf-8')

psw_pro = []
acc_rank = [1]
st_vec = [1] * gram

#########################cal char feature dic
keyboard_pattern = ["1234567890-=", "qwertyuiop[]\\", "asdfghjkl;\'", "zxcvbnm,./"]
shift_keyboard_pattern = ["!@#$%^&*()_+", "QWERTYUIOP{}|", "ASDFGHJKL:\"", "ZXCVBNM<>?"]

kp_dic = {}  # 键盘特征
for i in range(len(keyboard_pattern)):
    for j in range(len(keyboard_pattern[i])):
        kp_dic[keyboard_pattern[i][j]] = (i + 1, j + 1)
        kp_dic[shift_keyboard_pattern[i][j]] = (i + 1, j + 1)

no_dic = {1:(1, 0)}  # 序号特征
spchr = 1
for i in range(32, 127):  # 可见字符32-126
    if i >= ord('0') and i <= ord('9'):
        no_dic[i] = (0, i - ord('0'))
        if i == ord('0'):
            no_dic[i] = (0, 10)
    elif i >= ord('a') and i <= ord('z'):
        no_dic[i] = (3, i - ord('a') + 1)
    elif i >= ord('A') and i <= ord('Z'):
        no_dic[i] = (2, i - ord('A') + 1)
    else :
        no_dic[i] = (1, spchr)
        spchr += 1

feature_dic={}
for k in no_dic:
    if k==1:
        feature_dic[1]=[1,0,0,0]
    else:
        if chr(k) in kp_dic:
            feature_dic[k]=list(no_dic[k]+kp_dic[chr(k)])
        else :
            feature_dic[k]=list(no_dic[k]+(0,0))

#############################

def str2vec(str):
    vec = st_vec[:]
    for c in str:
        vec.append(ord(c))
        # print ord(c)
    vec.append(0)
    return vec

def getType(ch):
    if ch == 1:
        return 0
    if ch >= ord('0') and ch <= ord('9'):
        return 1
    elif ch >= ord('a') and ch <= ord('z') or ch >= ord('A') and ch <= ord('Z'):
        return 2
    else:
        return 3
    
    
def getFeaturevec(vec, last_pos):
    #ch = vec[-1]
    l = vec.__len__() - gram
    cur_len = l + gram - 1 - last_pos
    rtn = [l, cur_len] 
    
    for c in vec[-gram:]:
        if c in feature_dic:
            rtn += feature_dic[c]
        else:
            rtn +=feature_dic[ord('?')]
        
    return rtn
    


smooth_classNum = 0
def randPick(rf, vec,last_pos):
    rand_pro = random.random()
    p = 0
    sum = 0
    feature_vec = getFeaturevec(vec, last_pos)
    cur_type = getType(vec[-1])
    
    for prob in rf.predict_proba([feature_vec])[0]:
        sum += (prob + smooth) / smooth_classNum
        if sum > rand_pro:
            g_type = getType(rfclass[p])
            g_pos = last_pos
            if cur_type != g_type:
                g_pos = vec.__len__() - 1
            return rfclass[p], (prob + smooth) / smooth_classNum,g_pos
        p+=1
        
def calPro(rf,vec,last_pos):
    pro=1

    for i in range(gram, vec.__len__()):
        feature_vec = getFeaturevec(vec[:i], last_pos)
        p=0
        for prob in rf.predict_proba([feature_vec])[0]:
            if rfclass[p]==vec[i]:
                pro*=(prob + smooth) / smooth_classNum
                cur_type = getType(vec[i - 1])
                g_type = getType(vec[i])
                
                if cur_type != g_type:
                    last_pos = i - 1
                break
            p+=1        
    return pro      
        
        

def rankCal(rf,psw):
    vec=str2vec(psw)
    last_pos=-1
    pro=calPro(rf,vec,last_pos)
    
    for i in range(N, 0, -1):
        p = psw_pro[i-1][1]
        if p > pro:
            return acc_rank[i]
    return acc_rank[0]   


n = 0

for line in file:
    n = n + 1

    psw = line.strip('\r\n')  # [:-1]#remove \n
    if len(psw)>30:
        continue
    vec = str2vec(psw)
# vec [1, 1, 1, 1, 1, 1, 54, 53, 54, 54, 52, 49, 55, 0]
    last_pos = -1
    for i in range(gram, vec.__len__()):
        if vec[i]>=127:
            break
        feature_vec = getFeaturevec(vec[:i], last_pos)
        # print vec[:i]
        # print feature_vec
        # feature_vec [0, 6, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        # [1, 1, 1, 1, 1, 1]
        # [0, 6, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        # [1, 1, 1, 1, 1, 1, 54]
        # [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 6, 1, 6]
        # [1, 1, 1, 1, 1, 1, 54, 53]
        # [2, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 6, 1, 6, 0, 5, 1, 5]
        # [1, 1, 1, 1, 1, 1, 54, 53, 54]
        # [3, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 6, 1, 6, 0, 5, 1, 5, 0, 6, 1, 6]
        # [1, 1, 1, 1, 1, 1, 54, 53, 54, 54]
        # [4, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 6, 1, 6, 0, 5, 1, 5, 0, 6, 1, 6, 0, 6, 1, 6]
        data.append(feature_vec)
        target.append(vec[i])
        cur_type = getType(vec[i - 1])
        g_type = getType(vec[i])
        
        if cur_type != g_type:
            last_pos = i - 1

# print data[:10]
# print target[:10]
# exit()
# data[1] [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 6, 1, 6]
# target[1] 53

rf = RandomForestClassifier(n_estimators=30, max_features=0.8, min_samples_leaf=10, random_state=10, n_jobs=6)
print('rrr')
rf.fit(data, target)
print('fff')

for k in rf.classes_:
    rfclass.append(k)
# print rf.classes_
# print rfclass
#print rf.oob_score_
classNum = rfclass.__len__()
smooth_classNum = 1 + smooth * classNum
rf.n_jobs=1

getcontext().prec=50

data=[]
target=[]

for i in range(0, N):
    vec = st_vec[:]
    pro = 1
    last_pos=-1
    while 1:
        c, prob,last_pos = randPick(rf, vec,last_pos)
        vec.append(c)
        pro *= prob
        if c == 0:
            break 
    # print i
    psw_pro.append([vec[gram:-1], pro])
    
psw_pro = sorted(psw_pro, key=lambda b:b[1], reverse=True)
for k in psw_pro:
    for t in k[0]:
        fout.write(chr(t))
    fout.write('\t'+str(k[1])+'\n')
    
# decendent

for i in range(0, N):
    acc_rank.append(acc_rank[i] + 1 / (N * psw_pro[i][1]))
    

for line in testfile:
    # list=line.split('\t')
    psw=line.strip('\r\n')
    gout.write(psw+'\t'+str(rankCal(rf,psw))+'\n')








