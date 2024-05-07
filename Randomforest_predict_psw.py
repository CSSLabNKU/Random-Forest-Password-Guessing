# -*- coding:utf8 -*-

"""
    Description:
    
    This script is used to predict the password based on the random forest model.
"""

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import tree
import numpy as np
import queue
import os
import time


threshold = 1e-7  # 10e-7:1 hour,60000+guess,10e-8:5h,80w guess
smooth = 0.001
gram = 6  # PREFIX LENGTH

data = []
target = []
classifier_class = []  # LABEL CLASS
classNum = 0
Guess = queue.Queue()  # tuple:pro
GuessSort = {}

dataset_folder = './datasets/'
result_folder = './results/'
train_filename = "csdn_50_train_big.txt"
output_filename_format = "csdn_{0}_result.txt"

file = open(f"{dataset_folder}{train_filename}", 'r', encoding = 'utf-8', errors = 'ignore')


st_vec = [1] * gram
######################### cal char feature dic
keyboard_pattern = [
    "1234567890-=", 
    "qwertyuiop[]\\", 
    "asdfghjkl;\'", 
    "zxcvbnm,./"
]

shift_keyboard_pattern = [
    "!@#$%^&*()_+", 
    "QWERTYUIOP{}|", 
    "ASDFGHJKL:\"", 
    "ZXCVBNM<>?"
]


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
    # ch = vec[-1]
    l = vec.__len__() - gram
    cur_len = l + gram - 1 - last_pos
    rtn = [l, cur_len]

    for c in vec[-gram:]:
        rtn += feature_dic[c]

    return rtn


smooth_classNum = 0


def predictpro_psw(classifier, vec, Pro, last_pos):
    # print Str, tar_vec[-gram-1:-1]
    p = -1
    Pro = Pro / smooth_classNum
    feature_vec = getFeaturevec(vec, last_pos)
    cur_type = getType(vec[-1])

    classifier_prob = classifier.predict_proba([feature_vec])[0]
    vec.append(-1)
    for prob in classifier_prob:
        p = p + 1
        newpro = Pro * (prob + smooth)  # laplace smooth
        if newpro < threshold:
            continue;
        vec[-1] = classifier_class[p]

        if classifier_class[p] == 0:
            if vec.__len__() == gram + 1:
                continue;
            # print vec, newpro
            # print("Type GuessSort", type(GuessSort))
            # print("Type vec: ", type(vec))
            GuessSort[tuple(vec[gram:-1])] = newpro
            # fout.write(newstr+'\t'+str(newpro)+'\n')
        else:
            g_type = getType(classifier_class[p])
            g_pos = last_pos
            if cur_type != g_type:
                g_pos = vec.__len__() - 2
            Guess.put((vec[:], newpro, g_pos))



rf = RandomForestClassifier(n_estimators=50, max_features=0.8, min_samples_leaf=10, random_state=10, oob_score=True, n_jobs=4)

classifiers = {
    "rf": rf,
}   
# 9w data,50 tree,3G

# key = "bag"
for key in classifiers:
    data = []
    target = []
    classifier_class = []  # LABEL CLASS
    classNum = 0
    Guess = queue.Queue()  # tuple:pro
    GuessSort = {}


    kp_dic = {}  # keyboard pattern
    for i in range(len(keyboard_pattern)):
        for j in range(len(keyboard_pattern[i])):
            kp_dic[keyboard_pattern[i][j]] = (i + 1, j + 1)
            kp_dic[shift_keyboard_pattern[i][j]] = (i + 1, j + 1)

    no_dic = {1: (1, 0)}  # TYPE pattern
    spchr = 1
    for i in range(32, 127):  # PRINTABLE ASCII CHARs (\x20-\x7e)
        if i >= ord('0') and i <= ord('9'):
            no_dic[i] = (0, i - ord('0'))
            if i == ord('0'):
                no_dic[i] = (0, 10)
        elif i >= ord('a') and i <= ord('z'):
            no_dic[i] = (3, i - ord('a') + 1)
        elif i >= ord('A') and i <= ord('Z'):
            no_dic[i] = (2, i - ord('A') + 1)
        else:
            no_dic[i] = (1, spchr)
            spchr += 1

    for i in range(0, 32):
        no_dic[i] = (4, i + 1)

    for i in range(127, 256):
        no_dic[i] = (4, i - 127 + 33)

    feature_dic = {}
    for k in no_dic:
        if k == 1:
            feature_dic[1] = [1, 2, 0, 0]
        else:
            if chr(k) in kp_dic:
                feature_dic[k] = list(no_dic[k] + kp_dic[chr(k)])
            else:
                feature_dic[k] = list(no_dic[k] + (0, 0))
    n = 0
    for line in file:
        n = n + 1
        psw = line.strip()
        vec = str2vec(psw)
        last_pos = -1
        for i in range(gram, vec.__len__()):
            if vec[i] >= 127:
                break
            feature_vec = getFeaturevec(vec[:i], last_pos)

            data.append(feature_vec)
            target.append(vec[i])
            cur_type = getType(vec[i - 1])
            g_type = getType(vec[i])

            if cur_type != g_type:
                last_pos = i - 1

    print(f"PROCESSING: {key} ...")
    classifier = classifiers[key]

    classifier.fit(data, target)

    print("fit DONE")
    for k in rf.classes_:
        classifier_class.append(k)
    classNum = classifier_class.__len__()
    smooth_classNum = 1 + smooth * classNum

    t3 = time.time()
    Guess.put((st_vec, 1, -1))  # The last position of the last segment
    while not Guess.empty():
        g = Guess.get()  # str:pro
        predictpro_psw(classifier, g[0], g[1], g[2])
    t4 = time.time() - t3
    print(t4)
    GuessSort = sorted(GuessSort.items(), key=lambda b: b[1], reverse=True)

    fout = open(result_folder + output_filename_format.format(key), 'w', encoding = 'utf-8', errors = 'ignore')
    for k in GuessSort:
        for t in k[0]:
            fout.write(chr(t))
        fout.write('\t' + str(k[1]) + '\n')
    fout.close()
    print(f"{key} DONE ... ")