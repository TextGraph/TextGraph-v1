import numpy as np
import pickle
import scipy.sparse as sp
import re

global_word_limit = 150


def txt2pkl(path):
    file = open(path, "r", encoding="utf8")
    glove = {}
    for line in file:
        line_tra = line.strip().split(" ")
        glove[line_tra[0]] = np.array(list(map(float, line_tra[1:])))
    with open(r"../data/glove_300d.pkl", "wb") as f:
        pickle.dump(glove, f)


def get_sp_adj(mini_matrxi, word_count):
    row = []
    col = []
    data = []
    for key in mini_matrxi:
        row.append(key[0])
        col.append(key[1])
        if key[1] != key[0] and key[1] != 0:
            data.append(mini_matrxi[key] / np.maximum(word_count - 4 + 1, 1))  # np.maximum(word_count-4+1,1)
        else:
            data.append(mini_matrxi[key])
    item_adj_sp = sp.coo_matrix((data, (row, col)), shape=(global_word_limit, global_word_limit))
    return item_adj_sp


def computeTfidf(idf, line, word_dict, mini_matrxi):
    word_pre = {}
    count = 0
    for word in line:
        count += 1
        if word not in word_pre:
            word_pre[word] = 1
        else:
            word_pre[word] += 1
    for word in word_dict:
        mini_matrxi[(word_dict[word], 0)] = word_pre[word] * idf[word] / count
        # mini_matrxi[(0,word_dict[word])] = word_pre[word] * idf[word]/count
    mini_matrxi[(0, 0)] = 1
    return mini_matrxi


def windoms_adj(line, win_size, loc_item, words_limit, seq_limit, idf_dict):
    i = 0
    slice_count = 0
    j = win_size
    dict_fre = {}
    word_fre = {}
    word_fre[0] = 1
    mini_matrxi = {}
    seq_line = []
    seq_line.append(-1)
    word_count = 1
    seq_index = []
    length = len(line)
    if win_size > length:
        line_tra = line
        length_line = len(line_tra)
        for k in range(length_line - 1):
            if line_tra[k] not in dict_fre:
                dict_fre[line_tra[k]] = word_count
                word_this = loc_item[line_tra[k]]
                seq_line.append(word_this)
                word_count += 1
            if dict_fre[line_tra[k]] not in word_fre:
                word_fre[dict_fre[line_tra[k]]] = 1
            else:
                word_fre[dict_fre[line_tra[k]]] += 1
            for m in range(k + 1, length_line):
                if line_tra[m] not in dict_fre:
                    dict_fre[line_tra[m]] = word_count
                    word_this = loc_item[line_tra[m]]
                    seq_line.append(word_this)
                    word_count += 1
                    if word_count >= words_limit:
                        mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[m]])] = mini_matrxi.get(
                            (dict_fre[line_tra[k]], dict_fre[line_tra[m]]), 0) + 0.5
                        mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[k]])] = mini_matrxi.get(
                            (dict_fre[line_tra[m]], dict_fre[line_tra[k]]), 0) + 0.1
                        mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[k]])] = 1
                        mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[m]])] = 1
                        mini_matrxi = computeTfidf(idf_dict, line, dict_fre, mini_matrxi)
                        for word in line[:m]:
                            seq_index.append(dict_fre[word])
                        if len(seq_index) > seq_limit:
                            seq_index = seq_index[:seq_limit]
                        else:
                            seq_index += (seq_limit - len(seq_index)) * [-1]
                        if dict_fre[line_tra[m]] not in word_fre:
                            word_fre[dict_fre[line_tra[m]]] = 1
                        else:
                            word_fre[dict_fre[line_tra[m]]] += 1
                        return get_sp_adj(mini_matrxi, word_count), seq_line, seq_index
                mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[m]])] = mini_matrxi.get(
                    (dict_fre[line_tra[k]], dict_fre[line_tra[m]]), 0) + 0.5
                mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[k]])] = mini_matrxi.get(
                    (dict_fre[line_tra[m]], dict_fre[line_tra[k]]), 0) + 0.1
            mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[k]])] = 1
        if line_tra[-1] not in dict_fre:
            dict_fre[line_tra[-1]] = word_count
            word_this = loc_item[line_tra[-1]]
            seq_line.append(word_this)
            word_count += 1
        if dict_fre[line_tra[-1]] not in word_fre:
            word_fre[dict_fre[line_tra[-1]]] = 1
        else:
            word_fre[dict_fre[line_tra[-1]]] += 1
        mini_matrxi[(dict_fre[line_tra[-1]], dict_fre[line_tra[-1]])] = 1
        mini_matrxi = computeTfidf(idf_dict, line, dict_fre, mini_matrxi)
        seq_line += (words_limit - len(seq_line)) * [-1]
        for word in line:
            seq_index.append(dict_fre[word])
        if len(seq_index) > seq_limit:
            seq_index = seq_index[:seq_limit]
        else:
            seq_index += (seq_limit - len(seq_index)) * [-1]
        return get_sp_adj(mini_matrxi, word_count), seq_line, seq_index

    while j <= length:
        slice_count += 1
        line_tra = line[i:j]
        length_line = len(line_tra)
        for k in range(length_line - 1):
            if line_tra[k] not in dict_fre:
                dict_fre[line_tra[k]] = word_count
                word_this = loc_item[line_tra[k]]
                seq_line.append(word_this)
                word_count += 1
            if dict_fre[line_tra[k]] not in word_fre:
                word_fre[dict_fre[line_tra[k]]] = 1
            else:
                word_fre[dict_fre[line_tra[k]]] += 1
            for m in range(k + 1, length_line):
                if line_tra[m] not in dict_fre:
                    dict_fre[line_tra[m]] = word_count
                    word_this = loc_item[line_tra[m]]
                    seq_line.append(word_this)
                    word_count += 1
                    if word_count >= words_limit:
                        mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[m]])] = mini_matrxi.get(
                            (dict_fre[line_tra[k]], dict_fre[line_tra[m]]), 0) + 0.5
                        mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[k]])] = mini_matrxi.get(
                            (dict_fre[line_tra[m]], dict_fre[line_tra[k]]), 0) + 0.1
                        mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[k]])] = 1
                        mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[m]])] = 1
                        mini_matrxi = computeTfidf(idf_dict, line, dict_fre, mini_matrxi)
                        for word in line[:i + m + 1]:
                            seq_index.append(dict_fre[word])
                        if len(seq_index) > seq_limit:
                            seq_index = seq_index[:seq_limit]
                        else:
                            seq_index += (seq_limit - len(seq_index)) * [-1]
                        if dict_fre[line_tra[m]] not in word_fre:
                            word_fre[dict_fre[line_tra[m]]] = 1
                        else:
                            word_fre[dict_fre[line_tra[m]]] += 1
                        return get_sp_adj(mini_matrxi, word_count), seq_line, seq_index
                mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[m]])] = mini_matrxi.get(
                    (dict_fre[line_tra[k]], dict_fre[line_tra[m]]), 0) + 0.5
                mini_matrxi[(dict_fre[line_tra[m]], dict_fre[line_tra[k]])] = mini_matrxi.get(
                    (dict_fre[line_tra[m]], dict_fre[line_tra[k]]), 0) + 0.1
            mini_matrxi[(dict_fre[line_tra[k]], dict_fre[line_tra[k]])] = 1
        if line_tra[-1] not in dict_fre:
            dict_fre[line_tra[-1]] = word_count
            word_this = loc_item[line_tra[-1]]
            seq_line.append(word_this)
            word_count += 1
        if dict_fre[line_tra[-1]] not in word_fre:
            word_fre[dict_fre[line_tra[-1]]] = 1
        else:
            word_fre[dict_fre[line_tra[-1]]] += 1
        mini_matrxi[(dict_fre[line_tra[-1]], dict_fre[line_tra[-1]])] = 1
        i += 1
        j += 1
    seq_line += (words_limit - len(seq_line)) * [-1]
    mini_matrxi = computeTfidf(idf_dict, line, dict_fre, mini_matrxi)
    for word in line:
        seq_index.append(dict_fre[word])
    if len(seq_index) > seq_limit:
        seq_index = seq_index[:seq_limit]
    else:
        seq_index += (seq_limit - len(seq_index)) * [-1]
    return get_sp_adj(mini_matrxi, word_count), seq_line, seq_index


def get_index(file_name):
    loc_index = {}
    idf_dict = {}
    count = 0
    count_line = 0
    file = open(r"../data/" + file_name + ".txt", "r", encoding="utf8")
    count_pass = 0
    word_pre = {}
    for line in file:
        line_ = line.strip().split("\t")
        if len(line_) <= 1:
            print("line:", line_)
            count_pass += 1
            continue
        word_line = line_[1]
        line_tra = word_line.split(" ")
        count_line += 1
        for i in line_tra:
            if i not in word_pre:
                word_pre[i] = 1
            else:
                word_pre[i] += 1
            if i not in loc_index:
                loc_index[i] = count
                count += 1
        set_line = set(line_tra)
        for word in set_line:
            if word not in idf_dict:
                idf_dict[word] = 1
            else:
                idf_dict[word] += 1
    print("count_pass:", count_pass)
    for idf in idf_dict:
        idf_dict[idf] = np.log(count_line / idf_dict[idf])
    with open("../data/loc_index.pkl", "wb") as f:
        pickle.dump(loc_index, f)
    with open("../data/word_pre.pkl", "wb") as f2:
        pickle.dump(word_pre, f2)
    with open("../data/idf_dict.pkl", "wb") as f3:
        pickle.dump(idf_dict, f3)
    file.close()


def get_inputs():
    with open("../data/loc_index.pkl", "rb") as f:
        loc_index = pickle.load(f)
    with open("../data/glove_300d.pkl", "rb") as f1:  # glove_300d
        word_emb = pickle.load(f1)

    length = len(loc_index)
    emb_matr = np.zeros((length, 300))
    count = 0
    for i in loc_index:
        if i in word_emb:
            emb_matr[loc_index[i]] = word_emb[i]
        else:
            count += 1
            # emb_matr[loc_index[i]] = np.random.standard_normal((1, 300))
            emb_matr[loc_index[i]] = np.zeros((1, 300))
    with open("../data/inputs_loc.pkl", "wb") as f3:
        pickle.dump(emb_matr.astype(np.float32), f3)
    print("non_word:", count)


def get_adj(win_size, words_limit, seq_limit, file_name):
    seq_list = []
    label_list = []
    user_index = []
    seq_index_list = []
    line_count = 0
    file = open(r"../data/" + file_name + ".txt", "r", encoding="utf8")
    with open("../data/loc_index.pkl", "rb") as f:
        loc_item = pickle.load(f)
    with open("../data/idf_dict.pkl", "rb") as f3:
        idf_dict = pickle.load(f3)
    for line in file:
        line_ = line.strip().split("\t")
        if len(line_) <= 1:
            print("line:", line_)
            continue
        word_line = line_[1]
        line_tra = word_line.split(" ")
        mini_mat, seq_line, seq_index = windoms_adj(line_tra, win_size, loc_item, words_limit, seq_limit, idf_dict)
        seq_list.append(mini_mat.copy())
        seq_index_list.append(seq_index.copy())
        user_index.append(seq_line.copy())
        label_list.append(line_[0])
        line_count += 1
    label_dict = {}
    label_count = 0
    label_new = []
    for key in label_list:
        if key not in label_dict:
            label_dict[key] = label_count
            label_new.append(label_count)
            label_count += 1
        else:
            label_new.append(label_dict[key])
    print("label_uni:", len(label_dict))
    print(label_dict)
    print("seq_list:", seq_list[0])
    save_file("seq_list_" + str(win_size), seq_list)
    save_file("label", label_new)
    save_file("user_index", user_index)
    print("user_index:", user_index[5])
    print("seq_index:", seq_index_list[5])
    save_file("cnn_seq", seq_index_list)


def save_file(name, file):
    with open("../data/" + name + ".pkl", "wb") as f:
        pickle.dump(file, f)


def getGlobelAdj(line, win_size, loc_item, globel_dic):
    i = 0
    j = win_size
    length = len(line)
    if win_size > length:
        line_tra = line
        length_line = len(line_tra)
        for k in range(length_line - 1):
            for m in range(k + 1, length_line):
                if (loc_item[line_tra[k]], loc_item[line_tra[m]]) not in globel_dic:
                    globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[m]])] = 1
                    globel_dic[(loc_item[line_tra[m]], loc_item[line_tra[k]])] = 1
                else:
                    globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[m]])] += 1
                    globel_dic[(loc_item[line_tra[m]], loc_item[line_tra[k]])] += 1
            globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[k]])] = 1
        globel_dic[(loc_item[line_tra[-1]], loc_item[line_tra[-1]])] = 1
        return globel_dic

    while j <= length:
        line_tra = line[i:j]
        length_line = len(line_tra)
        for k in range(length_line - 1):
            for m in range(k + 1, length_line):
                if (loc_item[line_tra[k]], loc_item[line_tra[m]]) not in globel_dic:
                    globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[m]])] = 1
                    globel_dic[(loc_item[line_tra[m]], loc_item[line_tra[k]])] = 1
                else:
                    globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[m]])] += 1
                    globel_dic[(loc_item[line_tra[m]], loc_item[line_tra[k]])] += 1
            globel_dic[(loc_item[line_tra[k]], loc_item[line_tra[k]])] = 1
        globel_dic[(loc_item[line_tra[-1]], loc_item[line_tra[-1]])] = 1
        i += 1
        j += 1
    return globel_dic


def main(win_size, words_limit, seq_len, dataset):
    get_index(dataset + "_clear")
    for size in win_size:
        get_adj(size, words_limit, seq_len, dataset + "_clear")
    get_inputs()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
