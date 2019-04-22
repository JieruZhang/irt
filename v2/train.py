import os
import time
from v2.Irm import Irm
import numpy as np


def getData(file_list):
    """
    累计一个文件列表里所有的用户得分信息
    :param file_list:
    :return:
    """
    type2key2user2scores = {}

    for file in file_list:
        with open(file, 'r') as f:
            rawData = eval(f.read())
            for skillType in rawData.keys():
                if skillType in type2key2user2scores.keys():
                    key2user2scores = type2key2user2scores[skillType]
                else:
                    key2user2scores = {}
                for key in rawData[skillType].keys():
                    if key in key2user2scores.keys():
                        user2scores = key2user2scores[key]
                    else:
                        user2scores = {}
                    for user in rawData[skillType][key]["userId2Scores"].keys():
                        if user in user2scores.keys():
                            user2scores[user] += rawData[skillType][key]["userId2Scores"][user]
                        else:
                            user2scores[user] = rawData[skillType][key]["userId2Scores"][user]
                    key2user2scores[key] = user2scores
                type2key2user2scores[skillType] = key2user2scores
    return type2key2user2scores


def getSharedKey2Users(dic1, dic2):
    """
    得到两组数据里面交集的key和user, 筛选掉只有一次做题纪录的case
    :param dic1:
    :param dic2:
    :return:
    """

    shared_type2keys = {}
    for skillType in dic1.keys():
        if skillType not in dic2.keys():
            continue
        else:
            keys1 = set(dic1[skillType].keys())
            keys2 = set(dic2[skillType].keys())
            sharedKeys = keys1.intersection(keys2)
            if len(sharedKeys) == 0:
                continue
            else:
                shared_type2keys[skillType] = sharedKeys

    shared_type2keys2users = {}
    for skillType in dic1.keys():
        if skillType not in dic2.keys():
            continue
        else:
            shared_keys2users = {}
            for key in shared_type2keys[skillType]:
                users1 = set(dic1[skillType][key].keys())
                users2 = set(dic2[skillType][key].keys())
                shared_users = users1.intersection(users2)
                if len(shared_users) == 0:
                    continue
                else:
                    shared_keys2users[key] = shared_users
            shared_type2keys2users[skillType] = shared_keys2users

    for skillType in shared_type2keys2users.keys():
        sorted1 = sorted(shared_type2keys2users[skillType].items(), key=lambda x: len(x[1]), reverse=True)
        shared_keys2users = {}
        for item in sorted1:
            shared_keys2users[item[0]] = item[1]
        shared_type2keys2users[skillType] = shared_keys2users

    shared_type2keys2users2score = {}
    for skillType in shared_type2keys2users.keys():
        shared_keys2users2scores = {}
        for key in shared_type2keys2users[skillType].keys():
            shared_users2scores = {}
            for user in shared_type2keys2users[skillType][key]:
                if len(dic1[skillType][key][user]) < 3 or len(dic2[skillType][key][user]) < 3:
                    continue
                else:
                    ys_train = [int(item == 3) for item in dic1[skillType][key][user]]
                    ys_test = [int(item == 3) for item in dic2[skillType][key][user]]
                    shared_users2scores[user] = [ys_train, ys_test]
            if len(shared_users2scores.keys()) != 0:
                shared_keys2users2scores[key] = shared_users2scores
        shared_type2keys2users2score[skillType] = shared_keys2users2scores
    return shared_type2keys2users2score



def getPreparedDataForOneKey(user2scores):
    user2scores_train = {}
    user2scores_test = {}
    user2theta = {}
    total_ys_train = []
    total_ys_test = []
    for user in user2scores.keys():
        user2scores_train[user] = user2scores[user][0]
        user2scores_test[user] = user2scores[user][1]
        user2theta[user] = 0
        total_ys_train += user2scores[user][0]
        total_ys_test += user2scores[user][1]
    return user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test


def train(user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test, iters, tolerance):
    # train
    start = time.time()
    irm = Irm()
    irm.init_b(total_ys_train)
    iteration = 1
    total_error = 0
    while iteration <= iters:
        user2theta = irm.update_theta(user2scores_train, user2theta)
        irm.update_b(user2scores_train, user2theta)
        for user in user2scores_train.keys():
            ys = user2scores_train[user]
            total_error += np.abs(irm.p(user2theta[user]) - ys.count(1) / len(ys))
        total_error /= len(user2theta.keys())
        if iteration == iters:
            print("No outer convergence.")
        if total_error < tolerance:
            print("Outer convergence at outer iteration: ", iteration)
            break
        iteration += 1
    end = time.time()
    print("Training time: ", end - start)
    print("Trained b: ", irm.b)

    # test
    # 若不更新模型，预测结果
    irm_origin = Irm()
    irm_origin.init_b(total_ys_train)
    print("None trained b: ", irm_origin.b)
    user_predict_prob_origin = [irm_origin.p(0) for _ in range(len(user2scores_test.keys()))]

    # 若训练模型，预测结果
    user_predict_prob = []
    user_true_prob = []
    for user in user2scores_test.keys():
        ys = user2scores_test[user]
        user_true_prob.append(ys.count(1) / len(ys))
        user_predict_prob.append(irm.p(user2theta[user]))

    new_diff_abs = 0
    old_diff_abs = 0
    for i in range(len(user_true_prob)):
        new_diff_abs += np.abs(user_predict_prob[i] - user_true_prob[i])
        old_diff_abs += np.abs(user_predict_prob_origin[i] - user_true_prob[i])
    new_diff_abs /= len(user_true_prob)
    old_diff_abs /= len(user_true_prob)

    return new_diff_abs, old_diff_abs

################################################################################################

file = "/Users/zhangjieru/Documents/conan/irt/data/scores-04-"
prepared_file = "/Users/zhangjieru/Documents/conan/irt/data/prepared_data"
file_list_train = []
file_list_test = []
for item in ['01', '02', '03', '04', '05', '08', '09', '10', '11', '12']:
    file_list_train.append(file + item)
for item in ['15', "16", '17', "18", '19']:
    file_list_test.append(file + item)


if __name__ == '__main__':
    if os.path.exists(prepared_file):
        with open(prepared_file, 'r') as f:
            shared_type2keys2users2score = eval(f.read())
    else:
        type2key2user2scores_train = getData(file_list_train)
        type2key2user2scores_test = getData(file_list_test)
        shared_type2keys2users2score = getSharedKey2Users(type2key2user2scores_train, type2key2user2scores_test)
        with open(prepared_file, 'w') as f:
            f.write(str(shared_type2keys2users2score))

    diff_abs_old_total = []
    diff_abs_new_total = []
    # 获取训练集，测试集
    start_total = time.time()
    for skillType in list(shared_type2keys2users2score.keys())[1:]:
        for key in list(shared_type2keys2users2score[skillType].keys())[:5]:
            print("start training for keyword:" + skillType + '_' + key)
            user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test = getPreparedDataForOneKey(
                shared_type2keys2users2score[skillType][key])
            new_diff_abs, old_diff_abs = train(user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test, iters=300, tolerance=1e-5)
            print("new_diff_abs: ", new_diff_abs)
            print("old_diff_abs: ", old_diff_abs)
            diff_abs_new_total.append(new_diff_abs)
            diff_abs_old_total.append(old_diff_abs)
    end_total = time.time()
    print("Total time consumption: ", end_total-start_total)
    print("diff_abs_new_total: ", diff_abs_new_total)
    print("diff_abs_old_total: ", diff_abs_old_total)

