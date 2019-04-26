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
    得到两组数据里面交集的key和user
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
                # 过滤掉做题训练集和测试集上，对于每个知识点，每个用户做题次数少于三次的纪录
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
    '''
    对于每一个知识点，处理数据
    :param user2scores:
    :return:
    '''
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


def train(user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test, iters, tolerance, model_names):
    '''
    对于每个知识点，训练对应的各模型
    :param user2scores_train:
    :param user2scores_test:
    :param user2theta:
    :param total_ys_train:
    :param total_ys_test:
    :param iters:
    :param tolerance:
    :return:
    '''

    models = []
    if "b" in model_names:
        irm_b = Irm(model='b')
        irm_b.init_b(total_ys_train)
        models.append(irm_b)
    if "ab" in model_names:
        irm_ab = Irm(model='ab')
        irm_ab.init_b(total_ys_train)
        models.append(irm_ab)
    if "bc" in model_names:
        irm_bc = Irm(model='bc', c=0.1)
        irm_bc.init_b(total_ys_train)
        models.append(irm_bc)
    if "abc" in model_names:
        irm_abc = Irm(model='abc', c=0.1)
        irm_abc.init_b(total_ys_train)
        models.append(irm_abc)

    res_dic = {}

    for irm in models:

        total_error = 0
        origin_b = irm.b
        iteration = 1

        # 训练每个模型之前，初始化用户能力
        for user in user2theta.keys():
            user2theta[user] = 0

        start = time.time()
        while iteration <= iters:
            user2theta = irm.update_theta(user2scores_train, user2theta)
            irm.update_params(user2scores_train, user2theta)
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
        print("Training time for model " + irm.model + " is: ", end - start)

        # 用户训练集上准确率
        user_old_prob = []
        for user in user2scores_train.keys():
            ys = user2scores_train[user]
            user_old_prob.append(ys.count(1) / len(ys))

        # 若训练模型，预测结果
        user_predict_prob = []
        user_true_prob = []
        train_test_ratios = []
        for user in user2scores_test.keys():
            ys = user2scores_test[user]
            user_true_prob.append(ys.count(1) / len(ys))
            user_predict_prob.append(irm.p(user2theta[user]))
            train_test_ratios.append(len(user2scores_train[user]) / len(user2scores_test[user]))

        acc_diff = sum(np.abs(np.array(user_old_prob) - np.array(user_true_prob))) / len(user_true_prob)
        old_diff = sum(np.abs(np.array([irm.p(0) for _ in range(len(user_true_prob))]) - np.array(user_true_prob))) / len(user_true_prob)
        new_diff = sum(np.abs(np.array(user_predict_prob) - np.array(user_true_prob))) / len(user_true_prob)

        res_dic[irm.model] = {}
        res_dic[irm.model]["diff_new"] = new_diff
        res_dic[irm.model]["diff_old"] = old_diff
        res_dic[irm.model]["diff_acc"] = acc_diff
        res_dic[irm.model]["b"] = irm.b
        res_dic[irm.model]["a"] = irm.a
        res_dic[irm.model]["origin_b"] = origin_b
        res_dic[irm.model]["user_true_prob"] = np.mean(user_true_prob)
        res_dic[irm.model]["user_old_prob"] = np.mean(user_old_prob)
        res_dic[irm.model]["train_test_ratio"] = np.mean(train_test_ratios)
        res_dic[irm.model]["user2theta"] = user2theta

    return res_dic


################################################################################################
bath_path = os.path.abspath('..')
file = bath_path + "/data/scores-04-"
prepared_file = bath_path + "/data/prepared_data_0401_0419_10:5_3_3"
save_path = bath_path + "/data/params"
file_list_train = []
file_list_test = []
for item in ['01', '02', '03', '04', '05', '08', '09', '10', '11', '12']:
    file_list_train.append(file + item)
for item in ['15', "16", '17', "18", '19']:
    file_list_test.append(file + item)

if __name__ == '__main__':
    # 加载处理好的数据，或者创建该数据
    if os.path.exists(prepared_file):
        with open(prepared_file, 'r') as f:
            shared_type2keys2users2score = eval(f.read())
    else:
        type2key2user2scores_train = getData(file_list_train)
        type2key2user2scores_test = getData(file_list_test)
        shared_type2keys2users2score = getSharedKey2Users(type2key2user2scores_train, type2key2user2scores_test)
        with open(prepared_file, 'w') as f:
            f.write(str(shared_type2keys2users2score))

    #开始训练每个知识点
    res = {}
    avg_res = {}
    params = {}
    for skillType in list(shared_type2keys2users2score.keys())[1:]:
        for key in list(shared_type2keys2users2score[skillType].keys())[:25]:
            user_size = len(shared_type2keys2users2score[skillType][key].keys())
            if user_size >= 400:
                print("start training for keyword:" + skillType + '_' + key)
                # 处理数据
                user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test = getPreparedDataForOneKey(shared_type2keys2users2score[skillType][key])
                model_names = ["b", "ab", "bc", "abc"]
                #开始训练
                res_dic = train(user2scores_train, user2scores_test, user2theta, total_ys_train, total_ys_test, iters=500, tolerance=1e-2, model_names=model_names)
                params[skillType + '_' + key] = res_dic

                for model in ["b", 'ab', 'bc', 'abc']:
                    if model in res.keys():
                        res[model]["diff_acc"].append(res_dic[model]["diff_acc"])
                        res[model]["diff_new"].append(res_dic[model]["diff_new"])
                        res[model]["diff_old"].append(res_dic[model]["diff_old"])
                    else:
                        res[model] = {}
                        res[model]["diff_acc"] = [res_dic[model]["diff_acc"]]
                        res[model]["diff_new"] = [res_dic[model]["diff_new"]]
                        res[model]["diff_old"] = [res_dic[model]["diff_old"]]
    for model in ["b", 'ab', 'bc', 'abc']:
        avg_res[model] = [np.mean(res[model]["diff_acc"]), np.mean(res[model]["diff_new"]),
                          np.mean(res[model]["diff_old"])]

    # 保存训练参数
    with open(save_path, 'w') as f:
        f.write(str(params) + "\n")
        f.write(str(res) + "\n")
        f.write(str(avg_res))

    print(avg_res)
