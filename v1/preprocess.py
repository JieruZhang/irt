import re
import operator
import numpy as np

"""
处理原始数据
"""

train_file = './data/scores-04-09'
test_file = './data/scores-04-11'
outPath = './data/'


def getKeyIds(type2Keys):
    keyIds = dict()
    for skillType in type2Keys.keys():
        keyIds[skillType] = set(type2Keys[skillType].keys())
    return keyIds


'''

返回两天用户，知识点有交集的部分
'''


def getkey2Users(type2Keys1, type2Keys2):
    type2key2Users2Scores1 = {}
    type2key2Users2Scores2 = {}
    for skillType in type2Keys1.keys():
        key2Users2Scores1 = {}
        key2Users2Scores2 = {}
        flag1 = False
        flag2 = False
        for key in type2Keys1[skillType].keys():
            if key in type2Keys2[skillType].keys():
                users1 = set(type2Keys1[skillType][key]["userId2Scores"].keys())
                users2 = set(type2Keys2[skillType][key]["userId2Scores"].keys())
                sharedUsers = users1.intersection(users2)
                user2Scores1 = {}
                user2Scores2 = {}
                for userId in type2Keys1[skillType][key]["userId2Scores"].keys():
                    if userId in sharedUsers:
                        user2Scores1[userId] = type2Keys1[skillType][key]["userId2Scores"][userId]
                        flag1 = True
                for userId in type2Keys2[skillType][key]["userId2Scores"].keys():
                    if userId in sharedUsers:
                        user2Scores2[userId] = type2Keys2[skillType][key]["userId2Scores"][userId]
                        flag2 = True
                if flag1:
                    key2Users2Scores1[key] = user2Scores1
                if flag2:
                    key2Users2Scores2[key] = user2Scores2
                flag1 = False
                flag2 = False

        # 对每个key按照用户量由大到小排序
        sorted1 = sorted(key2Users2Scores1.items(), key=lambda x: len(x[1]), reverse=True)
        sorted2 = sorted(key2Users2Scores2.items(), key=lambda x: len(x[1]), reverse=True)
        # 转成dic
        key2Users2Scores1 = {}
        key2Users2Scores2 = {}
        for item in sorted1:
            key2Users2Scores1[item[0]] = item[1]
        for item in sorted2:
            key2Users2Scores2[item[0]] = item[1]

        type2key2Users2Scores1[skillType] = key2Users2Scores1
        type2key2Users2Scores2[skillType] = key2Users2Scores2

    return type2key2Users2Scores1, type2key2Users2Scores2


def extract(file1, file2, outPath):

    with open(file1, 'r') as f:
        type2KeysTrain = eval(f.read())

    with open(file2, 'r') as f:
        type2KeysTest = eval(f.read())

    trainDb, testDb = getkey2Users(type2KeysTrain, type2KeysTest)

    print("for debug")

    #############
    f = open(outPath + '/' + '1_db', 'w')
    users = trainDb['1']['1259'].keys()
    for user in users:
        for key in ['1259', '1261', '1263']:
            scores = trainDb['1'][key][user] + testDb['1'][key][user]
            if key == '1263':
                f.write(str(scores) + '\n')
            else:
                f.write(str(scores) + '\t')
    f.close()

    ###################
    with open(outPath + '/' + '1_db', 'r') as f:
        lines = f.readlines()
        size = len(lines)
        f = open(outPath + '/' + '1_traindb', 'w')
        #读取训练集，多纪录当做多个用户
        for i in range(int(size/2)):
            line = [eval(item) for item in lines[i].split('\t')]
            num = min([len(item) for item in line])
            for j in range(num):
                f.write(str(int(line[0][j] == 3)) + '\t' + str(int(line[1][j] == 3)) + '\t' + str(int(line[2][j]==3)) + '\n')
        f.close()
        #读取测试集，前几次纪录的均值作为能力，最后一次的作为label


        f1 = open(outPath + '/' + '1_testdb', 'w')
        f2 = open(outPath + '/' + '1_valdb', 'w')

        skills = [[] for i in range(3)]
        for i in range(int(size/2), size):
            line = [eval(item) for item in lines[i].split('\t')]
            skills[0].append(np.mean(line[0][:-1]))
            skills[1].append(np.mean(line[1][:-1]))
            skills[2].append(np.mean(line[2][:-1]))
            string1 = str(int(line[0][-1]==3)) + '\t' + str(int(line[1][-1]==3)) + '\t' + str(int(line[2][-1]==3)) + '\n'
            string2 = str(line[0][:-1]) + '\t' + str(line[1][:-1]) + '\t' + str(line[2][:-1]) + '\n'
            f1.write(string1)
            f2.write(string2)
        f1.close()
        f2.close()

    # for type in ['1']:
    #     f = open(outPath + '/' + type + '_traindb_old', 'w')
    #     for key in trainDb[type]['1259'].keys():
    #         for id in ['1259', '1261', '1263']:
    #             score = int(trainDb[type][id][key].count(3)/len(trainDb[type][id][key]) > 0.5)
    #             if id == "1263":
    #                 f.write(str(score) + '\n')
    #             else:
    #                 f.write(str(score) + '\t')
    #     f.close()
    #
    # for type in ['1']:
    #     f = open(outPath + '/' + type + '_testdb_old', 'w')
    #     for key in testDb[type]['1259'].keys():
    #         for id in ['1259', '1261', '1263']:
    #             score = int(testDb[type][id][key].count(3)/len(testDb[type][id][key]) > 0.5)
    #             if id == "1263":
    #                 f.write(str(score) + '\n')
    #             else:
    #                 f.write(str(score) + '\t')
    #     f.close()
    return




if __name__ == '__main__':
    extract(train_file, test_file, outPath)


