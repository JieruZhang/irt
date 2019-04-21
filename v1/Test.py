import numpy as np

from Irt2PL import Irt2PL
from EAPIrt2PLModel import EAPIrt2PLModel


def loadFile(file):
    res = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [eval(item) for item in line.split('\t')]
            res.append(line)
    return res

def predictParams(score):
    params = Irt2PL(scores=score, max_iter=150, tol=0.00001).em()
    return params


def predictThetaPerPerson(scores, slop, threshold):
    thetas = []
    for score in scores:
        eap = EAPIrt2PLModel(score, slop, threshold)
        thetas.append(eap.res)
    return np.mean(thetas)


def predictTheta(val_file, slops, thresholds):
    thetas = []
    slops = slops.tolist()
    thresholds = thresholds.tolist()
    scores = loadFile(val_file)
    for i in range(len(slops)):
        temp_theta = []
        for score_list in scores:
            slop = np.array([slops[i]])
            threshold = np.array([thresholds[i]])
            temp_theta.append(predictThetaPerPerson(score_list[i], slop, threshold))
        thetas.append(temp_theta)
    return thetas

def predictRes(thetas, slops, thresholds):
    p_vals = [[] for i in range(len(thetas))]
    slops = slops.tolist()
    thresholds = thresholds.tolist()
    for i in range(len(thetas)):
        for j in range(len(thetas[i])):
            z = np.exp(slops[i] * thetas[i][j] + thresholds[i])
            p_val = z / (1 + z)
            p_vals[i].append(p_val)
    return p_vals



def prediction(train_file, val_file, prediction_file):
    #训练集使用用户组a训练题目参数
    score = np.loadtxt(train_file, delimiter='\t')
    res = predictParams(score)
    slop, threshold = res[0], res[1]
    #验证集计算用户组b的能力
    thetas = predictTheta(val_file, slop, threshold)
    p_vals = predictRes(thetas, slop, threshold)
    p_vals = np.array(p_vals)
    m, n = p_vals.shape
    p_vals = p_vals.T
    with open(prediction_file, 'w') as f:
        for i in range(n):
            for j in range(m):
                f.write(str(p_vals[i][j]))
                if j == m - 1:
                    f.write('\n')
                else:
                    f.write('\t')

def evaluation(y, prediction):
    y_arr = np.array(y)
    prediction_arr = np.array(prediction)
    prediction = [int(item>0.5) for item in prediction]
    mse = np.sum((y_arr - prediction_arr) * (y_arr - prediction_arr)) / len(y)
    diff_abs_sum = np.sum(np.abs(y_arr - prediction_arr)) / len(y)
    diff_sum = np.sum(y_arr - prediction_arr) / len(y)
    num = 0
    for i in range(len(y)):
        if y[i] == prediction[i]:
            num += 1
    acc = num/len(y)
    return mse, diff_abs_sum, diff_sum, acc


if __name__ == '__main__':

    train_file = "data/1_traindb"
    val_file = "data/1_valdb"
    test_file = "data/1_testdb"
    prediction_file = "data/1_python_prediction"

    #预测
    prediction(train_file, val_file, prediction_file)

    #统计结果
    ys = np.array(loadFile(test_file)).T.tolist()
    predictions = np.array(loadFile(prediction_file)).T.tolist()
    mses = []
    diff_abs_sums = []
    diff_sums = []
    accs = []
    for i in range(len(ys)):
        mse, diff_abs_sum, diff_sum, acc = evaluation(ys[i], predictions[i])
        mses.append(mse)
        diff_abs_sums.append(diff_abs_sum)
        diff_sums.append(diff_sum)
        accs.append(acc)
    print("mses: ", mses)
    print("diff_abs_sums: ", diff_abs_sums)
    print("diff_sums: ", diff_sums)
    print("accs: ", accs)

    print("mse avg: ", np.mean(mses))
    print("diff_abs_sums avg: ", np.mean(diff_abs_sums))
    print("diff_sums avg: ", np.mean(diff_sums))

    print("success!")
