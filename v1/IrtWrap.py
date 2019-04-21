import numpy as np

from Irt2PL import Irt2PL
from EAPIrt2PLModel import EAPIrt2PLModel


def predictParams(score):
    res = Irt2PL(scores=score, max_iter=150, tol=0.00001).em()
    return res


def predictThetaPerPerson(score, slop, threshold):
    thetas = 1
    z = Irt2PL.z(slop, threshold, thetas)
    p = Irt2PL.p(z)
    eap = EAPIrt2PLModel(score, slop, threshold)
    return eap.res


def predictThetaAndRes(scores, slops, thresholds):
    thetas = []
    p_vals = []
    slops = slops.tolist()
    thresholds = thresholds.tolist()
    for i in range(len(slops)):
        temp_theta = []
        temp_p_val = []
        for score in scores:
            slop = np.array([slops[i]])
            threshold = np.array([thresholds[i]])
            theta = predictThetaPerPerson(score, slop, threshold)
            z = np.exp(slops[i] * theta + thresholds[i])
            p_val = z / (1 + z)
            temp_theta.append(predictThetaPerPerson(score, slop, threshold))
            temp_p_val.append(p_val)
        thetas.append(temp_theta)
        p_vals.append(temp_p_val)
    return thetas, p_vals


def prediction():
    score = np.loadtxt("data/1_traindb_old", delimiter='\t')
    res = predictParams(score)
    slop, threshold = res[0], res[1]
    thetas, p_vals = predictThetaAndRes(score, slop, threshold)
    p_vals = np.array(p_vals)
    m, n = p_vals.shape
    p_vals = p_vals.T
    a = p_vals[0][0]
    with open("data/1_python_prediction", 'w') as f:
        for i in range(n):
            for j in range(m):
                f.write(str(p_vals[i][j]))
                if j == m - 1:
                    f.write('\n')
                else:
                    f.write('\t')


'''
计算prediction和真实得分之间的mse和diff
'''


def getDataFromFile(file1, file2):
    predictions = {}
    ys = {}
    with open(file1) as f:
        for line in f.readlines():
            line = [float(item) for item in line.strip().split('\t')]
            for i in range(len(line)):
                if i in ys.keys():
                    ys[i].append(line[i])
                else:
                    ys[i] = []
    with open(file2) as f:
        for line in f.readlines():
            line = [float(item) for item in line.strip().split('\t')]
            for i in range(len(line)):
                if i in predictions.keys():
                    predictions[i].append(line[i])
                else:
                    predictions[i] = []
    return ys, predictions


def evaluation(y, prediction):
    y_arr = np.array(y)
    prediction_arr = np.array(prediction)
    prediction = [int(item > 0.5) for item in prediction]
    mse = np.sum((y_arr - prediction_arr) * (y_arr - prediction_arr)) / len(y)
    diff_abs_sum = np.sum(np.abs(y_arr - prediction_arr)) / len(y)
    diff_sum = np.sum(y_arr - prediction_arr) / len(y)
    num = 0
    for i in range(len(y)):
        if y[i] == prediction[i]:
            num += 1
    acc = num / len(y)
    return mse, diff_abs_sum, diff_sum, acc


if __name__ == '__main__':

    #预测
    prediction()

    #统计结果
    ys, predictions = getDataFromFile("data/1_testdb_old", "data/1_python_prediction")
    mses = []
    diff_abs_sums = []
    diff_sums = []
    accs = []
    for i in range(len(ys.keys())):
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
