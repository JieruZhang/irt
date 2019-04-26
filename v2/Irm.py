import numpy as np


class Irm(object):
    def __init__(self, a=1, b=0, c=0, model='b'):
        self.b = b
        self.a = a
        self.c = c
        self.model = model
        self.threshold = 1e-5
        self.iters = 50
        self.lr = 0.02

    def z(self, theta):
        return np.exp(self.a * (self.b - theta))

    def p(self, theta):
        return (1 / (1 + self.z(theta))) * (1 - self.c) + self.c

    def init_b(self, ys):
        """
        假设所有用户在该知识点上的能力均为0，各个题目的区分度均为1， 初始化b
        :param ys:
        :return:
        """
        prob = ys.count(1) / len(ys)
        if prob == self.c or prob == 1:
            return self.b
        else:
            self.b = np.log((1 - self.c) / (prob - self.c) - 1) / self.a
            return self.b

    def update_theta(self, user2scores, user2theta):
        """
        更新每个用户的能力值, 平方差损失函数
        :param user2scores: dict, key is userid, value is score list
        :param user2theta: dict, key is userid, value is theta
        :return:
        """
        ##################内层单轮迭代－平方差损失####################
        for user in user2theta.keys():
            ys = user2scores[user]
            prob = ys.count(1) / len(ys)
            predict_prob = self.p(user2theta[user])
            z = self.z(user2theta[user])
            gradient = 2 * (predict_prob - prob) * self.a * (1 - self.c) * z / ((1 + z) * (1 + z))
            user2theta[user] -= self.lr * gradient
        return user2theta

    def update_params(self, user2scores, user2theta):
        """
        更新该知识点对应的难度b
        :param user2scores:
        :param user2theta:
        :return:
        """
        ##################内层单轮迭代－平方差损失####################
        num_users = len(user2theta.keys())
        gradient_b = 0
        gradient_a = 0

        for user in user2theta.keys():
            ys = user2scores[user]
            prob = ys.count(1)/len(ys)
            predict_prob = self.p(user2theta[user])
            z = self.z(user2theta[user])
            gradient_b += (- 2 * (predict_prob - prob) * self.a * z * (1 - self.c) / ((1 + z) * (1 + z)))
        gradient_b /= num_users
        self.b -= self.lr * gradient_b

        if self.model == 'abc' or self.model == 'ab':
            for user in user2theta.keys():
                ys = user2scores[user]
                prob = ys.count(1) / len(ys)
                predict_prob = self.p(user2theta[user])
                z = self.z(user2theta[user])
                gradient_a += (2 * (predict_prob - prob) * (user2theta[user] - self.b) * z * (1 - self.c) / ((1 + z) * (1 + z)))
            gradient_a /= num_users
            self.a -= self.lr * gradient_a

        return self.a, self.b
