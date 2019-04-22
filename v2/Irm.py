import numpy as np


class Irm(object):
    def __init__(self, b=0):
        self.b = b
        self.threshold = 1e-5
        self.iters = 50
        self.lr = 0.02

    def z(self, theta):
        return np.exp(self.b - theta)

    def p(self, theta):
        return 1 / (1 + self.z(theta))

    def init_b(self, ys):
        """
        假设所有用户在该知识点上的能力均为0，初始化b
        :param ys:
        :return:
        """
        prob = ys.count(1) / len(ys)
        if prob == 0 or prob == 1:
            return self.b
        else:
            self.b = np.log(1 / prob - 1)
            return self.b

    def update_theta(self, user2scores, user2theta):
        """
        更新每个用户的能力值, 平方差损失函数
        :param user2scores: dict, key is userid, value is score list
        :param user2theta: dict, key is userid, value is theta
        :return:
        """
        # for user in user2theta.keys():
        #     ys = user2scores[user]
        #     prob = ys.count(1) / len(ys)
        #     iteration = 1
        #     while iteration <= self.iters:
        #         predict_prob = self.p(user2theta[user])
        #         #if iteration == self.iters:
        #             #print("theta, no convergence, max iteration.")
        #         if np.abs(predict_prob - prob) < self.threshold:
        #             #print("theta, convergence at iteration: ", iteration)
        #             break
        #         z = self.z(user2theta[user])
        #         gradient = 2 * (predict_prob - prob) * z / ((1 + z) * (1 + z))
        #         user2theta[user] -= self.lr * gradient
        #         iteration += 1
        #####################################
        for user in user2theta.keys():
            ys = user2scores[user]
            prob = ys.count(1) / len(ys)
            if prob == 0 or prob == 1:
                continue
            else:
                user2theta[user] = self.b - np.log(1 / prob - 1)
        ######################################
        # for user in user2theta.keys():
        #     ys = user2scores[user]
        #     prob = ys.count(1) / len(ys)
        #     predict_prob = self.p(user2theta[user])
        #     z = self.z(user2theta[user])
        #     gradient = 2 * (predict_prob - prob) * z / ((1 + z) * (1 + z))
        #     user2theta[user] -= self.lr * gradient

        return user2theta

    def update_b(self, user2scores, user2theta):
        """
        更新该知识点对应的难度b
        :param user2scores:
        :param user2theta:
        :return:
        """
        # num_users = len(user2theta.keys())
        # iteration = 1
        # while iteration <= self.iters:
        #     gradient = 0
        #     error = 0
        #     for user in user2theta.keys():
        #         ys = user2scores[user]
        #         prob = ys.count(1)/len(ys)
        #         predict_prob = self.p(user2theta[user])
        #         error += np.abs(predict_prob - prob)
        #         z = self.z(user2theta[user])
        #         gradient += (- 2 * (predict_prob - prob) * z / ((1 + z) * (1 + z)))
        #     error /= num_users
        #     gradient /= num_users
        #     #if iteration == self.iters:
        #         #print("b, no convergence, max iteration.", iteration)
        #     if error < self.threshold:
        #         #print("b, convergence at iteration: ", iteration)
        #         break
        #     self.b -= self.lr * gradient
        #     iteration += 1
        #################################
        bs = []
        for user in user2theta.keys():
            ys = user2scores[user]
            prob = ys.count(1) / len(ys)
            if prob == 0 or prob == 1:
                bs.append(self.b)
            else:
                bs.append(user2theta[user] + np.log(1 / prob - 1))
        self.b = sum(bs)/len(bs)
        #############################
        # num_users = len(user2theta.keys())
        # gradient = 0
        # error = 0
        # for user in user2theta.keys():
        #     ys = user2scores[user]
        #     prob = ys.count(1)/len(ys)
        #     predict_prob = self.p(user2theta[user])
        #     error += np.abs(predict_prob - prob)
        #     z = self.z(user2theta[user])
        #     gradient += (- 2 * (predict_prob - prob) * z / ((1 + z) * (1 + z)))
        # gradient /= num_users
        # self.b -= self.lr * gradient

        return self.b
