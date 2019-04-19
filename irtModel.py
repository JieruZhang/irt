import math
import numpy as np

'''
irt 模型
'''


class irtModel(object):

    def __init__(self) -> object:
        self.a = 1.0
        self.b = 0

    def likelihood(self, thetas):
        return 1 / 1 + math.exp(- self.a * thetas + self.a * self.b)

    def logLikelihood(self, thetas, ys):
        z = np.exp(self.a * thetas - self.a * self.b)
        return np.dot(ys, np.log(z)) - np.sum(np.log(1 + z))

    def derivB(self, thetas, ys):
        z = np.exp(self.a * thetas - self.a * self.b)
        return - np.dot(self.a, z) * (np.sum(ys / z - 1 / (1 + z)))

    def derivTheta(self, thetas, ys):
        z = np.exp(self.a * thetas - self.a * self.b)
        return np.multiply(ys / z - 1 / (1 + z), self.a * z)


if __name__ == '__main__':
    irtm = irtModel()
