import numpy as np
import math

import matplotlib.pyplot as plt

class JointActionLearn():
    def __init__(self,sigma0,sigma,sigma1):

        self._musigma=[[[11,sigma0],[-30,sigma],[0,sigma]],[[-30,sigma],[7,sigma1],[6,sigma]],[[0,sigma],[0,sigma],[5,sigma]]]
        self.q = []
        self.action = []
        self.qStar=[[[],[],[]],[[],[],[]],[[],[],[]]]
        self.step = 0


    def _playtau(self, tau):
        self.step += 1

        reward_action = []
        for i in range(3):
            tempR = []
            for j in range(3):
                if len(self.qStar[i][j]) > 0:
                    tempR.append(np.mean(self.qStar[i][j]))
                else:
                    tempR.append(np.random.normal(self._musigma[i][j][0], self._musigma[i][j][1]))
            reward_action.append(tempR)


        eqt = []
        for i in range(3):
            teqt = []
            for j in range(3):
                teqt.append(np.exp(reward_action[i][j] / tau))
            eqt.append(teqt)


        sumEqt = 0
        for i in eqt:
            sumEqt += sum(i)

        prob = []
        for i in range(3):
            tprob = []
            for j in range(3):
                tprob.append(eqt[i][j] / sumEqt)
            prob.append(tprob)

        # choosing action
        randomNum = np.random.uniform(0,1)
        if randomNum < sum(prob[0]):
            if randomNum < prob[0][0]:
                actionChose = [0,0]

            elif randomNum < sum(prob[0][:2]):
                actionChose = [0,1]

            else:
                actionChose = [0,2]
        elif randomNum < sum(prob[0]) + sum(prob[1]):
            if randomNum <sum(prob[0])+prob[1][0]:
                actionChose = [1,0]
            elif randomNum <sum(prob[0])+sum(prob[1][:2]):
                actionChose = [1,1]
            else:
                actionChose = [1,2]
        else:
            if randomNum < sum(prob[0])+sum(prob[1]) + prob[2][0]:
                actionChose = [2,0]
            elif randomNum < sum(prob[0])+sum(prob[1]) + sum(prob[2][:2]):
                actionChose = [2,1]
            else:
                actionChose = [2,2]

        self.action.append(actionChose)
        rewardNow = np.random.normal(self._musigma[actionChose[0]][actionChose[1]][0],self._musigma[actionChose[0]][actionChose[1]][1])
        self.q.append(rewardNow)

        # update the q_star
        for i in range(3):
            for j in range(3):
                if i == actionChose[0] and j == actionChose[1]:
                    self.qStar[i][j].append(rewardNow)
                else:
                    self.qStar[i][j].append(reward_action[i][j])


    def playtau1(self):
        for i in range(5000):
            self._playtau(0.95)






a = JointActionLearn(0.2,0.2,0.2)
a.playtau1()
for i in range(3):
    for j in range(3):
        print(np.mean(a.qStar[i][j]))

result = np.zeros(9)
action = []
for i in a.action:
    action.append(i[0]*3+i[1])
    result[i[0]*3+i[1]] +=1

print(result)
plt.figure()
plt.hist(action[2500:])


plt.figure()
plt.scatter(range(len(action)),action)
plt.show()







