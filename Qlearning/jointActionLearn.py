import numpy as np
import math
import matplotlib.pyplot as plt

class JointActionLearn():
    def __init__(self,sigma0,sigma,sigma1):

        self._musigma=[[[11,sigma0],[-30,sigma],[0,sigma]],[[-30,sigma],[7,sigma1],[6,sigma]],[[0,sigma],[0,sigma],[5,sigma]]]
        self.q = []
        self.qStar=[[[],[],[]],[[],[],[]],[[],[],[]]]
        self.step = 0
        self.probA = np.zeros(3)+1/3
        self.probB = np.zeros(3)+1/3
        self.probAList = [[1/3],[1/3],[1/3]]
        self.probBList = [[1/3],[1/3],[1/3]]


    def _playBoltz(self, tau):
        self.step += 1

        # choose action for A
        randomNum = np.random.uniform(0,1)
        if randomNum < self.probA[0]:
            aaction = 0
        elif randomNum < sum(self.probA[:2]):
            aaction = 1
        else:
            aaction = 2

            # choose action for B
        randomNum = np.random.uniform(0, 1)
        if randomNum < self.probB[0]:
            baction = 0
        elif randomNum < sum(self.probB[:2]):
            baction = 1
        else:
            baction = 2

        # get the reward
        rewardnow  = np.random.normal(self._musigma[baction][aaction][0],self._musigma[baction][aaction][1])

        # update the qStar/q/action
        self.qStar[baction][aaction].append(rewardnow)
        self.q.append(rewardnow)


        # update the reward for per action
        EVB = []
        EVA = []
        for i in range(3):
            s = 0
            for j in range(3):
                if len(self.qStar[i][j])>0:
                    s += np.mean(self.qStar[i][j])*self.probA[j]
                else:
                    s +=  0

            EVB.append(s)

        for j in range(3):
            s = 0
            for i in range(3):
                if len(self.qStar[i][j]) > 0:
                    s += np.mean(self.qStar[i][j])*self.probB[i]
                else:
                    s += 0

            EVA.append(s)

        if self.step == 4999:
            print("EVA:",EVA)
            print("EVB:",EVB)

        # update the eqt
        eqtA = []
        eqtB = []
        for i in range(3):
            eqtA.append(np.exp(EVA[i]/tau))
            eqtB.append(np.exp(EVB[i]/tau))


        # update the Prob
        for i in range(3):

            self.probB[i] = eqtB[i]/sum(eqtB)
            self.probBList[i].append(eqtB[i] / sum(eqtB))

        for i in range(3):
            self.probA[i] = eqtA[i]/sum(eqtA)
            self.probAList[i].append(eqtA[i] / sum(eqtA))



    def _playOptiBoltz(self,tau):
        self.step += 1

        # choose action for A
        randomNum = np.random.uniform(0,1)
        if randomNum < self.probA[0]:
            aaction = 0
        elif randomNum < sum(self.probA[:2]):
            aaction = 1
        else:
            aaction = 2

            # choose action for B
        randomNum = np.random.uniform(0, 1)
        if randomNum < self.probB[0]:
            baction = 0
        elif randomNum < sum(self.probB[:2]):
            baction = 1
        else:
            baction = 2

        # get the reward
        rewardnow  = np.random.normal(self._musigma[baction][aaction][0],self._musigma[baction][aaction][1])

        # update the qStar/q/action
        self.qStar[baction][aaction].append(rewardnow)
        self.q.append(rewardnow)


        # update the reward for per action
        EVB = []
        EVA = []
        for i in range(3):
            # every action reward
            reward = []
            for j in range(3):
                if len(self.qStar[i][j])>0:
                    reward.append(np.mean(self.qStar[i][j]))
                else:
                    reward.append(0)

            EVB.append(max(reward))

        for j in range(3):
            # every action reward
            reward = []
            for i in range(3):
                if len(self.qStar[i][j]) > 0:
                    reward.append(np.mean(self.qStar[i][j]))
                else:
                    reward.append(0)

            EVA.append(max(reward))

        if self.step == 4999:
            print("EVA:", EVA)
            print("EVB:", EVB)

        # update the eqt
        eqtA = []
        eqtB = []
        for i in range(3):
            eqtA.append(np.exp(EVA[i] / tau))
            eqtB.append(np.exp(EVB[i] / tau))

        # update the Prob
        for i in range(3):
            self.probB[i] = eqtB[i] / sum(eqtB)
            self.probBList[i].append(eqtB[i] / sum(eqtB))

        for i in range(3):
            self.probA[i] = eqtA[i] / sum(eqtA)
            self.probAList[i].append(eqtA[i] / sum(eqtA))


    def _playFCQ(self,c):
        self.step += 1

        # choose action for A
        randomNum = np.random.uniform(0, 1)
        if randomNum < self.probA[0]:
            aaction = 0
        elif randomNum < sum(self.probA[:2]):
            aaction = 1
        else:
            aaction = 2

            # choose action for B
        randomNum = np.random.uniform(0, 1)
        if randomNum < self.probB[0]:
            baction = 0
        elif randomNum < sum(self.probB[:2]):
            baction = 1
        else:
            baction = 2

        # get the reward
        rewardnow = np.random.normal(self._musigma[baction][aaction][0], self._musigma[baction][aaction][1])
        # update the qStar/q/action
        self.qStar[baction][aaction].append(rewardnow)
        self.q.append(rewardnow)

        # update the reward for per action
        EVB = []
        EVA = []
        for i in range(3):
            # every action reward
            reward = []
            for j in range(3):
                if len(self.qStar[i][j]) > 0:
                    reward.append(np.mean(self.qStar[i][j]))
                else:
                    reward.append(0)

            EVB.append(max(reward))

        for j in range(3):
            # every action reward
            reward = []
            for i in range(3):
                if len(self.qStar[i][j]) > 0:
                    reward.append(np.mean(self.qStar[i][j]))
                else:
                    reward.append(0)

            EVA.append(max(reward))



    def playBolt(self):
        for i in range(5000):
            tau = 16*(0.999**i)
            self._playBoltz(tau)

    def playOptiBolt(self):
        for i in range(5000):
            tau = 16 * (0.999 ** i)
            self._playOptiBoltz(tau)


a = JointActionLearn(0.2,0.2,0.2)
a.playOptiBolt()
print(a.probA)
print(a.probB)
for i in range(3):
    for j in range(3):
        if len(a.qStar[i][j])>0:
            print(np.mean(a.qStar[i][j]))
        else:
            print(0)
        print(len(a.qStar[i][j]))
print(np.mean(a.q))

plt.figure()
t = range(len(a.probAList[0]))
for i in range(3):
    plt.plot(t,a.probAList[i])
plt.show()