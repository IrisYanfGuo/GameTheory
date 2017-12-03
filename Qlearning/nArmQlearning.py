import numpy as np
import math

import matplotlib.pyplot as plt

reward1 = [{'mu': 2.1, 'sigma': 0.9}, {'mu': 1.1, 'sigma': 0.8}, {'mu': 0.7, 'sigma': 2}, {'mu': 1.9, 'sigma': 0.9}]


class NArmQlearning(object):
    def __init__(self, musigma):

        # there are 4 actions
        self._musigma = musigma
        # intialize the Q value

        self.qStar = [[], [], [], []]
        # the actions selected in each run
        self.reward = []
        self.action = []
        self.step = 0

    def _reward(self, mu, sigma):
        return np.random.normal(mu, sigma, 1)

    def _play_epsilon(self, epsilon):
        self.step += 1
        reward_action = []
        for i in range(4):
            if len(self.qStar[i]) > 0:
                reward_action.append(np.mean(self.qStar[i]))
            else:
                reward_action.append(0)

        maxIndex = np.argmax(reward_action)
        maxReward = np.random.normal(self._musigma[maxIndex]['mu'], self._musigma[maxIndex]['sigma'])

        randomAction = np.random.randint(0, 4)
        randomReward = np.random.normal(self._musigma[randomAction]['mu'], self._musigma[randomAction]['sigma'])

        randomNum = np.random.uniform(0, 1)
        if randomNum < epsilon:
            # if smaller than episilon, random select an action
            self.reward.append(randomReward)
            self.action.append(randomAction)
            for i in range(4):
                if i != randomAction:
                    self.qStar[i].append(reward_action[i])
                else:

                    self.qStar[randomAction].append(randomReward)
        else:
            self.reward.append(maxReward)
            self.action.append(maxIndex)
            for i in range(4):
                if i != maxIndex:
                    self.qStar[i].append(reward_action[i])
                else:

                    self.qStar[maxIndex].append(maxReward)

    def _playtau(self, tau):
        self.step += 1
        reward_action = []
        for i in range(4):
            if len(self.qStar[i]) > 0:
                reward_action.append(np.mean(self.qStar[i]))
            else:
                reward_action.append(0)

        eqt = []
        for i in range(4):
            eqt.append(np.exp(reward_action[i] / tau))
        prob = []
        for i in range(4):
            prob.append(eqt[i] / sum(eqt))

        # choosing action
        randomNum = np.random.uniform(0, 1)
        if randomNum < prob[0]:
            # choosing action 0
            self.action.append(0)
            reward = np.random.normal(self._musigma[0]['mu'], self._musigma[0]['sigma'])
            self.reward.append(reward)
            self.qStar[0].append(reward)
            for i in [1, 2, 3]:
                self.qStar[i].append(reward_action[i])
        elif randomNum < sum(prob[0:2]):
            # choosing action 1
            self.action.append(1)
            reward = np.random.normal(self._musigma[1]['mu'], self._musigma[1]['sigma'])
            self.reward.append(reward)
            self.qStar[1].append(reward)
            for i in [0, 2, 3]:
                self.qStar[i].append(reward_action[i])
        elif randomNum < sum(prob[0:3]):
            # choosing action 3
            self.action.append(2)
            reward = np.random.normal(self._musigma[2]['mu'], self._musigma[2]['sigma'])
            self.reward.append(reward)
            self.qStar[2].append(reward)
            for i in [0, 1, 3]:
                self.qStar[i].append(reward_action[i])
        else:
            self.action.append(3)
            reward = np.random.normal(self._musigma[3]['mu'], self._musigma[3]['sigma'])
            self.reward.append(reward)
            self.qStar[3].append(reward)
            for i in [0, 1, 2]:
                self.qStar[i].append(reward_action[i])

    def playepsilon0(self):
        for i in range(1000):
            self._play_epsilon(0)

    def playepsilon1(self):
        for i in range(1000):
            self._play_epsilon(0.01)

    def playepsilon2(self):
        for i in range(1000):
            self._play_epsilon(0.1)

    def playepisilonT(self):
        for i in range(1000):
            epi = 1/np.sqrt(i+1)
            self._play_epsilon(epi)

    def playtau1(self):
        for i in range(1000):
            self._playtau(1)

    def playtau2(self):
        for i in range(1000):
            self._playtau(0.1)

    def playtauT(self):
        for i in range(1000):
            tau = 4*(1000-i)/1000
            self._playtau(tau)




def draw_q(list):
    a = []
    t = range(len(list))
    for i in t:
        a.append(np.mean(list[:i + 1]))

    plt.ylim([0, 3])
    plt.plot(t, a)


'''
a = NArmQlearning(reward1)
a.playepsilon1()
print(np.mean(a.reward))
plt.figure()
draw_q(a.reward)
draw_q(a.qStar[0])
draw_q(a.qStar[1])
draw_q(a.qStar[2])
draw_q(a.qStar[3])
plt.legend(['q','qStar1','qStar2','qStar3','qStar4'])
plt.show()

'''
a = NArmQlearning(reward1)
a.playtau1()
plt.figure()
draw_q(a.reward)
draw_q(a.qStar[0])
draw_q(a.qStar[1])
draw_q(a.qStar[2])
draw_q(a.qStar[3])
plt.legend(['q','qStar1','qStar2','qStar3','qStar4'])
plt.show()

plt.figure()
plt.hist(a.action)
plt.show()