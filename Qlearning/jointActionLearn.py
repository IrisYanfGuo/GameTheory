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
                    reward.append(np.max(self.qStar[i][j]))
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
a1 = JointActionLearn(0.2,0.2,0.2)
b = JointActionLearn(4,0.1,0.1)
b1= JointActionLearn(4,0.1,0.1)
c = JointActionLearn(0.1,0.1,4)
c1= JointActionLearn(0.1,0.1,4)

def movAvg(alist,window):
    # use an odd integer
    step = int((window-1)/2)
    start = step
    end = len(alist)-step
    res = []
    for i in range(0,start):
        res.append(np.mean(alist[0:i+1]))
    for i in range(start,end):
        res.append(np.mean(alist[i-step:i+step+1]))
    for i in range(end,len(alist)-end):
        res.append(alist[i])
    return res


a.playBolt()
a1.playOptiBolt()
b.playBolt()
b1.playOptiBolt()
c.playBolt()
c1.playOptiBolt()

aq = movAvg(a.q,51)
a1q = movAvg(a1.q,51)

bq = movAvg(b.q,51)
b1q = movAvg(b1.q,51)
cq = movAvg(c.q,51)
c1q = movAvg(c1.q,51)

plt.figure()
plt.plot(aq)
plt.plot(a1q)
plt.title("Reward(window=51) sigma0=sigma1=sigma= 0.2")
plt.xlabel("step")
plt.ylabel("reward")
plt.legend(["simple Boltzmann","(optimistic boltzmann"])
plt.savefig("./JAL/jalr1.png")

plt.figure()
plt.plot(bq)
plt.plot(b1q)
plt.title("Reward(window=51) sigma0=4,sigma1=sigma= 0.1")
plt.xlabel("step")
plt.ylabel("reward")
plt.legend(["simple Boltzmann","(optimistic boltzmann"])
plt.savefig("./JAL/jalr2.png")

plt.figure()
plt.plot(cq)
plt.plot(c1q)
plt.title("Reward(window=51) sigma1=4,sigma0=sigma= 0.1")
plt.xlabel("step")
plt.ylabel("reward")
plt.legend(["simple Boltzmann","(optimistic boltzmann"])
plt.savefig("./JAL/jalr3.png")

plt.figure(figsize=(15,20))
plt.subplot(3,2,1)
plt.plot(a.probAList[0])
plt.plot(a.probAList[1])
plt.plot(a.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma0=sigma1=sigma2=0.2,optimistic boltzmann)")

plt.subplot(3,2,2)
plt.plot(a.probBList[0])
plt.plot(a.probBList[1])
plt.plot(a.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma0=sigma1=sigma2=0.2,optimistic boltzmann)")


plt.subplot(3,2,3)
plt.plot(b.probAList[0])
plt.plot(b.probAList[1])
plt.plot(b.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma0=4,sigma1=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,4)
plt.plot(b.probBList[0])
plt.plot(b.probBList[1])
plt.plot(b.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma0=4,sigma1=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,5)
plt.plot(c.probAList[0])
plt.plot(c.probAList[1])
plt.plot(c.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma1=4,sigma0=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,6)
plt.plot(c.probBList[0])
plt.plot(c.probBList[1])
plt.plot(c.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma1=4,sigma0=sigma= 0.1,optimistic boltzmann)")

plt.savefig("./JAL/prob1.png")

plt.figure(figsize=(15,20))
plt.subplot(3,2,1)
plt.plot(a1.probAList[0])
plt.plot(a1.probAList[1])
plt.plot(a1.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma0=sigma1=sigma2=0.2,optimistic boltzmann)")

plt.subplot(3,2,2)
plt.plot(a1.probBList[0])
plt.plot(a1.probBList[1])
plt.plot(a1.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma0=sigma1=sigma2=0.2,optimistic boltzmann)")


plt.subplot(3,2,3)
plt.plot(b1.probAList[0])
plt.plot(b1.probAList[1])
plt.plot(b1.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma0=4,sigma1=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,4)
plt.plot(b1.probBList[0])
plt.plot(b1.probBList[1])
plt.plot(b1.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma0=4,sigma1=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,5)
plt.plot(c1.probAList[0])
plt.plot(c1.probAList[1])
plt.plot(c1.probAList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['A-action0','A-action1','A-action2'])
plt.title("Probability of A's actions\n(sigma1=4,sigma0=sigma= 0.1,optimistic boltzmann)")

plt.subplot(3,2,6)
plt.plot(c.probBList[0])
plt.plot(c.probBList[1])
plt.plot(c.probBList[2])
plt.xlabel("step")
plt.ylabel("Probability")
plt.legend(['B-action0','B-action1','B-action2'])
plt.title("Probability of B's actions\n(sigma1=4,sigma0=sigma= 0.1,optimistic boltzmann)")

plt.savefig("./JAL/prob2.png")
