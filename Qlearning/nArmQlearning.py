import numpy as np
import math

import matplotlib.pyplot as plt

trials = 10

def meanList(list):
    if len(list)==0:
        return 0
    else:
        return np.mean(list)

reward1 = [{'mu': 2.1, 'sigma': 0.9}, {'mu': 1.1, 'sigma': 0.8}, {'mu': 0.7, 'sigma': 2}, {'mu': 1.9, 'sigma': 0.9}]
reward2 = [{'mu': 2.1, 'sigma': 0.9*2}, {'mu': 1.1, 'sigma': 0.8*2}, {'mu': 0.7, 'sigma': 2*2}, {'mu': 1.9, 'sigma': 0.9*2}]


class NArmQlearning(object):
    def __init__(self, musigma):

        # there are 4 actions
        self._musigma = musigma
        # intialize the Q value
        self.qStarSum = [0,0,0,0]
        self.actionTimes=[0,0,0,0]
        self.qStarAvg = [[0], [0], [0], [0]]
        # the actions selected in each run
        self.rewardSum = 0
        self.rewardAvg = []
        self.action = []
        self.step = 0


    def _play_epsilon(self, epsilon):
        self.step += 1
        reward_action = []
        for i in range(4):
            reward_action.append(self.qStarAvg[i][-1])

        maxIndex = np.argmax(reward_action)

        randomAction = np.random.randint(0, 4)

        randomNum = np.random.uniform(0, 1)
        if randomNum < epsilon:
            actionChosen = randomAction
        else:
            actionChosen = maxIndex

        # update the q/action/qstar/sum
        self.action.append(actionChosen)
        self.actionTimes[actionChosen] +=1
        reward = np.random.normal(self._musigma[actionChosen]['mu'], self._musigma[actionChosen]['sigma'])
        self.rewardSum +=  reward
        self.rewardAvg.append(self.rewardSum/self.step)

        self.qStarSum[actionChosen] += reward
        for i in range(4):
            if i == actionChosen:
                self.qStarAvg[i].append(self.qStarSum[i]/self.actionTimes[i])
            else:
                self.qStarAvg[i].append(self.qStarAvg[i][-1])



    def _playtau(self, tau):
        self.step += 1
        reward_action = []
        for i in range(4):
            reward_action.append(self.qStarAvg[i][-1])

        eqt = []
        for i in range(4):
            eqt.append(np.exp(reward_action[i] / tau))
        prob = []
        for i in range(4):
            prob.append(eqt[i] / sum(eqt))

        # choosing action
        randomNum = np.random.uniform(0, 1)
        if randomNum<prob[0]:
            actionChosen = 0
        elif randomNum < sum(prob[:2]):
            actionChosen =1
        elif randomNum <sum(prob[:3]):
            actionChosen = 2
        else:
            actionChosen = 3

        self.action.append(actionChosen)
        self.actionTimes[actionChosen] += 1
        reward = np.random.normal(self._musigma[actionChosen]['mu'], self._musigma[actionChosen]['sigma'])
        self.rewardSum += reward
        self.rewardAvg.append(self.rewardSum / self.step)

        self.qStarSum[actionChosen] += reward
        for i in range(4):
            if i == actionChosen:
                self.qStarAvg[i].append(self.qStarSum[i] / self.actionTimes[i])
            else:
                self.qStarAvg[i].append(self.qStarAvg[i][-1])

    def _playrandom(self):
        self.step += 1

        randomNum = np.random.uniform(0,1)
        if randomNum<0.25:
            actionChosen = 0
        elif randomNum <0.5:
            actionChosen = 1
        elif randomNum <0.75:
            actionChosen = 2
        else:
            actionChosen = 3


        self.action.append(actionChosen)
        self.actionTimes[actionChosen] += 1
        reward = np.random.normal(self._musigma[actionChosen]['mu'], self._musigma[actionChosen]['sigma'])
        self.rewardSum += reward
        self.rewardAvg.append(self.rewardSum / self.step)

        self.qStarSum[actionChosen] += reward
        for i in range(4):
            if i == actionChosen:
                self.qStarAvg[i].append(self.qStarSum[i] / self.actionTimes[i])
            else:
                self.qStarAvg[i].append(self.qStarAvg[i][-1])



    def playepsilon0(self):
        for i in range(1000):
            self._play_epsilon(0)

    def playepsilon1(self):
        for i in range(1000):
            self._play_epsilon(0.01)

    def playepsilon2(self):
        for i in range(1000):
            self._play_epsilon(0.1)

    def playepsilonT(self):
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

    def playrandom(self):
        for i in range(1000):
            self._playrandom()




def draw_q(list):
    t = range(len(list))
    plt.ylim([0, 3])
    plt.plot(t, list)


agents=[]
for i in range(6):
    temp = []
    #play 10 times
    for j in  range(trials):
        temp.append(NArmQlearning(reward1))
    agents.append(temp)

# play 10 times and then avg the results
for i in range(10):
    agents[0][i].playepsilon0()
    agents[1][i].playepsilon1()
    agents[2][i].playepsilon2()
    agents[3][i].playtau1()
    agents[4][i].playtau2()
    agents[5][i].playrandom()

# avg the results
rewardAvg = []
rewardStd = []
for i in range(6):
    tavg=[]
    tstd=[]
    for j in range(1000):
        t = [agents[i][k].rewardAvg[j] for k in range(trials)]
        avg = np.mean(t)
        tavg.append(avg)
        tstd.append(np.std(t))

    rewardAvg.append(tavg)
    rewardStd.append(tstd)


plt.figure()

for i in range(6):
    draw_q(rewardAvg[i])


plt.legend(["epsilon0",'epsilon0.01','episilon0.1','tau1','tau0.1','random'])
plt.title("average reward over time for 6 algorithms(run each algorithm %d times)"%trials)
plt.xlabel("step")
plt.ylabel("reward")
plt.savefig("./fig/reward_all.png")

plt.figure()
for i in range(6):
    draw_q(rewardStd[i])

plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.title("reward std over time for 6 algorithms(run each algorithm %d times)" % trials)
plt.xlabel("step")
plt.ylabel("reward")
plt.savefig("./fig/reawrd_std.png")


qstarAvg = []
qstarStd = []

for i in range(6):
    agAvg = []
    agStd = []
    for action in range(4):
        tAvg = []
        tStd = []
        for step in range(1000):
            t = [agents[i][k].qStarAvg[action][step] for k in range(10)]
            tAvg.append(np.mean(t))
            tStd.append(np.std(t))
        agAvg.append(tAvg)
        agStd.append(tStd)
    qstarAvg.append(agAvg)
    qstarStd.append(agStd)

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarAvg[j][i])
    plt.title("the average reward of action %d" %i)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.savefig("./fig/actionAll.png")

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarStd[j][i])
    plt.title("the average std of action %d" %i)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.savefig("./fig/actionStd.png")

plt.figure(figsize=(15,20))
title= ['(epsilon=0)','(epsilon=0.01)','(epsilon=0.1)','(tau=1)','(tau=0.1)','(random)']
for i in range(6):
    plt.subplot(3,2,i+1)
    action = [0 for i in range(1000*trials)]
    for j in range(trials):
        action[j*1000:(j+1)*1000]= agents[i][j].action[:]
    plt.hist(action)
    plt.title("histogram of actions selection (%d) trials "%trials+title[i])
plt.savefig("./fig/hist.png")


# for double std
agents=[]
for i in range(6):
    temp = []
    #play 10 times
    for j in  range(trials):
        temp.append(NArmQlearning(reward2))
    agents.append(temp)

# play 10 times and then avg the results
for i in range(10):
    agents[0][i].playepsilon0()
    agents[1][i].playepsilon1()
    agents[2][i].playepsilon2()
    agents[3][i].playtau1()
    agents[4][i].playtau2()
    agents[5][i].playrandom()

# avg the results
rewardAvg = []
rewardStd = []
for i in range(6):
    tavg=[]
    tstd=[]
    for j in range(1000):
        t = [agents[i][k].rewardAvg[j] for k in range(trials)]
        avg = np.mean(t)
        tavg.append(avg)
        tstd.append(np.std(t))

    rewardAvg.append(tavg)
    rewardStd.append(tstd)


plt.figure()

for i in range(6):
    draw_q(rewardAvg[i])


plt.legend(["epsilon0",'epsilon0.01','episilon0.1','tau1','tau0.1','random'])
plt.title("average reward over time for 6 algorithms(run each algorithm %d times)"%trials)
plt.xlabel("step")
plt.ylabel("reward")
plt.savefig("./fig2/reward_all2.png")

plt.figure()
for i in range(6):
    draw_q(rewardStd[i])

plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.xlabel("step")
plt.ylabel("reward")
plt.title("reward std over time for 6 algorithms(run each algorithms %d times)" % trials)
plt.savefig("./fig2/reawrd_std2.png")


qstarAvg = []
qstarStd = []

for i in range(6):
    agAvg = []
    agStd = []
    for action in range(4):
        tAvg = []
        tStd = []
        for step in range(1000):
            t = [agents[i][k].qStarAvg[action][step] for k in range(10)]
            tAvg.append(np.mean(t))
            tStd.append(np.std(t))
        agAvg.append(tAvg)
        agStd.append(tStd)
    qstarAvg.append(agAvg)
    qstarStd.append(agStd)

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarAvg[j][i])
    plt.title("the average reward of action %d" %i)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.savefig("./fig2/actionAll2.png")

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarStd[j][i])
    plt.title("the average std of action %d" %i)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend(["epsilon0", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'random'])
plt.savefig("./fig2/actionStd2.png")

plt.figure(figsize=(15,20))
title= ['(epsilon=0)','(epsilon=0.01)','(epsilon=0.1)','(tau=1)','(tau=0.1)','(random)']
for i in range(6):
    plt.subplot(3,2,i+1)
    action = [0 for i in range(1000*trials)]
    for j in range(trials):
        action[j*1000:(j+1)*1000]= agents[i][j].action[:]
    plt.hist(action)
    plt.title("histogram of actions selection (%d) trials "%trials+title[i])
    plt.xlabel("step")
    plt.ylabel("reward")
plt.savefig("./fig2/hist2.png")


# change parameters

agents=[]
for i in range(6):
    temp = []
    #play 10 times
    for j in  range(trials):
        temp.append(NArmQlearning(reward1))
    agents.append(temp)

# play 10 times and then avg the results
for i in range(10):
    agents[0][i].playepsilonT()
    agents[1][i].playepsilon1()
    agents[2][i].playepsilon2()
    agents[3][i].playtau1()
    agents[4][i].playtau2()
    agents[5][i].playtauT()

# avg the results
rewardAvg = []
rewardStd = []
for i in range(6):
    tavg=[]
    tstd=[]
    for j in range(1000):
        t = [agents[i][k].rewardAvg[j] for k in range(trials)]
        avg = np.mean(t)
        tavg.append(avg)
        tstd.append(np.std(t))

    rewardAvg.append(tavg)
    rewardStd.append(tstd)


plt.figure()

for i in range(6):
    draw_q(rewardAvg[i])


plt.legend(["time varing epsilon",'epsilon0.01','episilon0.1','tau1','tau0.1','time varing tau'])
plt.title("average reward over time for 6 algorithms(run each algorithms %d times)"%trials)
plt.xlabel("step")
plt.ylabel("reward")
plt.savefig("./fig3/reward_all3.png")

plt.figure()
for i in range(6):
    draw_q(rewardStd[i])

plt.legend(["time varing epsilon",'epsilon0.01','episilon0.1','tau1','tau0.1','time varing tau'])
plt.title("reward std over time for 6 algorithms(run each algorithms %d times)" % trials)
plt.xlabel("step")
plt.ylabel("reward")
plt.savefig("./fig3/reawrd_std3.png")


qstarAvg = []
qstarStd = []

for i in range(6):
    agAvg = []
    agStd = []
    for action in range(4):
        tAvg = []
        tStd = []
        for step in range(1000):
            t = [agents[i][k].qStarAvg[action][step] for k in range(10)]
            tAvg.append(np.mean(t))
            tStd.append(np.std(t))
        agAvg.append(tAvg)
        agStd.append(tStd)
    qstarAvg.append(agAvg)
    qstarStd.append(agStd)

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarAvg[j][i])
    plt.title("the average reward of action %d" %i)
    plt.legend(["time varing epsilon", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'time varing tau'])
    plt.xlabel("step")
    plt.ylabel("reward")
plt.savefig("./fig3/actionAll3.png")

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(6):
        draw_q(qstarStd[j][i])
    plt.title("the average std of action %d" %i)
    plt.legend(["time varing epsilon", 'epsilon0.01', 'episilon0.1', 'tau1', 'tau0.1', 'time varing tau'])
    plt.xlabel("step")
    plt.ylabel("reward")
plt.savefig("./fig3/actionStd3.png")

plt.figure(figsize=(15,20))
plt.legend(["time varing epsilon",'epsilon0.01','episilon0.1','tau1','tau0.1','time varing tau'])
for i in range(6):
    plt.subplot(3,2,i+1)
    action = [0 for i in range(1000*trials)]
    for j in range(trials):
        action[j*1000:(j+1)*1000]= agents[i][j].action[:]
    plt.hist(action)
    plt.title("histogram of actions selection (%d) trials "%trials+title[i])
plt.savefig("./fig3/hist3.png")

