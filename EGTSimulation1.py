import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set()

coop = 1
defect = 0


class EGTSimulation(object):
    def __init__(self, n_lattice, T, R, P, S, n_neighbor):
        # payoff matrix
        self.pmat = {coop: {coop: R, defect: S}, defect: {coop: T, defect: P}}
        # the number of lattice
        self.n_lattice = n_lattice
        self.n_neighbor = n_neighbor
        # cooperation level
        self._coopLevel = []
        self._allact = []
        self._allPay = []

        self.runtime = 0
        self.__intialize__()

    def __intialize__(self):
        # initialize the game
        # random choose 50% to be cooperators
        n = int(self.n_lattice * self.n_lattice / 2)
        array = [1 for i in range(n)] + [0 for i in range(self.n_lattice * self.n_lattice - n)]
        array = np.array(array)
        np.random.shuffle(array)
        self._act = array.reshape([self.n_lattice, self.n_lattice])

        self._allact.append(np.copy(self._act))

        coopRate = sum(sum(self._act)) / (self.n_lattice * self.n_lattice)
        self._coopLevel.append(coopRate)

        # calculate the payoff
        self._updatePayoff()

    def __neighbor__(self, index, indey):
        # return the index of the neighbors
        t = [index, indey]
        if self.n_neighbor == 4:
            up = [index, (indey + self.n_lattice - 1) % self.n_lattice]
            down = [index, (indey + self.n_lattice + 1) % self.n_lattice]
            left = [(index + self.n_lattice - 1) % self.n_lattice, indey]
            right = [(index + self.n_lattice + 1) % self.n_lattice, indey]
            return [up, down, left, right]
        elif self.n_neighbor == 8:
            up = [index, (indey + self.n_lattice - 1) % self.n_lattice]
            down = [index, (indey + self.n_lattice + 1) % self.n_lattice]
            left = [(index + self.n_lattice - 1) % self.n_lattice, indey]
            right = [(index + self.n_lattice + 1) % self.n_lattice, indey]
            leftup = [(index + self.n_lattice - 1) % self.n_lattice, (indey + self.n_lattice - 1) % self.n_lattice]
            rightup = [(index + self.n_lattice + 1) % self.n_lattice, (indey + self.n_lattice - 1) % self.n_lattice]
            leftdown = [(index + self.n_lattice - 1) % self.n_lattice, (indey + self.n_lattice + 1) % self.n_lattice]
            rightdown = [(index + self.n_lattice + 1) % self.n_lattice, (indey + self.n_lattice + 1) % self.n_lattice]
            return [up, down, left, right, leftup, leftdown, rightup, rightdown]

    def __payoff__(self, index, indey):
        # calculate the payoff
        ngb = self.__neighbor__(index, indey)
        payoff = 0
        for pos in ngb:
            payoff += self.pmat[self._act[index, indey]][self._act[pos[0], pos[1]]]
        return payoff

    '''
    def __update__(self, index, indey):
        # update the action, finding the neighbor with the highest earnings
        ngb = self.__neighbor__(index, indey)
        # the position of  maxpay off
        posMax = [index,indey]

        pmax = self._payoff[index,indey]
        for pos in ngb:
            temp = self._payoff[pos[0], pos[1]]
            if temp>pmax:
                posMax = pos
        return self._allact[-1][posMax[0],posMax[1]]
    '''

    def _updateAct(self):
        act = np.copy(self._act)
        for i in range(self.n_lattice):
            for j in range(self.n_lattice):
                posMax = [i, j]

                pmax = self._payoff[i, j]
                ngb = self.__neighbor__(i, j)
                for pos in ngb:
                    temp = self._payoff[pos[0], pos[1]]
                    if temp > pmax:
                        posMax = pos
                        pmax = temp
                self._act[i, j] = act[posMax[0], posMax[1]]
        self._allact.append(np.copy(self._act))
        coopRate = sum(sum(self._act)) / (self.n_lattice * self.n_lattice)
        self._coopLevel.append(coopRate)

    def _updatePayoff(self):
        self.runtime += 1
        self._payoff = np.zeros([self.n_lattice, self.n_lattice])
        for i in range(self.n_lattice):
            for j in range(self.n_lattice):
                self._payoff[i, j] = self.__payoff__(i, j)
        self._allPay.append(np.copy(self._payoff))

    def _play(self):
        # update the action then calculate the payoff
        self._updateAct()
        self._updatePayoff()

    def play_ntimes(self, times):
        for i in range(times):
            self._play()

    '''
    def _draw(self,act,payoff,cooprate,times):
        payoff = payoff.astype(int)

        sns.heatmap(act, fmt="d", linewidths=.5, annot=payoff,cmap='Set1',cbar=False)

    '''

    def draw_4times(self, start):
        # draw 4 actions in one graph
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            payoff = self._allPay[start + i].astype(int)
            sns.heatmap(self._allact[start + i], fmt="d", linewidths=.5, annot=payoff, cmap='Set1', cbar=False)
            plt.title("step: " + str(start + i) + ", cooperate rate: " + str(self._coopLevel[start + i]))

    def draw0_1_10_20_50(self):
        step = 1
        for i in [0, 1,5, 10, 20, 50]:
            plt.subplot(3, 2, step)
            step += 1
            payoff = self._allPay[i].astype(int)
            sns.heatmap(self._allact[i], fmt="d", linewidths=.5, annot=payoff, cmap='Set1', cbar=False)
            plt.title("step: " + str(i) + ", cooperate rate: " + str(self._coopLevel[i]))

    def draw0_1_30_60_100_300(self):
        step = 1
        for i in [0, 1, 30, 60, 100, 300]:
            plt.subplot(3, 2, step)
            step += 1
            payoff = self._allPay[i].astype(int)
            sns.heatmap(self._allact[i], fmt="d", linewidths=.5, annot=payoff, cmap='Set1', cbar=False)
            plt.title("step: " + str(i) + ", cooperate rate: " + str(self._coopLevel[i]))

    def draw(self, step):
        payoff = self._allPay[step].astype(int)
        sns.heatmap(a._allact[step], fmt="d", linewidths=.5, annot=payoff, cmap='Set1', cbar=False)
        plt.title("step: " + str(step) + ", cooperate rate: " + str(self._coopLevel[step]))
        plt.show()


class EGTReplicator(EGTSimulation):
    def _updateAct(self):
        act = np.copy(self._act)
        R = self.pmat[coop][coop]
        T = self.pmat[defect][coop]
        S = self.pmat[coop][defect]
        P = self.pmat[defect][defect]

        max_gap = max([R, T, S, P]) - min([R, T, S, P])

        for i in range(self.n_lattice):
            for j in range(self.n_lattice):

                ngb = self.__neighbor__(i, j)
                # random choose a neighbor
                np.random.shuffle(ngb)
                pos = ngb[0]
                # the prob of
                prob = 1 / 2 * (1 + (self._payoff[pos[0], pos[1]] - self._payoff[i, j]) / (self.n_neighbor * max_gap))

                actSelf = act[i, j]
                actNgb = act[pos[0], pos[1]]

                if actSelf == actNgb:
                    # do not change
                    pass
                else:
                    # print(actSelf,actNgb)
                    # print(prob)
                    '''
                    choice_arr = np.random.choice([actNgb,actSelf],300,True,[prob,1-prob])
                    #print(sum(choice_arr))
                    np.random.shuffle(choice_arr)
                    self._act[i, j] = choice_arr[0]
                    '''
                    number = np.random.uniform(0, 1)
                    if number < prob:
                        self._act[i, j] = actNgb
        self._allact.append(np.copy(self._act))
        coopRate = sum(sum(self._act)) / (self.n_lattice * self.n_lattice)
        self._coopLevel.append(coopRate)


def randomPick(prob, aSelf, aNgb):
    length = 1000
    n = int(prob * length)
    array = [aNgb for i in range(n)] + [aSelf for i in range(length - n)]
    array = np.array(array)
    np.random.shuffle(array)
    return np.random.choice(array)

# step 0 ,1,5,10,20,50

'''

for lattice in [4, 8, 12, 20, 50]:
    for ngb in [4, 8]:
        a = EGTSimulation(lattice, 10, 7, 0, 0, ngb)
        a.play_ntimes(51)
        plt.figure(figsize=(15, 20))
        a.draw0_1_10_20_50()

        plt.savefig("./fig/" + str(lattice) + "lattice" + str(ngb) + "ngb")

        #plt.figure(figsize=(20, 20))
        #a.draw_4times(0)
        #plt.savefig("./fig/" + str(lattice) + "lattice" + str(ngb) + "ngb4run")


       # t = range(52)
        #plt.figure()
        #plt.plot(t, a._coopLevel)
        #plt.ylim(0, 1)
        #plt.savefig("./fig/" + str(lattice) + "lattice" + str(ngb) + "ngbCop")

for lattice in [4, 8, 12, 20, 50]:
    for ngb in [4, 8]:
        a = EGTReplicator(lattice, 10, 7, 0, 3, ngb)
        a.play_ntimes(51)
        plt.figure(figsize=(15, 20))
        a.draw0_1_10_20_50()

        plt.savefig("./fig2/" + str(lattice) + "lattice" + str(ngb) + "ngbRepli")

            # plt.figure(figsize=(20, 20))
            # a.draw_4times(0)
            # plt.savefig("./fig/" + str(lattice) + "lattice" + str(ngb) + "ngb4run")

           # t = range(52)
            #plt.figure()
            #plt.plot(t, a._coopLevel)
            #plt.ylim(0, 1)
            #plt.savefig("./fig2/" + str(lattice) + "lattice" + str(ngb) + "ngbRepli")
'''

# cooperational level comparison

'''
for ngb in [4, 8]:
    plt.figure()
    for lattice in [4, 8, 12, 20, 50]:
        a = EGTSimulation(lattice, 10, 7, 0, 0, ngb)
        a.play_ntimes(51)
        t = range(52)
        plt.plot(t,a._coopLevel)
        plt.ylim(0, 1)
    plt.legend(("4 lattices","8 lattices","12 lattices","20 lattices","50 lattices"))
    plt.title("cooperation level with %d neighbors(Unconditional Imitation)"%(ngb))
    plt.savefig("./fig/" + str(ngb) + "ngbCop")


for ngb in [4, 8]:
    plt.figure()
    for lattice in [4, 8, 12, 20, 50]:
        a = EGTReplicator(lattice, 10, 7, 0, 3, ngb)
        a.play_ntimes(51)
        t = range(52)
        plt.plot(t,a._coopLevel)
        plt.ylim(0, 1)
    plt.legend(("4 lattices","8 lattices","12 lattices","20 lattices","50 lattices"))
    plt.title("cooperation level with %d neighbors(Replicator Rule)"%(ngb))
    plt.savefig("./fig2/" + str(ngb) + "ngbCopRepli")

'''

'''
for lattice in [4, 8, 12, 20, 50]:
    for ngb in [4, 8]:
        cop20 = []
        for i in range(100):
            a = EGTReplicator(lattice,10,7,0,3,ngb)
            a.play_ntimes(50)
            cop20.append(a._coopLevel[50])

        plt.figure()
        plt.hist(cop20)
        plt.title("Hist of end coop level " + str(lattice) + " lattices and " + str(ngb) + " ngbs(unconditionImi) mean: %2.3f std: %2.3f" %(np.mean(cop20),np.std(cop20)))
        plt.savefig("./cophist_imi/" + str(lattice) + "lattice" + str(ngb) + "ngbCop_1")

'''

# histggram

'''

for ngb in [4, 8]:
    plt.figure(figsize=(15, 20))
    step = 0
    for lattice in [4, 8, 12, 20, 50]:

        cop20 = []
        for i in range(100):
            a = EGTSimulation(lattice,10,7,0,0,ngb)
            a.play_ntimes(50)
            cop20.append(a._coopLevel[50])
        step +=1
        plt.subplot(3,2,step)
        plt.hist(cop20)
        plt.title("Hist of end coop level " + str(lattice) + " lattices and " + str(ngb) + " ngbs(unconditionImi) mean: %2.3f std: %2.3f" %(np.mean(cop20),np.std(cop20)))
    plt.savefig("./cophist_imi/+%dngbCop_hist"%ngb)
'''

'''

for ngb in [4, 8]:
    plt.figure(figsize=(15, 20))
    step = 0
    for lattice in [4, 8, 12, 20, 50]:

        cop20 = []
        for i in range(100):
            a = EGTReplicator(lattice,10,7,0,3,ngb)
            a.play_ntimes(50)
            cop20.append(a._coopLevel[50])
        step +=1
        plt.subplot(3,2,step)
        plt.hist(cop20)
        plt.title("Hist of end coop level " + str(lattice) + " lattices and " + str(ngb) + " ngbs(Repli) mean: %2.3f std: %2.3f" %(np.mean(cop20),np.std(cop20)))
    plt.savefig("./cophist_imi/+%dngbCop_hist_repli"%ngb)

'''
'''

    cop20 = []
    for i in range(100):

        a = EGTReplicator(4,10,7,0,3,4)
        a.play_ntimes(10)
        cop20.append(a._coopLevel[10])

    plt.figure()

    plt.hist(cop20)
    plt.title("The histgram of " + " lattices and "  + "neighbors mean: %2.3f" %(np.mean(cop20)) )
    plt.show()
    '''
