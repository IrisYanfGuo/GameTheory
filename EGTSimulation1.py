import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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
        n = int(self.n_lattice*self.n_lattice/2)
        array = [1 for i in range(n)]+[0 for i in range(self.n_lattice*self.n_lattice-n)]
        array = np.array(array)
        np.random.shuffle(array)
        self._act = array.reshape([self.n_lattice,self.n_lattice])

        self._allact.append(np.copy(self._act))

        coopRate = sum(sum(self._act)) / (self.n_lattice * self.n_lattice)
        self._coopLevel.append(coopRate)

        # calculate the payoff
        self._updatePayoff()



    def __neighbor__(self, index, indey):
        # return the index of the neighbors
        t=[index,indey]
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
        for pos in ngb :
            payoff += self.pmat[self._act[index,indey]][self._act[pos[0],pos[1]]]
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
                ngb = self.__neighbor__(i,j)
                for pos in ngb:
                    temp = self._payoff[pos[0], pos[1]]
                    if temp > pmax:
                        posMax = pos
                        pmax = temp
                self._act[i, j] = act[posMax[0],posMax[1]]
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


    def play_ntimes(self,times):
        for i in range(times):
            self._play()

    '''
    def _draw(self,act,payoff,cooprate,times):
        payoff = payoff.astype(int)

        sns.heatmap(act, fmt="d", linewidths=.5, annot=payoff,cmap='Set1',cbar=False)

    '''

    def draw_4times(self,start):
        # draw 4 actions in one graph
        for i in range(4):
            plt.subplot(2,2,i+1)
            payoff = self._allPay[start+i].astype(int)
            sns.heatmap(self._allact[start+i], fmt="d", linewidths=.5, annot=payoff, cmap='Set1', cbar=False)
            plt.title("step: "+str(start+i)+", cooperate rate: "+str(self._coopLevel[start+i]))
        plt.show()


    def draw(self,step):
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

        max_gap = max([R,T,S,P]) - min([R,T,S,P])

        for i in range(self.n_lattice):
            for j in range(self.n_lattice):

                ngb = self.__neighbor__(i,j)
                # random choose a neighbor
                pos = np.random.shuffle(ngb)[0]
                # the prob of
                prob= 1/2*(1+(self._payoff[pos[0],pos[1]]-self._payoff[i,j])/(self.n_neighbor*max_gap))

                print(prob)
                print(sum(prob))






a = EGTReplicator(4,10,7,0,0,4)
a._play()






