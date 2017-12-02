from EGT.EGTSimulation1 import *

plt.figure(figsize=(20, 40))
step = 0
meancop = []
for TRratio in [0.8,1,1.2,1.4,1.43,1.47,1.5,1.7,1.8,2]:


    cop = []
    for i in range(50):
        a = EGTSimulation(50,TRratio*7,7,0,0,8)
        a.play_ntimes(50)
        cop.append(a._coopLevel[50])
    step +=1
    plt.subplot(5, 2, step)
    plt.hist(cop)
    plt.title("T/R= %f mean: %2.3f, std: %2.3f"%(TRratio,np.mean(cop),np.std(cop)))
    meancop.append(np.mean(cop))
plt.savefig("egt.png")

plt.figure()
plt.plot([0.8,1,1.2,1.4,1.43,1.47,1.5,1.7,1.8,2],meancop)
plt.xlabel("T/R ratio")
plt.ylabel("mean cooperate level")
plt.savefig("mean.png")