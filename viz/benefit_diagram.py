import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#baseline rappresenta una grandezza baseline (ad esempio, accuratezza senza AI)
#difference è la differenza tra due accuratezze (ad esempio, accuratezza con AI - accuratezza senza AI)
def benefit_diagram(baseline, difference, filename="benefit-diagram"):
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=baseline,
                    y=difference, color="black", alpha=0.5)

    vals = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.fill_between(vals, 1 - vals, alpha=0.3, color="blue" )
    plt.fill_between(vals, - vals, alpha=0.3, color="red" )

    plt.axhline(np.mean(difference), color="black")
    plt.axhline(np.mean(difference) + 1.96*np.std(difference)/np.sqrt(len(difference)), color="black", alpha=0.25)
    plt.axhline(np.mean(difference) - 1.96*np.std(difference)/np.sqrt(len(difference)), color="black", alpha=0.25)
    plt.xlim(0,1)
    plt.ylim(-1,1)
    plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")
