import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np

def trust_diagram(RAIRs, RSRs, RAIDs, RSTs, labels, confidence=None, filename="trust-diagram"):
  fig, axs = plt.subplots(ncols=2, sharex=False, sharey=False, figsize=(10,5))
  plt.subplots_adjust(wspace=0.05, hspace=0)

  palette = sns.color_palette("colorblind", 2)

  for i in range(len(RAIDs)):
    if confidence == 'ellipse':
      el = Ellipse(xy = (RAIDs[i].mean(), RSRs[i].mean()),
          width=2*1.96*RAIDs[i].std()/np.sqrt(len(RAIDs[i])),
          height=2*1.96*RSRs[i].std()/np.sqrt(len(RSRs[i])),
          label = labels[i], edgecolor=palette[i], facecolor=palette[i], alpha=0.6)
      
      axs[0].add_patch(el)
    elif confidence == 'error':
      axs[0].errorbar(RAIDs[i].mean(), RSRs[i].mean(),
                      xerr=1.96*RAIDs[i].std()/np.sqrt(len(RAIDs[i])),
                      yerr=1.96*RSRs[i].std()/np.sqrt(len(RSRs[i])),
                      label=labels[i], color=palette[i])
    else:
      axs[0].scatter(RAIDs[i].mean(), RSRs[i].mean(), label=labels[i], color=palette[i])

  axs[0].axhline(y = 0.5, color='k', ls='--', alpha=0.5)
  axs[0].axvline(x = 0.5, color='k', ls='--', alpha=0.5)
  axs[0].fill_between([-0.02,1.02], -0.02, 0.5, color='red', alpha=0.2, label="Automation Bias") #RSR Basso
  axs[0].fill_betweenx([-0.02,1.02], -0.02, 0.5, color='blue', alpha=0.2, label="Automation Complacency") #RAIR Basso

  axs[0].set_xlim(-0.02, 1.02)
  axs[0].set_ylim(-0.02, 1.02)
  axs[0].set_xlabel("Relative Beneficial Over-distrust (RBOD)")
  axs[0].set_ylabel("Relative Beneficial Distrust (RBD)")
  axs[0].legend(loc='center', bbox_to_anchor=(0.5,1.1), ncol=2)

  for i in range(len(RSTs)):
    if confidence == 'ellipse':
      el = Ellipse(xy = (RAIRs[i].mean(), RSTs[i].mean()),
          width=2*1.96*RAIRs[i].std()/np.sqrt(len(RAIRs[i])),
          height=2*1.96*RSTs[i].std()/np.sqrt(len(RSTs[i])),
          label = labels[i], edgecolor=palette[i], facecolor=palette[i], alpha=0.6)
      
      axs[1].add_patch(el)
    elif confidence == 'error':
      axs[1].errorbar(RAIRs[i].mean(), RSTs[i].mean(),
                      xerr=1.96*RAIRs[i].std()/np.sqrt(len(RAIRs[i])),
                      yerr=1.96*RSTs[i].std()/np.sqrt(len(RSTs[i])),
                      label=labels[i], color=palette[i])
    else:
      axs[1].scatter(RAIRs[i].mean(), RSTs[i].mean(), label=labels[i], color=palette[i])

  axs[1].axhline(y = 0.5, color='k', ls='--', alpha=0.5)
  axs[1].axvline(x = 0.5, color='k', ls='--', alpha=0.5)
  axs[1].fill_between([-0.02,1.02], -0.02, 0.5, color='red', alpha=0.2, label="Algorithmic Aversion") #RAID basso
  axs[1].fill_betweenx([-0.02,1.02], -0.02, 0.5, color='blue', alpha=0.2, label="Conservatism Bias") #RST basso

  axs[1].set_xlim(-0.02, 1.02)
  axs[1].set_ylim(-0.02, 1.02)
  axs[1].set_xlabel("Relative Beneficial Over-trust (RBOT)")
  axs[1].set_ylabel("Relative Beneficial Trust (RBT)")
  axs[1].yaxis.tick_right()
  axs[1].yaxis.set_label_position("right")
  axs[1].invert_xaxis()
  axs[1].legend(loc='center', bbox_to_anchor=(0.5,1.1), ncol=2)

  plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")