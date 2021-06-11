import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_grid(ax, acceptable, good, excellent, top, 
               x_sim, y_low, y_slight, y_moderate, 
               y_met, x_acceptable, x_good, x_excellent, neg=False):
  delta = 0.001 if not neg else -0.001
  ax.fill_between(np.arange(excellent, top, delta), 0.0, 0.2, color="cornflowerblue", alpha=0.8 )

  ax.fill_between(np.arange(excellent, top, delta), 0.2, 0.4, color="cornflowerblue", alpha=0.6 )
  ax.fill_between(np.arange(good, excellent, delta), 0.0, 0.2, color="cornflowerblue", alpha=0.6 )

  ax.fill_between(np.arange(excellent, top, delta), 0.4, 0.6, color="cornflowerblue", alpha=0.4 )
  ax.fill_between(np.arange(good, excellent, delta), 0.2, 0.4, color="cornflowerblue", alpha=0.4 )
  ax.fill_between(np.arange(acceptable, good, delta), 0.0, 0.2, color="cornflowerblue", alpha=0.4 )

  ax.fill_between(np.arange(good, excellent, delta), 0.4, 0.6, color="cornflowerblue", alpha=0.2 )
  ax.fill_between(np.arange(acceptable, good, delta), 0.2, 0.4, color="cornflowerblue", alpha=0.2 )

  ax.fill_between(np.arange(acceptable, good, delta), 0.4, 0.6, color="cornflowerblue", alpha=0.1 )

  ax.axhline(y=0.2, linestyle='--', linewidth=0.8)
  ax.axhline(y=0.4, linestyle='--', linewidth=0.8)
  ax.axhline(y=0.6, linestyle='--', linewidth=0.8)

  ax.axvline(x=acceptable, linestyle='--', linewidth=0.8)
  ax.axvline(x=good, linestyle='--', linewidth=0.8)
  ax.axvline(x=excellent, linestyle='--', linewidth=0.8)

  ax.text(x_acceptable, y_met, "acceptable", rotation=90)
  ax.text(x_good, y_met, "good", rotation=90)
  ax.text(x_excellent, y_met, "excellent", rotation=90)

  ax.text(x_sim, y_low, "low", rotation=90)
  ax.text(x_sim, y_slight, "slight", rotation=90)
  ax.text(x_sim, y_moderate, "moderate", rotation=90)

def plot_points(ax, sims, metrics, instances, sample_size, texts=None, offsets_x=None, offsets_y=None, variances=None):
  if offsets_x is None:
    offsets_x = np.zeros(len(sims))
  if offsets_y is None:
    offsets_y = np.zeros(len(sims))
  if variances is None:
    variances = np.zeros(len(sims))
  scs = []
  for i in range(len(sims)):
    ax.annotate(texts[i], (metrics[i]+offsets_x[i], sims[i]+offsets_y[i]))
    width = 2*1.96*np.sqrt(variances[i]/instances[i])
    e = Ellipse(xy=[metrics[i], sims[i]], width=width, height=0.01)
    ax.add_artist(e)
    e.set_facecolor("orange")
    e.set_edgecolor("none" if instances[i]/sample_size[i] < 1.1 else 'k')
    e.set_alpha(np.min([1.0,instances[i]/sample_size[i]]))
    
    sc = ax.scatter(metrics[i]+1000, sims[i]+1000,
                alpha=np.min([1.0,instances[i]/sample_size[i]]),
                edgecolors="none" if instances[i]/sample_size[i] < 1.1 else 'k',
                c='orange'  )
    scs.append(sc)
  return scs

def step_two(sims, aucs, nbs, briers, labels, texts, instances, samples_auc, samples_nb, samples_brier,
             offsets_x_auc, offsets_y_auc, offsets_x_nb, offsets_y_nb, offsets_x_brier,
             offsets_y_brier, var_auc, var_nb, var_brier):
  fig, axs = plt.subplots(1, 3, figsize=(15,15))

  fig.subplots_adjust(wspace=0.1)

  plt.rc('font', size=12) #controls default text size
  plt.rc('axes', labelsize=15) #fontsize of the x and y labels
  plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
  plt.rc('ytick', labelsize=14) #fontsize of the y tick labels
  plt.rc('legend', fontsize=14)

  for ax in axs:
      ax.label_outer()

  axs[0].set_ylabel("Similarity")
  axs[0].set_xlabel("AUC")
  axs[0].set_xlim([0.5, 1.0])
  axs[0].set_ylim([0,1])
  plot_grid(axs[0], 0.7, 0.8, 0.9, 1.0, 0.51, 0.1, 0.3, 0.5, 0.9, 0.78, 0.88, 0.98)
  scs = plot_points(axs[0], sims, aucs, instances, samples_auc, texts, offsets_x_auc, offsets_y_auc, var_auc)
  axs[0].legend(scs, labels)

  axs[1].set_xlabel("Standardized Net Benefit")
  axs[1].set_xlim([-0.1, 1.0])
  axs[1].set_yticks([])
  axs[1].set_ylim([0,1])
  plot_grid(axs[1], 0.4, 0.6, 0.8, 1.0, -0.08, 0.1, 0.3, 0.5, 0.9, 0.555, 0.755, 0.955)
  scs = plot_points(axs[1], sims, nbs, instances, samples_nb, texts, offsets_x_nb, offsets_y_nb, var_nb)
  axs[1].legend(scs, labels)

  legend_labels = [">110% MSS","100% MSS", "50% MSS", "20% MSS"]
  scs = plot_points(axs[2], [-1,-1,-1,-1], [-1,-1,-1,-1], [110,100,50,20],[100,100,100,100])
  legend1 = plt.legend(scs, legend_labels, frameon=False, bbox_to_anchor=(-0.1,-0.05), ncol=2)
  plt.gca().add_artist(legend1)

  axs[2].set_ylabel("Similarity")
  axs[2].set_xlabel("Brier Score")
  axs[2].set_xlim([0, 0.5])
  axs[2].set_ylim([0,1])
  plot_grid(axs[2], 0.25, 0.15, 0.08, 0.0, 0.49, 0.1, 0.3, 0.5, 0.9, 0.17, 0.10, 0.02, neg=True)
  scs = plot_points(axs[2], sims, briers, instances, samples_brier, texts, offsets_x_brier, offsets_y_brier, var_brier)
  axs[2].legend(scs, labels)
  axs[2].invert_xaxis()
  axs[2].yaxis.set_label_position("right")
  axs[2].yaxis.tick_right()

  fig.savefig('step2.png', bbox_inches='tight', dpi=300)