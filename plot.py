import numpy as np
import matplotlib.pyplot as plt
import util
import seaborn as sns
from matplotlib.colors import ListedColormap
from util import eprint
from pdb import set_trace

labels = {'baseline_fc_plots': 'Supervised -> Supervised',
          'None_fc_plots': 'Supervised (Baseline)', 
          'ladder_fc_plots': 'Semi-Supervised -> Supervised',
          'ladder_pre_fc_plots': 'Pre-trained Supervised'}

def analyze_perturbations():
    baseline = util.load_file('baseline_fc_plots', 'baseline_aps')
    ladder = util.load_file('ladder_fc_plots', 'ladder_aps')

    for (i, c) in enumerate(['src', 'target']):
        for l in range(7):
            ladder[c][l] = 1.0 / (ladder[c][l]**2)
            baseline[c][l] = 1.0 / (baseline[c][l]**2)
        ladder[c][7] = i
        baseline[c][7] = i

    for l in range(7):
        ladder_denom = ladder['src'][l] + ladder['target'][l]
        baseline_denom = baseline['src'][l] + baseline['target'][l]
        ladder['src'][l] = ladder['src'][l] / ladder_denom
        ladder['target'][l] = ladder['target'][l] / ladder_denom
        baseline['src'][l] = baseline['src'][l] / baseline_denom
        baseline['target'][l] = baseline['target'][l] / baseline_denom

    data = 5 * np.ones((16, 16))

    set_trace()

    eprint(ladder)
    
    for (i, c) in enumerate(['src', 'target']):
        for l in range(1, 8): 
            x_start = 2*(8 - l) + 1 
            y_start = 9*i + 2
            data[x_start, y_start:y_start + 3] = ladder[c][l] * np.ones((1, 3))

    light_color = [sns.light_palette((210, 90, 60), input="husl").as_hex()[0]]
    cmap = ListedColormap(sns.color_palette("Blues", 10).as_hex() + ['#000000'])
    clipped_cmap = ListedColormap(sns.color_palette("Blues", 10).as_hex())
    cbar_kws = { 'ticks': np.arange(0, 1.2, 0.2)} 
    ax = sns.heatmap(data, vmin=0, vmax=1.2, cmap=cmap, xticklabels=False, yticklabels=False, cbar_kws=cbar_kws)
    plt.xlabel('Source                                    Target')
    plt.savefig('ladder_aps.png')

    plt.clf()

    data = 5 * np.ones((16, 16))

    eprint(baseline)
    
    for (i, c) in enumerate(['src', 'target']):
        for l in range(1, 8): 
            x_start = 2*(8 - l) + 1 
            y_start = 9*i + 2
            data[x_start, y_start:y_start + 3] = baseline[c][l] * np.ones((1, 3))

    light_color = [sns.light_palette((210, 90, 60), input="husl").as_hex()[0]]
    cmap = ListedColormap(sns.color_palette("Blues", 10).as_hex() + ['#000000'])
    clipped_cmap = ListedColormap(sns.color_palette("Blues", 10).as_hex())
    cbar_kws = { 'ticks': np.arange(0, 1.2, 0.2)} 
    ax = sns.heatmap(data, vmin=0, vmax=1.2, cmap=cmap, xticklabels=False, yticklabels=False, cbar_kws=cbar_kws)
    plt.xlabel('Source                                    Target')
    plt.savefig('baseline_aps.png')


def plot_accuracy(folders):
    for (dir_name, file_name) in folders:
        accs = util.load_file(dir_name, file_name) 
        plt.plot(accs, label=labels[dir_name])
    plt.xlabel('Number of epochs')
    plt.ylabel('SVHN Test Accuracy')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc=4)
    #plt.show()
    plt.savefig('fc_accs.png')

def plot_loss(folders):
    for (dir_name, file_name) in folders:
        loss = util.load_file(dir_name, file_name) 
        plt.plot(loss, label=labels[dir_name])
    plt.xlabel('Number of epochs')
    plt.ylabel('Training Loss')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend()
    plt.show()
    #plt.savefig('fc_loss.png')



def main():
    """
    folders = [('baseline_fc_plots', 'all_accs75'), ('None_fc_plots', 'all_accs'), \
               ('ladder_fc_plots', 'all_accs'), ('ladder_pre_fc_plots', 'all_accs')]
    plot_accuracy(folders)

    folders = [('baseline_fc_plots', 'all_losses75'), ('None_fc_plots', 'all_losses'), \
               ('ladder_fc_plots', 'all_losses'), ('ladder_pre_fc_plots', 'all_losses54')]
    plot_loss(folders)
    """
    analyze_perturbations()

main()
