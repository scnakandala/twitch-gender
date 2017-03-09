import powerlaw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'medium',
          'axes.titlesize': 'medium',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}

pylab.rcParams.update(params)

def plot_ccdf():
    df1 = pd.read_csv('../../data/channel_chat_lines_count.csv', header=None)
    df1.columns = ['word', 'chat_count']
    data1 = df1.chat_count.values.tolist()
    fit1 = powerlaw.Fit(data1, xmin=1.0, discrete=True)

    df2 = pd.read_csv('../../data/user_chat_counts.csv', header=None)
    df2.columns = ['user', 'chat_count']
    data2 = df2.chat_count.values.tolist()
    fit2 = powerlaw.Fit(data2, xmin=1.0, discrete=True)

    df3 = pd.read_csv('../../data/users_per_channel_counts.csv', header=None)
    df3.columns = ['user_count', 'channel_count']
    data3 = df3.channel_count.values.tolist()
    fit3 = powerlaw.Fit(data3, xmin=1.0, discrete=True)

    fig, (ax1, ax2) = plt.subplots(figsize=(4, 3), nrows=1, ncols=2,
                                   sharey=True)
#    ax1.set_title('Messages')

    fit1.plot_ccdf(color='black', linewidth=2, label='Channels', ax=ax1)
    fit2.plot_ccdf(color='black', linewidth=2, linestyle='--', ax=ax1, label='Users')

    ax1.set_xlabel('No. messages $n$')
    ax1.set_ylabel(r'$p(N \geq n)$')
    # ax1.legend(loc='best', frameon=False, fontsize='x-small')
    handles, labels = ax1.get_legend_handles_labels()

    fit3.plot_ccdf(color='black', linewidth=2, label='Channels', ax=ax2)

    ax2.set_xlabel('No. users $n$')
    ax2.legend(handles, labels, loc='best', frameon=False, fontsize='x-small')
#    ax2.set_ylabel(r'$p(U \geq u)$')
#    ax2.set_title('Users')

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    plt.tight_layout()

    fig.savefig('power_law.eps', format='eps')


if __name__ == '__main__':
    plot_ccdf()
