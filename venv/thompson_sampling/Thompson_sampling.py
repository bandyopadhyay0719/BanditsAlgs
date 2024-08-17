import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from Bandit import Bandit
from scipy.stats import beta
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


def create_bandits(num_bandits):

    bandits = []
    for i in range(num_bandits):
        bandits.append(Bandit(np.random.rand()))
    return bandits


def pull_arm(bandit):
    result = np.random.binomial(1, bandit.probability)
    if result:
        bandit.success_fail[0] += 1
        bandit.vals.append(1)
    else:
        bandit.success_fail[1] += 1
        bandit.vals.append(0)
    return bool(result)

def choose_bandit(bandit_list):
    chosen = bandit_list[0]
    chosen_val = np.random.beta(chosen.success_fail[0]+1,chosen.success_fail[1]+1)
    for bandit in bandit_list:
        rounds = bandit.success_fail[0]+bandit.success_fail[1]
        if rounds ==0:
            return bandit
        val = np.random.beta(bandit.success_fail[0]+1,bandit.success_fail[1]+1)
        if val>chosen_val:
            chosen_val = val
            chosen = bandit
    return chosen

def choose_bandit_average(bandit_list):
    chosen = bandit_list[0]
    chosen_average = np.mean(bandit_list[0].vals)
    for bandit in bandit_list:
        rounds = bandit.success_fail[0]+bandit.success_fail[1]
        if rounds ==0:
            return bandit
        average = np.mean(bandit.vals)
        if average>chosen_average:
            chosen_average = average
            chosen = bandit
    return chosen
def best_bandit(bandit_list):
    best = bandit_list[0]
    best_prob = bandit_list[0].probability
    for bandit in bandit_list:
        if bandit.probability > best_prob:
            best_prob = bandit.probability
            best = bandit
    return best

def best_outcome_experimental(best, samples):
    best_outcome = {}
    best_profit = 0

    for i in range(samples):
        best_outcome[i] = best_profit
        best_profit += np.random.binomial(1, best.probability)

    return best_outcome

def expiremental_regret_data(best_outcome, profit):
    regret = {}
    for key in best_outcome:
        regret[key] = best_outcome[key] - profit[key]

    return regret

def theoretical_regret_data(profit, best):
    theoretical_regret = {}
    for key in profit:
        theoretical_regret[key] = best.probability*key - profit[key]
    return theoretical_regret

# def calculated_average_vals(progress):
#     calculated_av = {}
#     for i in range(len(progress)):
#         sliced_list = progress[0: i]
#         if not sliced_list:
#             average = 0
#         else:
#             average = np.mean(sliced_list)
#             calculated_av[i] = average
#
#     return calculated_av

def calculated_average_vals(progress):
    x = []
    y = []
    for i in range(len(progress)):
        sliced_list = progress[0: i]
        x.append(float(i))
        if not sliced_list:
            average = 0
        else:
            average = np.mean(sliced_list)
        y.append(average)

    return x, y

def graph_mult_samples(bandit_list, profits, samples, progress, row, ax):

    best = best_bandit(bandit_list)

    for bandit in bandit_list:
        total_runs = sum(bandit.success_fail)
        i = bandit_list.index(bandit)
        x = np.linspace(0, 1, 1000)
        y = beta.pdf(x, bandit.success_fail[0]+1, bandit.success_fail[1]+1)
        ax[row,0].plot(x,y, label = f'B{i}->{total_runs}')


    ax[row,0].legend()
    ax[0,0].set_title("Beta Distributions", fontname="Times New Roman", fontsize = 14, fontweight = 'bold')

    sns.lineplot(data=profits, color= "blue", ax = ax[row,1])

    best_outcome = best_outcome_experimental(best, samples)
    sns.lineplot(data=best_outcome, color= "green", ax = ax[row,1])

    regret = expiremental_regret_data(best_outcome, profits)
    sns.lineplot(data = regret, color= 'red', ax = ax[row,1])

    theoretical_regret = theoretical_regret_data(profits, best)
    sns.lineplot(data = theoretical_regret, color= 'purple', ax = ax[row,1])

    profit_legend_lines = [Line2D([0], [0], color='blue', lw=4),
                           Line2D([0], [0], color='green', lw=4),
                           Line2D([0], [0], color='red', lw=4),
                           Line2D([0], [0], color='purple', lw=4)]
    ax[0,1].legend(profit_legend_lines, ['actual', 'optimal', 'expiremental regret', 'theoretical regret'])
    ax[0,1].set_title("Total Profit over time (samples)", fontname="Times New Roman", fontsize = 14, fontweight = 'bold')

    #plot calculated average overtime on rightmost plot

    x, y_av = calculated_average_vals(progress)
    ax[row, 2].plot(x,y_av, label = 'sample av')

    y_bestprob = [val*0+ best.probability for val in x]
    ax[row, 2].plot(x, y_bestprob, label = 'optimal av')

    ax[row, 2].fill_between(x, y_bestprob,y_av, alpha = .3)

    ax[0, 2].set_title("Calculated Average over time (samples)", fontname="Times New Roman", fontsize = 14, fontweight = 'bold')
    ax[0, 2].legend()

    #plot calculated probabilities and confidence bounds in leftmost graph

    bandit_num = []
    bandit_prob = []
    for bandit in bandit_list:
        if sum(bandit.success_fail)>0:
            x = f'b{bandit_list.index(bandit)}'
            y = np.mean(bandit.success_fail[0]/(sum(bandit.success_fail)))
            ax[row, 3].plot(x, y, marker = 'o')

        bandit_num.append(x)
        bandit_prob.append(bandit.probability)

    ax[row, 3].plot(bandit_num, bandit_prob, marker = 'x', color = 'black', linestyle = 'none', label = 'actual probability')

    ax[0, 3].legend(loc = 'upper right')
    ax[0, 3].set_title("Comparison", fontname="Times New Roman", fontsize = 14, fontweight = 'bold')

def graph_singular_sample(bandit_list, profits, samples, progress):

    fig, ax = plt.subplots(1,4)
    best = best_bandit(bandit_list)

    for bandit in bandit_list:
        total_runs = sum(bandit.success_fail)
        i = bandit_list.index(bandit)
        x = np.linspace(0, 1, 1000)
        y = beta.pdf(x, bandit.success_fail[0]+1, bandit.success_fail[1]+1)
        ax[0].plot(x,y, label = f'B{i}->{total_runs}')

    ax[0].set_title("Beta Distributions")
    ax[0].legend()

    sns.lineplot(data=profits, color= "blue", ax = ax[1])

    best_outcome = best_outcome_experimental(best, samples)
    sns.lineplot(data=best_outcome, color= "green", ax = ax[1])

    regret = expiremental_regret_data(best_outcome, profits)
    sns.lineplot(data = regret, color= 'red', ax = ax[1])

    theoretical_regret = theoretical_regret_data(profits, best)
    sns.lineplot(data = theoretical_regret, color= 'purple', ax = ax[1])

    profit_legend_lines = [Line2D([0], [0], color='blue', lw=4),
                           Line2D([0], [0], color='green', lw=4),
                           Line2D([0], [0], color='red', lw=4),
                           Line2D([0], [0], color='purple', lw=4)]
    ax[1].legend(profit_legend_lines, ['actual', 'optimal', 'expiremental regret', 'theoretical regret'])
    ax[1].set_title("Total Profit over time (samples)")

    #plot calculated average overtime on rightmost plot

    x, y_av = calculated_average_vals(progress)
    ax[2].plot(x,y_av, label = 'sample av')

    y_bestprob = [val*0+ best.probability for val in x]
    ax[2].plot(x, y_bestprob, label = 'optimal av')

    ax[2].fill_between(x, y_bestprob,y_av, alpha = .3)


    ax[2].set_title("Calculated Average over time (samples)")
    ax[2].legend()

    #plot calculated probabilities and confidence bounds in leftmost graph

    bandit_num = []
    bandit_prob = []
    for bandit in bandit_list:
        if sum(bandit.success_fail)>0:
            x = f'b{bandit_list.index(bandit)}'
            y = np.mean(bandit.success_fail[0]/(sum(bandit.success_fail)))
            ax[3].plot(x, y, marker = 'o')

        bandit_num.append(x)
        bandit_prob.append(bandit.probability)

    ax[3].plot(bandit_num, bandit_prob, marker = 'x', color = 'black', linestyle = 'none', label = 'actual probability')

    ax[3].legend(loc = 'upper right')

    ax[3].set_title("Comparison")

    # plt.show()

def find_mode(bandits):
    max_length_bandit = max(bandits, key=lambda bandit: len(bandit.vals))
    print(max_length_bandit)
    a = max_length_bandit.success_fail[0] +1
    b = max_length_bandit.success_fail[1]+1

    mode = (a-1)/(a+b-2)

    y_value_at_mode = stats.beta.pdf(mode, a, b)


    return y_value_at_mode




def animation(frame, bandit, line):

    x = np.linspace(0, 1, 1000)


    sliced_vals = bandit.vals[:frame]
    success = sliced_vals.count(1)
    fails = sliced_vals.count(0)

    y = beta.pdf(x, success+1, fails+1)

    line.set_ydata(y)
    return line




def play(bandits, samples):
    # bandits = create_bandits(num_bandits)
    profits = {}
    profit = 0
    progress = []
    samples.sort()
    fig, ax = plt.subplots(len(samples),4)
    fig2, ax2 = plt.subplots()




    for sample in samples:
        index = samples.index(sample)
        if index == 0:
            start = 0
        else:
            start = samples[index-1]
        for i in range(start, sample):
            chosen = choose_bandit(bandits)
            if(pull_arm(chosen)):
                profit+=1
                progress.append(1)
            else:
                progress.append(0)

            profits[i] = profit

        if len(samples) == 1:
            graph_singular_sample(bandits, profits, sample, progress)
        else:
            graph_mult_samples(bandits, profits, sample, progress, index, ax)


    for bandit in bandits:

        i = bandits.index(bandit)

        print(f'B{i}-> {bandit}')


    x = np.linspace(0, 1, 1000)

    # Initialize lines for each bandit
    lines = []
    for i, bandit in enumerate(bandits):
        line, = ax2.plot(x, beta.pdf(x, bandit.success_fail[0] + 1, bandit.success_fail[1] + 1), label=f'Bandit {i}')
        lines.append(line)

    # Set plot limits and labels
    mode = find_mode(bandits)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, mode)
    print(mode)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density')
    ax2.legend()

    # Update function for animation
    def update(frame):
        for i, bandit in enumerate(bandits):
            # Ensure we do not exceed the length of bandit.vals
            if frame <= len(bandit.vals):
                sliced_vals = bandit.vals[:frame]
                success = sliced_vals.count(1)
                fails = sliced_vals.count(0)

                a = success + 1
                b = fails + 1
                y = beta.pdf(x, a, b)
                lines[i].set_ydata(y)
                ax2.set_title(f'Frame {frame}')


        return lines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(1, max(len(bandit.vals) for bandit in bandits) + 1), interval=1, blit=True)
    # ani.save('beta_ani.gif', writer='imagemagick', fps=10)


    # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('scatter.gif', writer=writer)

    plt.show()

def play_average(bandits, samples, exploration_prob):
    # bandits = create_bandits(num_bandits)


    profits = {}
    profit = 0
    progress = []
    samples.sort()
    fig, ax = plt.subplots(len(samples),4)
    fig2, ax2 = plt.subplots()




    for sample in samples:
        index = samples.index(sample)
        if index == 0:
            start = 0
        else:
            start = samples[index-1]
        for i in range(start, sample):

            if np.random.rand() <= exploration_prob:
                chosen = choose_bandit(bandits)
            else:
                chosen = choose_bandit_average(bandits)

            if(pull_arm(chosen)):
                profit+=1
                progress.append(1)
            else:
                progress.append(0)

            profits[i] = profit

        if len(samples) == 1:
            graph_singular_sample(bandits, profits, sample, progress)
        else:
            graph_mult_samples(bandits, profits, sample, progress, index, ax)


    for bandit in bandits:

        i = bandits.index(bandit)

        print(f'B{i}-> {bandit}')


    x = np.linspace(0, 1, 1000)

    # Initialize lines for each bandit
    lines = []
    for i, bandit in enumerate(bandits):
        line, = ax2.plot(x, beta.pdf(x, bandit.success_fail[0] + 1, bandit.success_fail[1] + 1), label=f'Bandit {i}')
        lines.append(line)

    # Set plot limits and labels
    mode = find_mode(bandits)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, mode)
    print(mode)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density')
    ax2.legend()

    # Update function for animation
    def update(frame):
        for i, bandit in enumerate(bandits):
            # Ensure we do not exceed the length of bandit.vals
            if frame <= len(bandit.vals):
                sliced_vals = bandit.vals[:frame]
                success = sliced_vals.count(1)
                fails = sliced_vals.count(0)

                a = success + 1
                b = fails + 1
                y = beta.pdf(x, a, b)
                lines[i].set_ydata(y)
                ax2.set_title(f'Frame {frame}')


        return lines


    plt.show()




