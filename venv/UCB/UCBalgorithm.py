import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D





def UCB(vals, rounds, c):
    av = np.mean(vals)
    exploration = len(vals)
    upper_conf_bound = av+ c*np.sqrt( (2*np.log(rounds+1))/(exploration) )
    return upper_conf_bound

def choose_bandit(bandit_list, rounds, c):
    biggest = bandit_list[0]
    if not biggest.vals:
        return biggest
    biggest_ucb = UCB(biggest.vals, rounds, c)
    for bandit in bandit_list:
        if not bandit.vals:
            return bandit
        temp_ucb = UCB(bandit.vals, rounds, c)
        if(temp_ucb > biggest_ucb):
            biggest_ucb = temp_ucb
            biggest = bandit
    return biggest

def best_bandit(bandit_list):
    best = bandit_list[0]
    best_prob = bandit_list[0].probability
    for bandit in bandit_list:
        if bandit.probability > best_prob:
            best_prob = bandit.probability
            best = bandit
    return best

def play(bandits, samples, c):
    total_profit = 0
    profits = {}
    progress = []
    samples.sort()
    best = best_bandit(bandits)
    fig, ax = plt.subplots(len(samples),3)

    for val in samples:
        index = samples.index(val)
        if index == 0:
            start = 0
        else:
            start = samples[index-1]
        for i in range(start, val):
            bandit = choose_bandit(bandits, i, c)
            profit = np.random.binomial(1, bandit.probability)
            bandit.vals.append( profit )
            progress.append(profit)
            total_profit+= profit
            profits[i] = total_profit

        if len(samples) ==1:
            graph_singular_row(bandits, val, profits, best, progress, c, ax)
        else:
            graph_mult_rows(bandits, val, profits, best, progress, c, ax, index)

    plt.show()


def error_bars(bandit, samples, c):

    print(bandit)

    y = np.mean(bandit.vals)
    e = UCB(bandit.vals, samples, c)-np.mean(bandit.vals)
    return y, e


def best_outcome_experimental(best, samples):
    best_outcome = {}
    best_profit = 0

    for i in range(samples):
        best_outcome[i] = best_profit
        best_profit += np.random.binomial(1, best.probability)

    return best_outcome

def best_outcome_theoretical(best, samples):
    best_outcome = {}
    best_profit = 0

    for i in range(samples):

        best_profit += best.probability*samples
        best_outcome[i] = best_profit

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

def graph_mult_rows(bandit_list, samples, profit, best, progress, c, ax, row):


    #plot calculated probabilities and confidence bounds in leftmost graph
    bandit_num = []
    bandit_prob = []
    for bandit in bandit_list:
        # x = bandit_list.index(bandit)
        x = f'b{bandit_list.index(bandit)}'
        y, e = error_bars(bandit, samples, c)
        ax[row, 0].errorbar(x, y, e, marker = 'o')
        ax[row, 0].text(x, y, f'{len(bandit.vals)}')

        bandit_num.append(x)
        bandit_prob.append(bandit.probability)

    ax[row, 0].plot(bandit_num, bandit_prob, marker = 'x', color = 'black', linestyle = 'none', label = 'actual probability')


    # ax[row, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0, 0].legend()
    ax[0, 0].set_title("Comparison", fontname="Times New Roman", fontsize = 18, fontweight = 'bold')

    #plot profit and regret on middle graph

    sns.lineplot(data=profit, color= "blue", ax = ax[row, 1])

    best_outcome = best_outcome_experimental(best, samples)
    sns.lineplot(data =best_outcome, color = "green", ax = ax[row, 1])

    regret = expiremental_regret_data(best_outcome, profit)
    sns.lineplot(data = regret, color= 'red', ax = ax[row, 1])

    theoretical_regret = theoretical_regret_data(profit, best)
    sns.lineplot(data = theoretical_regret, color= 'purple', ax = ax[row, 1])


    profit_legend_lines = [Line2D([0], [0], color='blue', lw=4),
                           Line2D([0], [0], color='green', lw=4),
                           Line2D([0], [0], color='red', lw=4),
                           Line2D([0], [0], color='purple', lw=4)]
    ax[0, 1].legend(profit_legend_lines, ['actual', 'optimal', 'expiremental regret', 'theoretical regret'])
    ax[0, 1].set_title("Total Profit over time (samples)", fontname="Times New Roman", fontsize = 18, fontweight = 'bold')


    #plot calculated average overtime on rightmost plot

    x, y_av = calculated_average_vals(progress)
    ax[row, 2].plot(x,y_av, label = 'sample av')

    y_bestprob = [val*0+ best.probability for val in x]
    ax[row, 2].plot(x, y_bestprob, label = 'optimal av')

    ax[row, 2].fill_between(x, y_bestprob,y_av, alpha = .3)

    ax[0, 2].set_title("Calculated Average over time (samples)", fontname="Times New Roman", fontsize = 18, fontweight = 'bold')
    ax[0, 2].legend()

def graph_singular_row(bandit_list, samples, profit, best, progress, c, ax):
    #plot calculated probabilities and confidence bounds in leftmost graph
    bandit_num = []
    bandit_prob = []
    for bandit in bandit_list:
        x = f'b{bandit_list.index(bandit)}'
        y, e = error_bars(bandit, samples, c)
        ax[0].errorbar(x, y, e, marker = 'o')
        ax[0].text(x, y, f'{len(bandit.vals)}')

        bandit_num.append(x)
        bandit_prob.append(bandit.probability)

    ax[0].plot(bandit_num, bandit_prob, marker = 'x', color = 'black', linestyle = 'none', label = 'actual probability')


    # ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].legend()
    ax[0].set_title("Comparison")

    #plot profit and regret on middle graph

    sns.lineplot(data=profit, color= "blue", ax = ax[1])

    best_outcome = best_outcome_experimental(best, samples)
    sns.lineplot(data =best_outcome, color = "green", ax = ax[1])

    regret = expiremental_regret_data(best_outcome, profit)
    sns.lineplot(data = regret, color= 'red', ax = ax[1])

    theoretical_regret = theoretical_regret_data(profit, best)
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


