# Python 2.7
import os, csv

# Statistics libraries
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn import linear_model

# Plotting libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


# --------------------------------------------------- Functions ------------------------------------------------------ #


def count_turns(data):
    turn_str = ''
    for row in data:
        if isinstance(row, str):
            turn_str = turn_str + ' | ' + row
    turns = [int(s) for s in turn_str.split() if s.isdigit()]
    c1 = turns[0::2]
    c2 = turns[1::2]
    p1c1, p2c1 = c1.count(0), c1.count(1)
    p1c2, p2c2 = c2.count(0), c2.count(1)

    return turns, p1c1, p2c1, p1c2, p2c2


def find_correct(turns, loss):
    p1_answers = [val for idx, val in enumerate(loss) if turns[idx] == 0]
    p2_answers = [val for idx, val in enumerate(loss) if turns[idx] == 1]
    p1_correct = float(len(p1_answers) - sum(p1_answers))
    p2_correct = float(len(p2_answers) - sum(p2_answers))

    return p1_correct, p2_correct


def contextwise_accuracy(turns, loss, output='acc'):
    c1_turns, c2_turns = turns[0::2], turns[1::2]
    c1_loss, c2_loss = loss[0::2], loss[1::2]

    c1p1_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 0]
    c1p2_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 1]
    c2p1_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 0]
    c2p2_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 1]

    c1p1_acc = float(len(c1p1_ans) - sum(c1p1_ans)) / len(c1p1_ans)
    c1p2_acc = float(len(c1p2_ans) - sum(c1p2_ans)) / len(c1p2_ans)
    c2p1_acc = float(len(c2p1_ans) - sum(c2p1_ans)) / len(c2p1_ans)
    c2p2_acc = float(len(c2p2_ans) - sum(c2p2_ans)) / len(c2p2_ans)

    if output == 'acc':
        return c1p1_acc, c1p2_acc, c2p1_acc, c2p2_acc
    else:
        return len(c1p1_ans) - sum(c1p1_ans), len(c1p2_ans) - sum(c1p2_ans), len(c2p1_ans) - sum(c2p1_ans), len(
            c2p2_ans) - sum(c2p2_ans)


def total_disparity(base_turns, base_loss, new_turns, new_loss):
    c1_turns, c2_turns = base_turns[0::2], base_turns[1::2]
    c1_loss, c2_loss = base_loss[0::2], base_loss[1::2]
    b_c1p1_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 0]
    b_c1p2_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 1]
    b_c2p1_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 0]
    b_c2p2_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 1]

    c1_turns, c2_turns = new_turns[0::2], new_turns[1::2]
    c1_loss, c2_loss = new_loss[0::2], new_loss[1::2]
    n_c1p1_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 0]
    n_c1p2_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 1]
    n_c2p1_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 0]
    n_c2p2_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 1]

    c1p1_ans = b_c1p1_ans + n_c1p1_ans
    c1p2_ans = b_c1p2_ans + n_c1p2_ans
    c2p1_ans = b_c2p1_ans + n_c2p1_ans
    c2p2_ans = b_c2p2_ans + n_c2p2_ans

    c1p1_acc = float(len(c1p1_ans) - sum(c1p1_ans)) / len(c1p1_ans)
    c1p2_acc = float(len(c1p2_ans) - sum(c1p2_ans)) / len(c1p2_ans)
    c2p1_acc = float(len(c2p1_ans) - sum(c2p1_ans)) / len(c2p1_ans)
    c2p2_acc = float(len(c2p2_ans) - sum(c2p2_ans)) / len(c2p2_ans)

    disparity_p = np.mean([abs((c1p1_acc - c2p1_acc)), abs((c1p2_acc - c2p2_acc))])
    disparity_c = np.mean([abs((c1p1_acc - c1p2_acc)), abs((c2p1_acc - c2p2_acc))])

    return disparity_p, disparity_c


def disparity(turns, loss):
    c1_turns, c2_turns = turns[0::2], turns[1::2]
    c1_loss, c2_loss = loss[0::2], loss[1::2]
    c1p1_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 0]
    c1p2_ans = [val for idx, val in enumerate(c1_loss) if c1_turns[idx] == 1]
    c2p1_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 0]
    c2p2_ans = [val for idx, val in enumerate(c2_loss) if c2_turns[idx] == 1]

    c1p1_acc = float(len(c1p1_ans) - sum(c1p1_ans)) / len(c1p1_ans)
    c1p2_acc = float(len(c1p2_ans) - sum(c1p2_ans)) / len(c1p2_ans)
    c2p1_acc = float(len(c2p1_ans) - sum(c2p1_ans)) / len(c2p1_ans)
    c2p2_acc = float(len(c2p2_ans) - sum(c2p2_ans)) / len(c2p2_ans)

    disparity = abs((c1p1_acc - c2p1_acc) - (c1p2_acc - c2p2_acc))

    return disparity


def evaluate_question_difficulty(data):
    A_turns_p2 = count_turns(data["Turns"][:44])[0]
    B_turns_p2 = count_turns(data["Turns"][44:88])[0]

    A_turns_p1, B_turns_p1 = np.zeros(44), np.zeros(44)
    A_turns_p1[[i for i, val in enumerate(A_turns_p2) if val == 0]] = 1
    B_turns_p1[[i for i, val in enumerate(B_turns_p2) if val == 0]] = 1

    A_correct, B_correct = np.zeros(44), np.zeros(44)
    A_correct[[i for i, val in enumerate(list(data["lossVal"][:44])) if val == 0]] = 1
    B_correct[[i for i, val in enumerate(list(data["lossVal"][44:88])) if val == 0]] = 1
    A_correct_p1 = np.multiply(A_turns_p1, A_correct)
    A_correct_p2 = np.multiply(A_turns_p2, A_correct)
    B_correct_p1 = np.multiply(B_turns_p1, B_correct)
    B_correct_p2 = np.multiply(B_turns_p2, B_correct)

    return A_turns_p1, A_correct_p1, A_turns_p2, A_correct_p2, B_turns_p1, B_correct_p1, B_turns_p2, B_correct_p2


# ---------------------------------------------- Plotting Functions -------------------------------------------------- #


def get_probabilities(data, plot_it=False):
    c1p1 = list(data['prob-c1p1'][44:][data['prob-c1p1'].notnull()])
    c1p2 = list(data['prob-c1p2'][44:][data['prob-c1p2'].notnull()])
    c2p1 = list(data['prob-c2p1'][44:][data['prob-c2p1'].notnull()])
    c2p2 = list(data['prob-c2p2'][44:][data['prob-c2p2'].notnull()])

    if plot_it:
        sns.set_style("darkgrid")
        plt.figure()
        plt.plot(c1p1, label='c1p1')
        plt.plot(c1p2, label='c1p2')
        plt.plot(c2p1, label='c2p1')
        plt.plot(c2p2, label='c2p2')
        plt.legend()

    return c1p1, c1p2, c2p1, c2p2


def plot_stacked_bars(df1, df2, plot_title, colors):

    base_p1 = df1['p1']
    base_p2 = df1['p2']
    new_p1 = df2['p1']
    new_p2 = df2['p2']

    barWidth = 3
    r1 = 10*np.arange(min([len(base_p1), len(base_p2)]))
    r2 = [x + barWidth for x in r1]

    plt.figure()
    plt.bar(r1, base_p1, color=colors[0], edgecolor='white', width=barWidth, label='Base - p1')
    plt.bar(r1, base_p2, bottom=base_p1, color=colors[1], edgecolor='white', width=barWidth, label='Base - p2')
    plt.bar(r2, new_p1, color=colors[2], edgecolor='white', width=barWidth, label='New - p1')
    plt.bar(r2, new_p2, bottom=new_p1, color=colors[3], edgecolor='white', width=barWidth, label='New - p2')
    plt.xlabel("group")
    plt.title(plot_title)
    plt.legend(loc='upper left', prop={'size': 10})
    # plt.savefig('figures/'+plot_title+'.png')


def plot_reverse_bars(df1, df2, plot_title, colors):

    greenBars = df1['p1']
    blueBars = -1*df1['p2']
    orangeBars = df2['p1']
    yellowBars = -1*df2['p2']

    barWidth = 2
    r1 = 5*np.arange(min([len(greenBars), len(blueBars)]))
    r2 = [x + barWidth for x in r1]

    plt.figure()
    plt.bar(r1, greenBars, color=colors[0], edgecolor='white', width=barWidth, label='C1 - P1')
    plt.bar(r1, blueBars, color=colors[1], edgecolor='white', width=barWidth, label='C1 - P2')
    plt.bar(r2, orangeBars, color=colors[2], edgecolor='white', width=barWidth, label='C2 - P1')
    plt.bar(r2, yellowBars, color=colors[3], edgecolor='white', width=barWidth, label='C2 - P2')
    plt.xlabel("group")
    plt.title(plot_title)
    plt.legend(loc='upper left', prop={'size': 10})
    # plt.savefig('figures/'+plot_title+'.png')


# -------------------------------------------- Load and Pre-process data --------------------------------------------- #
# Variables to extract
new_c1_counts = {'p1': [], 'p2': []}
new_c2_counts = {'p1': [], 'p2': []}
base_c1_counts = {'p1': [], 'p2': []}
base_c2_counts = {'p1': [], 'p2': []}
new_correct = {'p1': [], 'p2': []}
base_correct = {'p1': [], 'p2': []}
new_c1_acc = {'p1': [], 'p2': []}
new_c2_acc = {'p1': [], 'p2': []}
base_c1_acc = {'p1': [], 'p2': []}
base_c2_acc = {'p1': [], 'p2': []}
accuracy = {'p1c1': [], 'p2c1': [], 'p1c2': [], 'p2c2': []}

disparity_p, disparity_c = [], []
base_disparity, new_disparity = [], []
fair1 = {'base': [], 'new': []}
fair2 = {'base': [], 'new': []}
trust = {'base': [], 'new': []}
fairness = {'base': [], 'new': []}
performance = {'base': [], 'new': []}
Comments = {"Part-A-Context": [], "Part-B-Context": [], "P1": [], "P2": []}

setA = {"Atp1": np.zeros(44), "Acp1": np.zeros(44), "Atp2": np.zeros(44), "Acp2": np.zeros(44)}
setB = {"Btp1": np.zeros(44), "Bcp1": np.zeros(44), "Btp2": np.zeros(44), "Bcp2": np.zeros(44)}
setA_correct = []
setB_correct = []

max_disparity, min_disparity = 0, 2
valid_users = 0  # to count valid users

# Load subjective data
dir = "subjective data"
for file in os.listdir(dir):
    fs = pd.read_csv(dir + "/" + file, header=1)

# Load and analyse objective data
dir = "objective data"
for file in os.listdir(dir):

    # Check if study was completed
    f = pd.read_csv(dir+"/"+file, header=2, sep='\s*,\s*', engine='python', usecols=[0])
    if f.shape[0] >= 88:

        # Load all data
        f = pd.read_csv(dir + "/" + file, header=2, sep='\s*,\s*', engine='python') #, usecols=range(1, 20))
        if f.shape[0] > 88:
            print(file,"contains duplicates")
            f.drop_duplicates('S. No', inplace=True)

        # Assign data to base and new
        p = pd.read_csv(dir + "/" + file, sep='\s*,\s*', engine='python', nrows=1)
        if p["SET-B_CONTEXT"][0]:
            base, new = f.loc[:43], f.loc[44:87]
        else:
            base, new = f.loc[44:87], f.loc[:43]

        f_turns_base, f_turns_new = base['Turns'], new['Turns']
        f_loss_base, f_loss_new = base["lossVal"], new["lossVal"]

        # Get player IDs
        # p_ids = list(set(f['player']))
        # p1_id, p2_id = p_ids[0], p_ids[1]
        p1_id, p2_id = p['Player_one'][0], p['Player_two'][0]

        # ---------------------------------------------- Performance ------------------------------------------------- #

        # Count number of questions each player answered
        base_turns, p1c1, p2c1, p1c2, p2c2 = count_turns(f_turns_base)
        base_c1_counts['p1'].append(p1c1)
        base_c1_counts['p2'].append(p2c1)
        base_c2_counts['p1'].append(p1c2)
        base_c2_counts['p2'].append(p2c2)
        fairness['base'].append(float(min([p1c1+p1c2, p2c1+p2c2])) / float(max([p1c1+p1c2, p2c1+p2c2])))

        new_turns, p1c1, p2c1, p1c2, p2c2 = count_turns(f_turns_new)
        new_c1_counts['p1'].append(p1c1)
        new_c1_counts['p2'].append(p2c1)
        new_c2_counts['p1'].append(p1c2)
        new_c2_counts['p2'].append(p2c2)
        fairness['new'].append(float(min([p1c1+p1c2, p2c1+p2c2])) / float(max([p1c1+p1c2, p2c1+p2c2])))

        # Count the number of correct answers (Baseline vs Fair CB)
        base_loss = list(f_loss_base)
        new_loss = list(f_loss_new)
        base_p1_correct, base_p2_correct = find_correct(base_turns, base_loss)
        new_p1_correct, new_p2_correct = find_correct(new_turns, new_loss)

        base_correct['p1'].append(base_p1_correct)
        base_correct['p2'].append(base_p2_correct)
        new_correct['p1'].append(new_p1_correct)
        new_correct['p2'].append(new_p2_correct)
        performance['base'].append(base_p1_correct + base_p2_correct)
        performance['new'].append(new_p1_correct + new_p2_correct)

        # Count the number of correct answers (Set A vs Set B)
        A_loss = sum(list(f.loc[:43]["lossVal"]))
        B_loss = sum(list(f.loc[44:87]["lossVal"]))
        setA_correct.append(44 - A_loss)
        setB_correct.append(44 - B_loss)

        # Context-wise accuracy
        c1p1_acc, c1p2_acc, c2p1_acc, c2p2_acc = contextwise_accuracy(base_turns, base_loss, output='acc')
        base_c1_acc['p1'].append(c1p1_acc)
        base_c1_acc['p2'].append(c1p2_acc)
        base_c2_acc['p1'].append(c2p1_acc)
        base_c2_acc['p2'].append(c2p2_acc)

        c1p1_acc, c1p2_acc, c2p1_acc, c2p2_acc = contextwise_accuracy(new_turns, new_loss, output='acc')
        new_c1_acc['p1'].append(c1p1_acc)
        new_c1_acc['p2'].append(c1p2_acc)
        new_c2_acc['p1'].append(c2p1_acc)
        new_c2_acc['p2'].append(c2p2_acc)

        bp1 = list(base['time'][base["player"] == p1_id])
        bp2 = list(base['time'][base["player"] == p2_id])
        np1 = list(new['time'][new["player"] == p1_id])
        np2 = list(new['time'][new["player"] == p2_id])

        # Total Disparity
        base_disparity.append(disparity(base_turns, base_loss))
        new_disparity.append(disparity(new_turns, new_loss))

        # Summed context-wise disparity
        # dp, dc = total_disparity(base_turns, base_loss, new_turns, new_loss)
        # disparity_p.append(dp)
        # disparity_c.append(dc)

        # Probabilities
        # p_c1p1, p_c1p2, p_c2p1, p_c2p2 = get_probabilities(f)

        # Calculate question difficulty
        Atp1, Acp1, Atp2, Acp2, Btp1, Bcp1, Btp2, Bcp2 = evaluate_question_difficulty(f)
        setA["Atp1"] = np.add(setA["Atp1"], Atp1); setA["Acp1"] = np.add(setA["Acp1"], Acp1)
        setA["Atp2"] = np.add(setA["Atp2"], Atp2); setA["Acp2"] = np.add(setA["Acp2"], Acp2)
        setB["Btp1"] = np.add(setB["Btp1"], Btp1); setB["Bcp1"] = np.add(setB["Bcp1"], Bcp1)
        setB["Btp2"] = np.add(setB["Btp2"], Btp2); setB["Bcp2"] = np.add(setB["Bcp2"], Bcp2)

        # Count number of valid users
        valid_users += 1

        # Find max and min disparity users
        if disparity(new_turns, new_loss) <= min_disparity:
            min_disparity = disparity(new_turns, new_loss)
            min_disparity_user = [new_turns, new_loss]
        if disparity(new_turns, new_loss) >= max_disparity:
            max_disparity = disparity(new_turns, new_loss)
            max_disparity_user = [new_turns, new_loss]

        # ----------------------------------- Find bad data points (DO NOT USE) -------------------------------------- #

        # # Remove if both players are from same location
        # p1_add = fs[['Location Latitude', 'Location Longitude']][fs['player_id'] == str(p1_id)].values
        # p2_add = fs[['Location Latitude', 'Location Longitude']][fs['player_id'] == str(p2_id)].values
        # if p1_add.all() == p2_add.all():
        #     print "remove this-"
        #     print file, (new_p1_correct+new_p2_correct) - (base_p1_correct+base_p2_correct), p1_add, p2_add
        # # If a player did not answer
        # if (sum(bp1) >= 9*len(bp1)) or (sum(bp2) >= 9*len(bp2)) or (sum(np1) >= 9*len(np1)) or (sum(np2) >= 9*len(np2)):
        #     print count, "remove this-", file, (new_p1_correct+new_p2_correct) - (base_p1_correct+base_p2_correct)
        # # If users are not attentive
        # if sum(f_loss_base[:4]) > 2 or sum(f_loss_new[:4]) > 2:
        #     print count, "not attentive-", file, (new_p1_correct + new_p2_correct) - (base_p1_correct + base_p2_correct)
        # # If baseline performs better than new method
        # if (base_p1_correct+base_p2_correct) > (new_p1_correct+new_p2_correct):
        #     print count, "poor result-", file, (new_p1_correct+new_p2_correct) - (base_p1_correct+base_p2_correct)

        # ----------------------------------------------- Fairness --------------------------------------------------- #

        # Check if the users's ids are in the subjective data
        check1 = list(fs.index[fs['player_id'] == str(p1_id)])
        check2 = list(fs.index[fs['player_id'] == str(p2_id)])
        if check1 and check2:
            # Check if the second set was FairCB
            if p["SET-B_CONTEXT"][0]:
                # hard coded indices of the 6 subjective questions
                idx = [10, 11, 12, 13, 14, 15]
            else:
                idx = [13, 14, 15, 10, 11, 12]

            # Populate the 1-7 Likert scale responses to subjective questions
            fair1["base"].append([fs[fs.columns[idx[0]]][check1[0]], fs[fs.columns[idx[0]]][check2[0]]])
            fair2["base"].append([fs[fs.columns[idx[1]]][check1[0]], fs[fs.columns[idx[1]]][check2[0]]])
            trust["base"].append([fs[fs.columns[idx[2]]][check1[0]], fs[fs.columns[idx[2]]][check2[0]]])
            fair1["new"].append([fs[fs.columns[idx[3]]][check1[0]], fs[fs.columns[idx[3]]][check2[0]]])
            fair2["new"].append([fs[fs.columns[idx[4]]][check1[0]], fs[fs.columns[idx[4]]][check2[0]]])
            trust["new"].append([fs[fs.columns[idx[5]]][check1[0]], fs[fs.columns[idx[5]]][check2[0]]])

            # Populate open-ended comments from users
            Comments["Part-A-Context"].append(p["SET-A_CONTEXT"][0])
            Comments["Part-B-Context"].append(p["SET-B_CONTEXT"][0])
            Comments["P1"].append(fs[list(fs.columns)[18]][check1[0]])
            Comments["P2"].append(fs[list(fs.columns)[18]][check2[0]])

# -------------------------------------------------- Make plots ------------------------------------------------------ #
sns.set(style="darkgrid", context="paper")

# Accuracy plots
df1 = pd.DataFrame(base_correct)
df2 = pd.DataFrame(new_correct)
# plot_stacked_bars(df1, df2, "Total number of correct answers", ['#3366ff', '#99b3ff', '#5bd75b', '#adebad'])

plt.figure()
X = ["Baseline"]*len(performance["base"]) + ["Fair CB"]*len(performance["new"])
Y = list(performance["base"]) + list(performance["new"])
sns.barplot(x=X, y=Y, capsize=.2)
plt.ylim(0, 44)
plt.ylabel("Number of correct answers", fontsize=28)
plt.xticks(fontsize=28)
plt.gcf().subplots_adjust(left=0.145)
# plt.savefig("figures/performance_study.jpg")

x1 = np.add(base_correct["p1"], base_correct["p2"])
x2 = np.add(new_correct["p1"], new_correct["p2"])
print("Base mean no. of correct answers",  np.mean(x1), np.mean(x1/44), "with std. error=", sp.stats.sem(x1))
print("New mean no. of correct answers", np.mean(x2), np.mean(x2/44), "with std. error=", sp.stats.sem(x2))
print(sp.stats.ttest_rel(performance["base"], performance["new"]))

# Performance Set A vs Set B
df1 = pd.DataFrame(setA_correct)
df2 = pd.DataFrame(setB_correct)
df1.to_csv(r'setA_correct.csv')
df2.to_csv(r'setB_correct.csv')
X = ["Set A"]*len(setA_correct) + ["Set B"]*len(setB_correct)
Y = setA_correct + setB_correct
print("Set A: performance=", np.mean(setA_correct), "with std. error=", sp.stats.sem(setA_correct))
print("Set B: performance=", np.mean(setB_correct), "with std. error=", sp.stats.sem(setB_correct))
print(sp.stats.ttest_rel(setA_correct, setB_correct))

# Disparity vs accuracy
plt.figure()
sns.regplot(base_disparity, performance["base"])
sns.regplot(new_disparity, performance["new"])
plt.ylim(0, 44)
plt.xlim(0, 2.0)
plt.xlabel("Disparity", fontsize=28)
plt.ylabel("Number of correct answers", fontsize=28)
plt.legend(["Baseline", "Fair CB"], loc='lower right', fontsize=28)
plt.gcf().subplots_adjust(bottom=0.16)
plt.gcf().subplots_adjust(left=0.145)
# plt.savefig("figures/context_disparity_study.jpg")

lm = linear_model.LinearRegression()
base_model = lm.fit(np.matrix(base_disparity).T, performance["base"])
lm = linear_model.LinearRegression()
new_model = lm.fit(np.matrix(new_disparity).T, performance["new"])
print("Base: coef=", base_model.coef_)
print("New: coef=", new_model.coef_)
df = pd.DataFrame({"base_disparity": base_disparity, "base_performance": performance["base"], "new_disparity": new_disparity, "new_performance": performance["new"]})
df.to_csv(r'disparity_vs_performance.csv')

# Disparity vs Assignment
plt.figure()
colors = ['g', 'r']
for question_num, turn in enumerate(max_disparity_user[0]):
    plt.plot(question_num+1, turn, color=colors[max_disparity_user[1][question_num]], marker='s', markersize=6)
plt.xlabel("Question number")
plt.yticks([0, 1], ['USA', 'India'])
plt.legend(["Correct", "Incorrect"], bbox_to_anchor=[0.4, 1.0])
plt.axes().set_aspect(1.5)
plt.savefig("figures/high_disparity_behavior.jpg")

# Fairness plots
Y1P1 = [float(x[0]) for x in np.array(fair1["base"])[:, 0] if str(x) != 'nan']
Y1P2 = [float(x[0]) for x in np.array(fair1["base"])[:, 1] if str(x) != 'nan']
Y2P1 = [float(x[0]) for x in np.array(fair1["new"])[:, 0] if str(x) != 'nan']
Y2P2 = [float(x[0]) for x in np.array(fair1["new"])[:, 1] if str(x) != 'nan']
Y_fair1 = Y1P1+Y1P2+Y2P1+Y2P2
X_fair1 = ["Baseline"]*(len(Y1P1)+len(Y1P2)) + ["Fair CB"]*(len(Y2P1)+len(Y2P2))
H = ["P1"]*len(Y1P1) + ["P2"]*len(Y1P2) + ["P1"]*len(Y2P1) + ["P2"]*len(Y2P2)

Mu1 = Y1P1+Y1P2
Mu2 = Y2P1+Y2P2
margin = 0.1*np.mean(Mu1)
tt = sp.stats.ttest_rel(np.subtract(Mu1, margin), Mu2)
print("Fair1:", "Base -", np.mean(Mu1), sp.stats.sem(Mu1), "New -", np.mean(Mu2), sp.stats.sem(Mu2))
print("Fair 1:", "statistic", tt.statistic, "p-value", tt.pvalue/2, "for margin =", margin)

Y1P1 = [float(x[0]) for x in np.array(fair2["base"])[:, 0] if str(x) != 'nan']
Y1P2 = [float(x[0]) for x in np.array(fair2["base"])[:, 1] if str(x) != 'nan']
Y2P1 = [float(x[0]) for x in np.array(fair2["new"])[:, 0] if str(x) != 'nan']
Y2P2 = [float(x[0]) for x in np.array(fair2["new"])[:, 1] if str(x) != 'nan']
Y_fair2 = Y1P1+Y1P2+Y2P1+Y2P2
X_fair2 = ["Baseline"]*(len(Y1P1)+len(Y1P2)) + ["Fair CB"]*(len(Y2P1)+len(Y2P2))
H = ["P1"]*len(Y1P1) + ["P2"]*len(Y1P2) + ["P1"]*len(Y2P1) + ["P2"]*len(Y2P2)

Mu1 = Y1P1+Y1P2
Mu2 = Y2P1+Y2P2
margin = 0.1*np.mean(Mu1)
tt = sp.stats.ttest_rel(np.subtract(Mu1, margin), Mu2)
print("Fair2:", "Base -", np.mean(Mu1), sp.stats.sem(Mu1), "New -", np.mean(Mu2), sp.stats.sem(Mu2))
print("Fair 2:", "statistic", tt.statistic, "p-value", tt.pvalue/2, "for margin =", margin)

Y1P1 = [float(x[0]) for x in np.array(trust["base"])[:, 0] if str(x) != 'nan']
Y1P1_idx = [idx for idx, x in enumerate(np.array(trust["base"])[:, 0]) if str(x) != 'nan']
Y1P2 = [float(x[0]) for i, x in enumerate(np.array(trust["base"])[:, 1]) if i in Y1P1_idx]
Y2P1 = [float(x[0]) for i, x in enumerate(np.array(trust["new"])[:, 0]) if i in Y1P1_idx]
Y2P2 = [float(x[0]) for i, x in enumerate(np.array(trust["new"])[:, 1]) if i in Y1P1_idx]
Y_trust = Y1P1+Y1P2+Y2P1+Y2P2
X_trust = ["Baseline"]*(len(Y1P1)+len(Y1P2)) + ["Fair CB"]*(len(Y2P1)+len(Y2P2))
H = ["P1"]*len(Y1P1) + ["P2"]*len(Y1P2) + ["P1"]*len(Y2P1) + ["P2"]*len(Y2P2)

Mu1 = Y1P1+Y1P2
Mu2 = Y2P1+Y2P2
margin = 0.1*np.mean(Mu1)
tt = sp.stats.ttest_rel(np.subtract(Mu1, margin), Mu2)
print("Trust:", "Base -", np.mean(Mu1), sp.stats.sem(Mu1), "New -", np.mean(Mu2), sp.stats.sem(Mu2))
print("Trust:", "statistic", tt.statistic, "p-value", tt.pvalue/2, "for margin =", margin)
print("Trust ANOVA:", sp.stats.f_oneway(Y1P1+Y1P2, Y2P1+Y2P2))

# Combined subjective plot
X = ["Q1"]*(len(Y_fair1)) + ["Q2"]*(len(Y_fair2)) + ["Q3"]*(len(Y_trust))
plt.figure()
sns.barplot(X, Y_fair1+Y_fair2+Y_trust, X_fair1+X_fair2+X_trust)
plt.legend(loc="center", bbox_to_anchor=(0.5, 0.87), fontsize=26)
plt.ylim(1, 7)
plt.xticks(fontsize=28)
plt.ylabel("Rating", fontsize=28)
# plt.savefig("figures/subjective_response.jpg")

# Comments
df = pd.DataFrame(Comments)
df.to_csv(r'comments.csv')

# # Count turns plots
# df1 = pd.DataFrame(base_c1_counts)
# df2 = pd.DataFrame(base_c2_counts)
# plot_reverse_bars(df1, df2, "Base - Number of turn each player-each context", ['#0077b3', '#80d4ff', '#9966ff', '#ccb3ff'])
# df1 = pd.DataFrame(new_c1_counts)
# df2 = pd.DataFrame(new_c2_counts)
# plot_reverse_bars(df1, df2, "New - Number of turn each player-each context", ['#00b33c', '#4dff88', '#00cc99', '#80ffdf'])
#
# # Context-wise accuracy
# df1 = pd.DataFrame(base_c1_acc)
# df2 = pd.DataFrame(base_c2_acc)
# plot_reverse_bars(df1, df2, "Base - Accuracy of each player-each context", ['#0077b3', '#80d4ff', '#9966ff', '#ccb3ff'])
# df1 = pd.DataFrame(new_c1_acc)
# df2 = pd.DataFrame(new_c2_acc)
# plot_reverse_bars(df1, df2, "New - Accuracy of each player-each context", ['#00b33c', '#4dff88', '#00cc99', '#80ffdf'])


# Individual question difficulty
#AB = {'A-P1': np.divide(setA["Acp1"], setA["Atp1"]), 'A-P2': np.divide(setA["Acp2"], setA["Atp2"]),
#      'B-P1': np.divide(setB["Bcp1"], setB["Btp1"]), 'B-P2': np.divide(setB["Bcp2"], setB["Btp2"])}
#df = pd.DataFrame(AB)
#df.to_csv(r'questions_difficulty.csv')

# Objective response count
# 1 - No difference, 2 - Some difference, 3 - Recognised Algo, 4 - Wrongly recognised diff or algo
#df = pd.read_csv("subjective data/comments.csv")
#responses = list(df['P1-Category']) + list(df['P2-Category'])
#print("No diff=", responses.count(1), "Some diff=", responses.count(2), "Algo rec=", responses.count(3))