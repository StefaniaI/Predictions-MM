import simulation as sim

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# This file is concerned with the analysis of SimulationStatistics

current_variable_params = ['random_seed_no', 'matching_algorithm', 'utility_composition',
                           'no_students', 'school_capacity']
no_seeds = 20


def read_sims(no, path='Simulation_results\\'):
    '''Constructs simulations for no many previously run ones.
    no = the number of simulations to be loaded
    path = the folder where the config and run files are'''

    all_sims = []
    for n in range(no):
        s = sim.Simulation(path + 'config' + str(n) + '.csv')
        # to save time, we only read the results of a simulation when needed
        s.read(False)
        all_sims.append(s)

    return all_sims


def plot_1b(sims, x_par='matching_algorithm', y_val='utility',
            style_par='utility_composition', fixed_pars={}, row_param='distr', col_param='sizee'):
    '''This polots bar plots for the plot in Part 1a of the results.
    It shows the how much incentive do agents have to deviate from truthful interaction.'''

    stra = 'strategy'

    cols = [x_par, y_val, stra, style_par, row_param, col_param, 'seed']
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df[x_par].append(s.config[x_par])
            df[y_val].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])
            df[style_par].append(s.config[style_par][0:3])
            df[row_param].append(s.config['mu'] + s.config['sigma'])
            no_sc = s.config['no_schools']
            comp_n = s.config['no_schools'] * \
                s.config['school_capacity'] / s.config['no_students']
            df[col_param].append(no_sc + comp_n)

            df['seed'].append(s.config['random_seed_no'])

    df = pd.DataFrame(data=df)

    # Rename
    match = {'SD-random': 'RSD', 'Boston-random': 'Boston', 'DA-random': 'DA', 'SD-by_result': 'SD'}
    for m in match:
        df['matching_algorithm'] = df['matching_algorithm'].replace(m, match[m])

    strat = {'best_for_cheap': 'Strategic', 'best_for_student': 'Truthful'}
    for m in strat:
        df['strategy'] = df['strategy'].replace(m, strat[m])

    df = df.rename(columns={'matching_algorithm': "Matching algorithm",
                            'utility': 'Average utility',
                            'strategy': 'Interaction'})
    x_par = "Matching algorithm"
    y_val = 'Average utility'
    stra = 'Interaction'

    changing = {}
    # changing['rows'] = [[2.5, 100], [3.2, 3], [3.2, 0.65], [1, 0.65], [1, 3]]
    changing['rows'] = [[1, 3]]
    n_rows = len(changing['rows'])
    # changing['cols'] = [[10, 1], [10, 4], [10, 1/4]]
    changing['cols'] = [[10, 4]]
    no_st_per_place = {1: 1, 4: 1/4, 1/4: 4}
    n_cols = len(changing['cols'])

    # for AAAI: variables for figsize (10, 3) -->
    sns.set(font_scale=1.1)
    fig, ax = plt.subplots(1, 2, figsize=(3.7*n_cols*0.9, 2.2*n_rows*0.9))
    df1 = df[df[row_param] == sum(changing['rows'][0])]
    df1 = df1[df1[col_param] == sum(changing['cols'][0])]
    df1 = df1.drop_duplicates()

    g = sns.barplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'pos'],
                    palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[0],
                    hue_order=['Truthful', 'Strategic'])
    title = 'Positive utility.'
    g.set_title(title)
    g.legend(loc='lower center', prop={'size': 10}, ncol=2)
    print(len(df1[df1[style_par] == 'pos']))

    g = sns.barplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'neg'],
                    palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[1],
                    hue_order=['Truthful', 'Strategic'])
    title = 'Negative utility.'
    g.set_title(title)
    print(len(df1[df1[style_par] == 'neg']))

    g.get_legend().remove()
    fig.tight_layout()
    plt.show()

    return no_sims


def plot_1a(sims, x_par='matching_algorithm', y_val='utility',
            style_par='utility_composition', fixed_pars={}, row_param='sizee', col_param='distr'):
    '''This polots bar plots for the plot in Part 1a of the results.
    It shows the how much incentive do agents have to deviate from truthful interaction.'''

    util = 'Average utility'

    stra = 'strategy'

    cols = [x_par, y_val, stra, style_par, row_param, col_param, 'seed']
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df[x_par].append(s.config[x_par])
            df[y_val].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])
            df[style_par].append(s.config[style_par][0:3])
            df[col_param].append(s.config['mu'] + s.config['sigma'])
            no_sc = s.config['no_schools']
            comp_n = s.config['no_schools'] * \
                s.config['school_capacity'] / s.config['no_students']
            df[row_param].append(no_sc + comp_n)

            df['seed'].append(s.config['random_seed_no'])

    df = pd.DataFrame(data=df)

    # Rename
    match = {'SD-random': 'RSD', 'Boston-random': 'Boston', 'DA-random': 'DA', 'SD-by_result': 'SD'}
    for m in match:
        df['matching_algorithm'] = df['matching_algorithm'].replace(m, match[m])

    strat = {'best_for_cheap': 'Strategic', 'best_for_student': 'Truthful'}
    for m in strat:
        df['strategy'] = df['strategy'].replace(m, strat[m])

    df = df.rename(columns={'matching_algorithm': "Matching algorithm",
                            'utility': util,
                            'strategy': 'Interaction'})
    x_par = "Matching algorithm"
    y_val = util
    stra = 'Interaction'

    changing = {}
    # changing['rows'] = [[2.5, 100], [3.2, 3], [3.2, 0.65], [1, 0.65], [1, 3]]
    changing['rows'] = [[10, 1], [10, 1/4]]
    no_st_per_place = {1: 1, 4: 1/4, 1/4: 4}
    n_rows = len(changing['rows'])
    # changing['cols'] = [[10, 1], [10, 4], [10, 1/4]]
    changing['cols'] = [[1, 3],  [3.2, 3], [3.2, 0.65]]
    n_cols = len(changing['cols'])

    # for AAAI: needed to cut one column; variables for figsize 5-->3.7, 3-->2.2
    n_cols -= 1
    # comp = '# students per place '
    comp = 'competition '

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3.7*n_cols*0.9, 2.2*n_rows*0.9))

    shown_legend = False
    sns.set(font_scale=1.1)
    for r in range(n_rows):
        for c in range(n_cols):
            df1 = df[df[row_param] == sum(changing['rows'][r])]
            df1 = df1[df1[col_param] == sum(changing['cols'][c])]
            df1 = df1[df1[style_par] == 'pos']
            df1 = df1.drop_duplicates()

            # add , join=False to remove lines in pointplot
            g = sns.barplot(x=x_par, y=y_val, hue=stra, data=df1,
                            palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c],
                            hue_order=['Truthful', 'Strategic'])
            print(len(df1))

            # drop labels that are not useful
            if r == 0:
                ax[r][c].set(xlabel=None)
            if c != 0:
                ax[r][c].set(ylabel=None)

            # g.set_xlabel("Matching algorithm", fontsize=0.9)
            # g.set_ylabel('Average utility', fontsize=0.9)
            title = 'Mean ' + str(changing['cols'][c][0])
            title += ', s.d. ' + str(changing['cols'][c][1]) + '; '
            title += comp + str(no_st_per_place[changing['rows'][r][1]])

            # show utility starting from 0
            # ax[r][c].set_ylim(ymin=0, ymax=max(df1[y_val]*1.1))

            # change the colours of the error bars
            '''
            lines = ax[r][c].get_lines()
            for i in range(0, len(lines)):
                if i % (len(lines)/2) != 0:
                    lines[i].set_color('black')'''

            g.set_title(title)
            if shown_legend:
                g.get_legend().remove()
            else:
                shown_legend = True
                # change the size of legend
                # plt.setp(g.get_legend().get_texts(), fontsize='10')
                # plt.setp(g.get_legend().get_title(), fontsize='10')
                g.legend(loc='lower center', prop={'size': 10}, ncol=2)

    fig.tight_layout()
    plt.show()

    return no_sims


def plot_part1_all(sims, x_par='matching_algorithm', y_val='utility',
                   style_par='utility_composition', fixed_pars={}, row_param='distr',
                   col_param='sizee'):
    '''Differently from plot_2, this function compares the utility of the same agent when switching
    streategies.'''

    stra = 'strategy'

    cols = [x_par, y_val, stra, style_par, row_param, col_param]
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df[x_par].append(s.config[x_par])
            df[y_val].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])
            df[style_par].append(s.config[style_par][0:3])
            df[row_param].append(s.config['mu'] + s.config['sigma'])
            no_sc = s.config['no_schools']
            comp_n = s.config['no_schools'] * \
                s.config['school_capacity'] / s.config['no_students']
            df[col_param].append(no_sc + comp_n)

    df = pd.DataFrame(data=df)

    changing = {}
    changing['rows'] = [[2.5, 100], [3.2, 3], [3.2, 0.65], [1, 0.65], [1, 3]]
    n_rows = len(changing['rows'])
    changing['cols'] = [[2, 1], [2, 4], [2, 1/4], [10, 1], [10, 4], [10, 1/4]]
    n_cols = len(changing['cols'])

    fig, ax = plt.subplots(n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            df1 = df[df[row_param] == sum(changing['rows'][r])]
            df1 = df1[df1[col_param] == sum(changing['cols'][c])]
            g = sns.pointplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'pos'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
            g = sns.pointplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'neg'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
    plt.show()

    return no_sims


def plot_old(sims, x_par='matching_algorithm', y_val='utility', hue_par='strategy',
             style_par='utility_composition', fixed_pars={}, row_param='distr', col_param='size'):
    '''Selects the relevant simulations and creates a data frame for future plotting.
    sims = list of simulations (possibly with configs only)
    x_par = the parameter on the x axis
    y_val = the value that will be reported on the y axis
    hue_par = the parameter that will be used for different bar columns
    fixed_parameters = the parameters that will remain fixed and their values
    e.g. da.make_df(a, fixed_pars = {'utility_composition': 'pos_min', 'school_capacity': 80})'''

    cols = [x_par, y_val, hue_par, style_par, row_param, col_param]
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            mean_utility_by_hue_par = s.results.groupby(hue_par)['utility'].mean()
            for i in mean_utility_by_hue_par.index:
                df[x_par].append(s.config[x_par])
                df[y_val].append(mean_utility_by_hue_par[i])
                df[hue_par].append(i)
                df[style_par].append(s.config[style_par][0:3])
                df[row_param].append(s.config['mu'] + s.config['sigma'])
                no_sc = s.config['no_schools']
                comp_n = s.config['no_schools'] * \
                    s.config['school_capacity'] / s.config['no_students']
                df[col_param].append(no_sc + comp_n)

    df = pd.DataFrame(data=df)

    changing = {}
    changing['rows'] = [[2.5, 100], [3.2, 3], [3.2, 0.65], [1, 3]]
    changing['cols'] = [[2, 1], [2, 4], [2, 1/4], [10, 1], [10, 4], [10, 1/4]]

    fig, ax = plt.subplots(4, 6)

    for r in range(4):
        for c in range(6):
            df1 = df[df[row_param] == sum(changing['rows'][r])]
            df1 = df1[df1[col_param] == sum(changing['cols'][c])]
            g = sns.pointplot(x=x_par, y=y_val, hue=hue_par, data=df1[df1[style_par] == 'pos'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
            g = sns.pointplot(x=x_par, y=y_val, hue=hue_par, data=df1[df1[style_par] == 'neg'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
    plt.show()

    return no_sims


def make_df(sims, x_par='matching_algorithm', y_val='utility', hue_par='strategy',
            style_par='utility_composition', fixed_pars={}, row_param='distr', col_param='size'):
    '''Selects the relevant simulations and creates a data frame for future plotting.
    sims = list of simulations (possibly with configs only)
    x_par = the parameter on the x axis
    y_val = the value that will be reported on the y axis
    hue_par = the parameter that will be used for different bar columns
    fixed_parameters = the parameters that will remain fixed and their values
    e.g. da.make_df(a, fixed_pars = {'utility_composition': 'pos_min', 'school_capacity': 80})'''

    cols = [x_par, y_val, hue_par, style_par, row_param, col_param]
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1
            if int(s.get_sim_no()) < 20160:
                print(s.get_sim_no())
                input()

            # append the relevant data
            mean_utility_by_hue_par = s.results.groupby(hue_par)['utility'].mean()
            for i in mean_utility_by_hue_par.index:
                df[x_par].append(s.config[x_par])
                df[y_val].append(mean_utility_by_hue_par[i])
                df[hue_par].append(i)
                df[style_par].append(s.config[style_par][0:3])
                df[row_param].append(s.config['mu'] + s.config['sigma'])
                no_sc = s.config['no_schools']
                comp_n = s.config['no_schools'] * \
                    s.config['school_capacity'] / s.config['no_students']
                df[col_param].append(no_sc + comp_n)

    df = pd.DataFrame(data=df)

    changing = {}
    changing['rows'] = [[2.5, 100], [3.2, 3], [3.2, 1.5], [3.2, 0.65], [1, 0.65], [1, 1.5], [1, 3]]
    changing['cols'] = [[2, 1], [2, 4], [2, 1/4], [10, 1], [10, 4], [10, 1/4]]

    fig, ax = plt.subplots(7, 6)

    for r in range(7):
        for c in range(6):
            df1 = df[df[row_param] == sum(changing['rows'][r])]
            df1 = df1[df1[col_param] == sum(changing['cols'][c])]
            g = sns.pointplot(x=x_par, y=y_val, hue=hue_par, data=df1[df1[style_par] == 'pos'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
            g = sns.pointplot(x=x_par, y=y_val, hue=hue_par, data=df1[df1[style_par] == 'neg'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
    plt.show()

    return no_sims


def group_results(all_sims, changing=[]):
    '''Forms a csv file with the results for the utility and different strategies.
    changing = all parameters that vary in the simulations'''

    # for ve_1
    changing_var_lists = ['utility_from_strategy', 'random_seed_no', 'mu', 'sigma',
                          'matching_algorithm',
                          'utility_composition', 'no_schools', '#students/place',
                          'rs_obs_noise', 'recommendation_weight']
    changing = changing_var_lists

    sc = 'school_capacity'
    no_st = 'no_students'
    df = {c: [] for c in changing}
    for s in all_sims:
        # load the simulation results
        s.read()

        # get mean utility depending on the type of strategy
        mean_utility_by_strategy = s.results.groupby('strategy')['utility'].mean()
        '''
        for i in mean_utility_by_strategy.index:
            for c in changing:
                if c == '#students/place':
                    df[c].append(s.config[no_st]/(s.config[sc]*s.config['no_schools']))
                elif c == 'utility':
                    df[c].append(mean_utility_by_strategy[i])
                elif c == 'strategy':
                    df[c].append(i)
                else:
                    df[c].append(s.config[c])'''

        for c in changing:
            if c == '#students/place':
                df[c].append(s.config[no_st]/(s.config[sc]*s.config['no_schools']))
            elif c == 'utility_from_strategy':
                df[c].append(mean_utility_by_strategy['best_for_cheap'] -
                             mean_utility_by_strategy['best_for_student'])
            else:
                df[c].append(s.config[c])

    df = pd.DataFrame(data=df)
    df.to_csv('simulation_results.csv')

# -------------------- Analysis part 2: accuracy & trust -------------------------------


def plot_2(sims, x_par='matching_algorithm', y_val='utility',
           style_par='utility_composition', fixed_pars={}, row_param='rs_obs_noise',
           col_param='recommendation_weight'):
    '''['rs_obs_noise'], ['recommendation_weight']'''

    stra = 'strategy'

    cols = [x_par, y_val, stra, style_par, row_param, col_param, 'seed']
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df[x_par].append(s.config[x_par])
            df[y_val].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])
            df[style_par].append(s.config[style_par][0:3])
            no_sc = s.config['no_schools']
            comp_n = s.config['no_schools'] * \
                s.config['school_capacity'] / s.config['no_students']
            df[row_param].append(s.config[row_param])
            df[col_param].append(s.config[col_param])

            df['seed'].append(s.config['random_seed_no'])

    df = pd.DataFrame(data=df)
    # Rename
    match = {'SD-random': 'RSD', 'Boston-random': 'Boston', 'DA-random': 'DA', 'SD-by_result': 'SD'}
    for m in match:
        df['matching_algorithm'] = df['matching_algorithm'].replace(m, match[m])

    strat = {'best_for_cheap': 'Strategic', 'best_for_student': 'Truthful'}
    for m in strat:
        df['strategy'] = df['strategy'].replace(m, strat[m])

    df = df.rename(columns={'matching_algorithm': "Matching algorithm",
                            'utility': 'Average utility',
                            'strategy': 'Interaction'})
    x_par = "Matching algorithm"
    y_val = 'Average utility'
    stra = 'Interaction'

    changing = {}
    changing['rows'] = [0.0005, 0.5, 1.5]
    noise = {0.0005: 0.01, 0.05: 1, 0.5: 10, 0.75: 15, 1.5: 30}  # 0.01%, 1%, 15%, 30%
    n_rows = len(changing['rows'])
    changing['cols'] = [1, 0.5]
    n_cols = len(changing['cols'])

    # for AAAI: needed to cut one column; variables for figsize 5-->3.7, 3-->2.1
    n_rows -= 1
    # comp = '# students per place '
    comp = 'competition '

    fig, ax = plt.subplots(n_cols, n_rows, figsize=(3.7*n_rows, 2.1*n_cols))

    shown_legend = False
    for r in range(n_rows):
        for c in range(n_cols):
            # print(df.rs_obs_noise.unique(), df.recommendation_weight.unique())
            df1 = df[df[row_param] == changing['rows'][r]]
            df1 = df1[df1[col_param] == changing['cols'][c]]
            df1 = df1[df1[style_par] == 'pos']
            df1 = df1.drop_duplicates()
            print(len(df1))

            g = sns.barplot(x=x_par, y=y_val, hue=stra, data=df1,
                            palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[c][r],
                            hue_order=['Truthful', 'Strategic'])

            # drop labels that are not useful
            if c == 0:
                ax[c][r].set(xlabel=None)
            if r != 0:
                ax[c][r].set(ylabel=None)

            title = 'Noise ' + str(noise[changing['rows'][r]]) + '%, '
            title += 'trust ' + str(changing['cols'][c])
            g.set_title(title)
            if shown_legend:
                g.get_legend().remove()
            else:
                shown_legend = True
                g.legend(loc='lower center', prop={'size': 10}, ncol=2)
            print(len(df1[df1[style_par] == 'pos']))

    fig.tight_layout()
    plt.show()

    return no_sims


def plot_part2_all(sims, x_par='matching_algorithm', y_val='utility',
                   style_par='utility_composition', fixed_pars={}, row_param='rs_obs_noise',
                   col_param='recommendation_weight'):
    '''['rs_obs_noise'], ['recommendation_weight']'''

    stra = 'strategy'

    cols = [x_par, y_val, stra, style_par, row_param, col_param]
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df[x_par].append(s.config[x_par])
            df[y_val].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])
            df[style_par].append(s.config[style_par][0:3])
            no_sc = s.config['no_schools']
            comp_n = s.config['no_schools'] * \
                s.config['school_capacity'] / s.config['no_students']
            df[row_param].append(s.config[row_param])
            df[col_param].append(s.config[col_param])

    df = pd.DataFrame(data=df)
    changing = {}
    changing['rows'] = [0.0005, 0.05, 0.75, 1.5]
    n_rows = len(changing['rows'])
    changing['cols'] = [1, 0.5]
    n_cols = len(changing['cols'])

    fig, ax = plt.subplots(n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            # print(df.rs_obs_noise.unique(), df.recommendation_weight.unique())
            df1 = df[df[row_param] == changing['rows'][r]]
            df1 = df1[df1[col_param] == changing['cols'][c]]
            g = sns.pointplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'pos'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
            g = sns.pointplot(x=x_par, y=y_val, hue=stra, data=df1[df1[style_par] == 'neg'],
                              kind='bar',
                              palette=sns.color_palette(['#437CDF', '#70AD47']), ax=ax[r][c])
            g.get_legend().remove()
    plt.show()

    return no_sims


# -------------------- Analysis for the best response -------------------------------
def read_best_response_data(file_name='best_response.csv', attribute_bins=[[0], [1, 2, 3, 4, 5]]):
    '''This functions selects the relevant statistics for the path taken with the best responses
    '''

    # read data
    df = pd.read_csv(file_name)

    def get_welfare_by_attribute_bin(df_cur):
        welfare_by_bin = []
        for a_list in attribute_bins:
            w = np.mean([np.mean(df_cur['welfare-'+str(a)]) for a in a_list])
            welfare_by_bin.append(w)
        return welfare_by_bin

    # in every round select the best response
    results = []
    no_rounds = max(df['round'])
    for r in range(no_rounds + 1):
        if r == 0:
            # this is the original scenario; need the all-best_for_student starategy
            df_orig = df[(df['round'] == 1) & (df['strategy'] == 0)]
            u0 = np.mean(df_orig['utility_0'])
            u1 = np.mean(df_orig['utility_1'])
            w = np.mean(df_orig['welfare'])
            strategy = min(df_orig['strategy'])
            results.append((u0, u1, 2, w, strategy, get_welfare_by_attribute_bin(df_orig)))
        else:
            df_now = df[(df['round'] == r) & (df['best_str'] == True)]
            u0 = np.mean(df_now['utility_0'])
            u1 = np.mean(df_now['utility_1'])
            deciding = min(df_now['deciding_agent'])
            strategy = min(df_now['strategy'])
            w = np.mean(df_now['welfare'])
            if u0 != results[-1][0] or u1 != results[-1][1]:
                results.append((u0, u1, deciding, w, strategy,
                                get_welfare_by_attribute_bin(df_now)))

    return results


def plot_student_welfare(file_name='best_response.csv', attribute_bins=[[0], [1, 2, 3, 4, 5]]):
    '''Plots the student welfare for each point on the path of best responses.'''

    from copy import deepcopy

    # read data
    df = pd.read_csv(file_name)

    # select the relevant rows - initial state or best response of round
    df_welfare = deepcopy(df[((df['round'] == 1) & (df['strategy'] == 0)) | (df['best_str'])])

    # create new rows with the attack level of each agent group
    df_welfare['attack_level'] = df_welfare['seed']
    attack = [0, 0]
    for i, row in df_welfare.iterrows():
        if row['strategy'] != attack[row['deciding_agent']]:
            attack[row['deciding_agent']] = row['strategy']
        df_welfare.at[i, 'attack_level'] = deepcopy(attack)

    # remove the rows where the strategy doesn't change
    to_remove = []
    previous_round = 0
    previous_attack = [-1, -1]
    for i, row in df_welfare.iterrows():
        if row['round'] != previous_round:
            decision_to_remove_round = False
            if previous_attack == row['attack_level']:
                to_remove.append(i)
                decision_to_remove_round = True
        elif decision_to_remove_round:
            to_remove.append(i)

        previous_round = row['round']
        previous_attack = row['attack_level']
    df_welfare = df_welfare.drop(to_remove)

    # create new rows for each group
    group = 'Entry knowledge'
    df_welfare[group] = 'all'
    df_basis = deepcopy(df_welfare)
    for a in range(2):
        df_a = deepcopy(df_basis)
        name = 'low' if a == 0 else 'high'
        df_a[group] = df_a[group].apply(lambda x: name)
        df_a['welfare'] = df_a['welfare-'+str(a)]
        df_welfare = pd.concat([df_welfare, df_a], axis=0)

    # plot results
    def format_strategy(x):
        # return ("{:.0%}".format(x[0]), "{:.0%}".format(x[1]))
        return (int(x[0]*100), int(x[1]*100))

    # for AAAI: variables for figsize 3-->2.6
    fig, ax = plt.subplots(1, figsize=(5.55, 2.6))

    df_welfare['attack_level'] = df_welfare['attack_level'].apply(format_strategy)
    sns.pointplot(x='attack_level', y='welfare', hue=group, data=df_welfare,
                  palette=sns.color_palette(['#70AD47', '#437CDF', '#c55911']),
                  markers=["o", "s", "^"], hue_order=['high', 'all', 'low'], ax=ax)
    plt.ylabel('Average student welfare')
    plt.xlabel('Level of attack (in %) by schools (A, B)')

    plt.setp(ax.get_legend().get_texts(), fontsize='11')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='11')  # for legend title

    fig.tight_layout()
    plt.show()

    return df_welfare


def plot_best_response(file_name='best_response.csv'):
    '''Makes a plot in 2D with change in utility for the agents'''

    # read data
    results = read_best_response_data(file_name)

    fig, ax = plt.subplots(1, figsize=(5, 2.4))

    x = [res[0] for res in results]
    y = [res[1] for res in results]
    col = ['#437CDF', '#70AD47', 'black']
    c = [col[res[2]] for res in results]
    ax.scatter(x, y, color=c)
    import matplotlib.patches as mpatches
    s0 = mpatches.Patch(color='#437CDF', label='A responds')
    s1 = mpatches.Patch(color='#70AD47', label='B responds')
    ini = mpatches.Patch(color='black', label='Initial scenario')
    # for AAAI:
    ax.legend(handles=[ini, s0, s1], prop={'size': 11})
    # ax.legend(handles=[ini, s0, s1], loc='center left', bbox_to_anchor=(1, 0.5))

    # draw the arrows and annotations for welfare
    unit = 50  # for AAAI: this gives a unit of singificant change for annotations
    x = [results[0][0], results[1][0]]
    y = [results[0][1], results[1][1]]
    epsilon_x = abs(x[1] - x[0])/20
    epsilon_y = abs(y[1] - y[0])/20
    for i in range(0, len(results)):
        # draw the arrows between points
        c = col[results[i][2]]
        if i > 0:
            x = [results[i-1][0], results[i][0]]
            y = [results[i-1][1], results[i][1]]
            # plt.plot(x, y, color=col[results[i][2]])
            ax.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], length_includes_head=True,
                     head_width=epsilon_x*3, head_length=epsilon_y*3, fc=c, ec=c)

        # add annotations for the welfare of the students
        direction = i % 2 * 2 - 1
        if i == (len(results) - 1):
            direction = -1
        change_x = direction * epsilon_x
        change_y = - change_x
        change_x = change_x if change_x > 0 else 4*change_x
        procentage = "{:.0%}".format(results[i][4])
        # orange - #c55911
        if i == 0:
            ax.annotate(procentage, xy=(x[0] - change_x/7, y[0] + change_y), color=c)
        elif i == (len(results) - 1):
            ax.annotate(procentage, xy=(x[1] + change_x-3*unit, y[1] - 6*change_y-unit), color=c)
        else:
            if change_y > 0:
                change_y *= (2.3 + (len(results)-i)/len(results))
                change_x = change_x/len(results) + unit if i == 2 else change_x*i/len(results)
            else:
                change_y = change_y+unit if i == 1 else change_y-2*unit
            ax.annotate(procentage, xy=(x[1] + 1.2*change_x, y[1] +
                                        change_y), color=c, annotation_clip=False)

    # ax.set_title('Best response evolution')
    ax.set_xlabel('Utility of School A')
    ax.set_ylabel('Utility of School B')

    # change the axis limit to include the annotations
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - epsilon_x, xmax + epsilon_x*6)
    ax.set_ylim(ymin - epsilon_x, ymax + epsilon_x*6)

    fig.tight_layout()

    plt.show()

    x = (1, 2, 3)
    y = (1, 2, 3)
    # plt.plot(x, y, color='green', marker='o', linestyle='dashed')
    # plt.arrow(1.5, 1.5, 0.01, 0.015, shape='full', lw=0, length_includes_head=True, head_width=.05)
    # plt.show()


def plot_1c(sims, fixed_pars={}, no_years=10):
    '''Plots the cumulative utility by the interation number '''

    import ast

    stra = 'strategy'

    cols = ['utility', stra, 'seed', 'utility_by_round_str']
    df = {c: [] for c in cols}

    no_sims = 0
    for s in sims:
        # check to see if the fixed parameters have the desired value
        if np.array([s.config[i] == fixed_pars[i] for i in fixed_pars]).all():
            # load the results
            s.read()
            no_sims += 1

            # append the relevant data
            i_sch = s.config['no_schools'] - 1
            df['utility'].append(s.results['utility'][i_sch])
            df[stra].append(s.results['strategy'][i_sch])

            df['seed'].append(s.config['random_seed_no'])

            # add the utility by round
            to_add = 'utility_strategy_' + s.results['strategy'][i_sch]
            df['utility_by_round_str'].append(s.results[to_add][i_sch])

    df = pd.DataFrame(data=df)
    df = df.drop_duplicates()

    df['utility_by_round'] = df['utility_by_round_str'].apply(ast.literal_eval)
    def cumulative(l): return [sum(l[:(x+1)]) for x in range(len(l))]
    df['utility_until_round'] = df['utility_by_round'].apply(cumulative)

    # create a dataframe with utility per years on separate rows
    cols = ['utility', stra, 'seed', 'round']
    df_utility = {c: [] for c in cols}

    for index, row in df.iterrows():
        for round in range(no_years):
            df_utility['round'].append(round+1)
            df_utility[stra].append(row[stra])
            df_utility['seed'].append(row['seed'])
            df_utility['utility'].append(row['utility_until_round'][round])

    df_utility = pd.DataFrame(data=df_utility)

    # Rename variables
    strat = {'best_for_cheap': 'Strategic', 'best_for_student': 'Truthful'}
    for m in strat:
        df_utility['strategy'] = df_utility['strategy'].replace(m, strat[m])

    df_utility = df_utility.rename(columns={'utility': 'Average utility',
                                            'strategy': 'Interaction',
                                            'round': 'Year'})
    stra = 'Interaction'

    fig, ax = plt.subplots(1, figsize=(5, 3))

    sns.lineplot(data=df_utility, x='Year', y='Average utility', hue=stra, style=stra,
                 markers=True, dashes=False, ax=ax)
    ax.set_title(' ')
    fig.tight_layout()
    plt.show()
