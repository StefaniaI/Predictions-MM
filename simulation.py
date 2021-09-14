import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import re
import matplotlib.pyplot as plt
import os

import market as mk


class Simulation:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = {}
        self.results = []

    def get_sim_no(self):
        '''Gets the number of the simulation from the file_name'''

        numbers = re.findall(r'\d+', self.file_name)

        if len(numbers) == 0:
            return ''

        return numbers[0]

    def get_path(self):
        '''If the file_name also has a path to another folder, keeps that separate'''

        file_name_components = self.file_name.split('\\')

        if len(file_name_components) == 1:
            return ''

        return file_name_components[:-1]

    def read_config(self):
        ''' Reads the file with parameters, given its name.'''

        infile = open(self.file_name, mode='r')
        reader = csv.reader(infile)
        config = {rows[0]: rows[1] for rows in reader}

        # transforms each parameter into the correct type
        int_param_names = ['no_eval_attributes', 'scale_eval_attributes', 'no_schools',
                           'school_capacity', 'no_years_rs', 'no_known_schools',
                           'no_students', 'rec_size', 'random_seed_no', 'no_years', 'k']
        for name in int_param_names:
            config[name] = int(config[name])

        float_param_names = ['mu', 'sigma', 'student_obs_noise', 'alpha', 'misbehaviour_%',
                             'rs_obs_noise', 'recommendation_weight', 'cheap_bound']
        for name in float_param_names:
            config[name] = float(config[name])

        # This keeps a list of all the attribute combinations possible
        all_attr_types = mk.Attributes(config).get_all_attr_combinations()
        config['all_attr_types'] = all_attr_types

        # Gets the maximum evaluation score attainable
        max_eval_score = max([a.eval_score() for a in all_attr_types])
        config['max_eval_score'] = max_eval_score

        # for each attribute type, make a list with the attributes in the order of their distance
        config['atts_by_distance'] = {}
        for attr in all_attr_types:
            distance_attr = {}
            distances = []
            for other in all_attr_types:
                d = attr.distance(other)
                if d in distance_attr:
                    distance_attr[d].append(other)
                else:
                    distance_attr[d] = [other]
                    distances.append(d)
            # update config
            config['atts_by_distance'][attr] = []
            distances.sort()
            for d in distances:
                config['atts_by_distance'][attr].append(distance_attr[d])

        config['all_strategies'] = ['standard', 'best_for_student', 'best_for_cheap']

        self.config = config

        return config

    def simulate(self, given_strategies=[], read_config=True):
        '''Runs a simulation, for a given random seed, and writes the results in a corresponding file.
        given_strategies == [] --> use the market_structure given in the config file
           otherwise, use the list of given strategies.
        '''

        # Load the parameters from the file "config.csv"
        if read_config:
            config = self.read_config()
        else:
            config = self.config
        no_file = self.get_sim_no()

        # fix the random seed
        seed = config['random_seed_no']
        np.random.seed(seed)

        # create a market
        m = mk.Market(config)
        m.generate()
        # now set the strategies of the schools
        if len(given_strategies) == 0:
            strategy = config['market_structure'].split('-')
            for s in m.schools:
                s.strategy = strategy[1]
            if len(strategy) > 2:
                m.schools[-1].strategy = strategy[3]
        else:
            no_schools = len(m.schools)
            for i in range(no_schools):
                m.schools[i].strategy = given_strategies[i][0]
                m.schools[i].misbehaviour_amount = given_strategies[i][1] * config['max_eval_score']

        # find the number of iterations
        no_years = config['no_years']

        for i in range(no_years):
            m.iterate_once(i)

        # create a dataframe with the simulation results
        res_keys = ['attributes', 'strategy', 'utility', 'welfare', 'average_student']
        for a in config['all_attr_types']:
            res_keys.append('welfare' + a.id())
        res_keys += ['utility_strategy_'+str for str in m.schools[0].all_strategies]
        sim_results = {k: [] for k in res_keys}

        for s in m.schools:
            sim_results['attributes'].append(s.attr.eval)
            sim_results['strategy'].append(s.strategy)
            sim_results['utility'].append(s.utility)
            for str in s.all_strategies:
                sim_results['utility_strategy_'+str].append(s.stats['utility_strategy_'+str])
            sim_results['welfare'].append(s.stats['welfare'])
            sim_results['average_student'].append(s.stats['average_student'])
            for a in config['all_attr_types']:
                sim_results['welfare'+a.id()].append(s.stats['welfare'+a.id()])
        self.results = pd.DataFrame(data=sim_results)

        # print(self.results)
        # return sim_results

    def read(self, all=True):
        '''This takes the results and configuration for a simulation that was run before'''

        # read the config file
        self.read_config()

        # load the results into the simulation object
        if all:
            no = self.get_sim_no()
            path = self.get_path()
            self.results = pd.read_csv(os.path.join(*path, 'run' + str(no) + '.csv'))

    def plot_eval_attribute_distribution(self, n):
        '''Generates several attributes and makes a plot to show their distribution'''

        attrs = []
        for i in range(n):
            a = mk.Attributes(self.config)
            a.generate()
            attrs.append(a.eval_score())

        plt.hist(attrs)
        plt.show()


def run_sim(file_name="Simulation_results\\config.csv"):
    '''Runs the simulation for one config file. The resulting dataframe is saved into a .csv
    path = by default it saves the results into a separate folder;
           define path as the empty string if you want it to be in the saved in the same folder'''

    # Run a simulation with the given parameters
    sim = Simulation(file_name)
    sim.simulate()
    path = "Simulation_results"

    # Find the config file number
    no = sim.get_sim_no()

    # print(sim.results['attributes'])
    # print(sim.results['strategy'])
    # print(sim.results['utility'])

    # Save the results into the respective folder
    save_path = os.path.join(path, 'run' + str(no) + '.csv')
    sim.results.to_csv(save_path)


def run_sims(file_name_configs='Simulation_results\\file_names1.txt'):
    '''Runs multiple configXXXXX.csv files.
    file_name_configs = a folder with all the names of the config files that need to be runned.
    '''

    file_name_configs = os.path.join(*file_name_configs.split('\\'))
    file_names = open(file_name_configs, mode='r').readlines()
    for file_name in file_names:
        file_name = file_name.rstrip("\n\r")
        print(file_name)
        run_sim(file_name)

# ----------------------- Find equilibria --------------------------------


def find_equilibria(granularity_level=0.1, file_name="config.csv", no_seeds=20):
    '''Finds the equilibria (agents in turn choose the best response from str.).
    '''

    # define the seeds we'll use
    seeds = [[97], [9301], [1807], [2184], [4089], [2695], [3807], [1608], [2318], [3240],
             [2700], [9358], [3794], [8381], [962], [9570], [8114], [9371], [9455], [7397]]

    # Run a simulation with the given parameters
    sim = Simulation(file_name)
    sim.read_config()
    no_schools = sim.config['no_schools']

    # what will be captured
    cols = ['round', 'deciding_agent', 'strategy', 'seed', 'best_str', 'welfare']
    cols += ['utility_'+str(s) for s in range(no_schools)]
    cols += ['welfare'+a.id() for a in sim.config['all_attr_types']]
    df = {c: [] for c in cols}

    def save_df(df):
        no = sim.get_sim_no()
        df = pd.DataFrame(data=df)
        df.to_csv('best_response' + no + '.csv')

    # initially all schools do the best for the student
    misbehaviour = [0 for i in range(no_schools)]

    def convert_to_strategies(misbehaviour):
        '''Converts a vector of misbehaviour level values to pairs containing the name of the
        strategy.'''

        strat = []
        for i in range(no_schools):
            if misbehaviour[i] == 0:
                strat.append(('best_for_student', misbehaviour[i]))
            else:
                strat.append(('best_for_cheap', misbehaviour[i]))
        return strat

    misbehaviour_alternatives = list(np.arange(0, 1.00001, granularity_level))
    misbehaviour_alternatives = [np.round(l, 2) for l in misbehaviour_alternatives]

    # no equilibria might extit (agents could keep changing their strategies)
    # threfore, stop after a number of rounds
    max_no_rounds = len(misbehaviour_alternatives) * 3

    # schools have in turn a chance to deviate to another strategy
    round = 0
    somebody_deviated = True
    while somebody_deviated:
        somebody_deviated = False

        # take schools in order, check if they would benefit from deviating
        for i in range(no_schools):
            if round > max_no_rounds:
                save_df(df)
                return 0
            round += 1
            if round > max_no_rounds:
                return df
            # consider all alternative strategies
            aggregate_results = {}
            original_str = misbehaviour[i]

            for l in misbehaviour_alternatives:
                # change the behaviour of that agent
                misbehaviour[i] = l

                # take different random seeds
                utility = []
                for rand_seed in seeds[0:no_seeds]:
                    sim.config['random_seed_no'] = rand_seed
                    sim.simulate(given_strategies=convert_to_strategies(misbehaviour),
                                 read_config=False)
                    # add results
                    df['round'].append(round)
                    df['deciding_agent'].append(i)
                    df['strategy'].append(l)
                    for s in range(no_schools):
                        df['utility_'+str(s)].append(sim.results['utility'][s])
                    utility.append(sim.results['utility'][i])
                    df['seed'].append(rand_seed)
                    df['best_str'].append(False)
                    c = sim.config['school_capacity']
                    df['welfare'].append(np.mean([np.mean(sim.results['welfare'][s])/c
                                                  for s in range(no_schools)]))
                    # find the average welfare by the student type
                    for a in sim.config['all_attr_types']:
                        average_w = 0
                        no_st = 0
                        for s in range(no_schools):
                            welfare_a_s = sim.results['welfare'+a.id()][s]
                            sum_w_s = 0
                            no_st_s = 0
                            for w in welfare_a_s:
                                if w is not None:
                                    sum_w_s += w[0]*w[1]
                                    no_st_s += w[1]
                            average_w = average_w * no_st + sum_w_s
                            no_st += no_st_s
                            average_w /= no_st
                        df['welfare'+a.id()].append(average_w)

                # find the mean utility and standard deviation for this strategy
                aggregate_results[l] = (np.mean(utility), np.std(utility))

            # from the strategies that are significantlly better,
            # chose the one with the highest mean utility
            max_str = original_str
            max_utility = aggregate_results[original_str][0]
            for l in misbehaviour_alternatives:
                if (abs(aggregate_results[l][0] - aggregate_results[original_str][0]) >
                        aggregate_results[l][1] + aggregate_results[original_str][1]):
                    if max_utility < aggregate_results[l][0]:
                        max_utility = aggregate_results[l][0]
                        max_str = l

            # make max_str the best strategy in the results dataframe (df)
            for j in range(no_seeds*len(misbehaviour_alternatives)):
                if df['strategy'][-j] == max_str:
                    df['best_str'][-j] = True

            if max_str != original_str:
                somebody_deviated = True

            print(i, 'adopted strategy', max_str)
            misbehaviour[i] = max_str
    save_df(df)

# ----------------------- Creating virtual experimetns --------------------------------


def configs_ve():
    '''Does the entire simulations for the relevant plots (1a, 1b, 2)
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['no_eval_attributes'] = 1
    config['scale_eval_attributes'] = 5
    config['eval_criteria'] = 'sum'

    config['cheap_bound'] = -1  # meaning schools adapt their strategies

    config['no_known_schools'] = 1

    config['school_attribute_values'] = 'all_max'
    config['no_years'] = 100

    config['recommendation_strategy'] = 'KNN'
    config['student_obs_noise'] = 0.05

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['random_seed_no'], ['mu', 'sigma'], ['matching_algorithm'],
                          ['utility_composition'],
                          ['no_schools', 'school_capacity', 'no_students', 'rec_size'],
                          ['market_structure'], ['rs_obs_noise'], ['no_years_rs'],
                          ['recommendation_weight'], ['alpha'], ['misbehaviour_%'],
                          ['k']]

    changing[0] = [[97], [9301], [1807], [2184], [4089], [2695], [3807], [1608], [2318], [3240],
                   [2700], [9358], [3794], [8381], [962], [9570], [8114], [9371], [9455], [7397]]
    changing[1] = [[3.2, 3], [3.2, 0.65], [1, 3]]

    # changing[3] = [['SD-random'], ['Boston-random'], ['DA-random'], ['SD-by_result'],
    #                ['Boston-by_result'], ['DA-by_result']]
    changing[2] = [['SD-random'], ['Boston-random'], ['DA-random'], ['SD-by_result']]

    changing[3] = [['pos_min'], ['neg_min']]
    # [2, 20, 40, 2], [2, 5, 40, 2], [2, 80, 40, 2]
    changing[4] = [[10, 20, 200, 10], [10, 5, 200, 10], [10, 80, 200, 10]]
    changing[5] = [['all-best_for_student-one-best_for_cheap'],
                   ['all-best_for_student-one-best_for_student']]

    # changing[7] = [[0.0005], [0.05], [0.5], [1.5]]
    changing[6] = [[0.0005], [0.5], [1.5]]

    # changing[8] = [[1], [3], [5]]
    changing[7] = [[3]]

    changing[8] = [[1], [0.5]]

    # changing[10] = [[0.95], [0.5]]
    changing[9] = [[0.95]]

    # changing[11] = [[0.04], [0], [0.25], [0.5], [0.75], [1]]
    changing[10] = [[0.04]]

    # changing[12] = [[1], [3], [5]]
    changing[11] = [[3]]

    return config, changing, changing_var_lists


def generate_config_csvs(base_config, changing, var_lists, path='Simulation_results',
                         no_folders=1, start_config_no=-1):
    '''Creates the .csv files given the the fix parameters and the cahnging parameters.
    base_config = dictionary with the fixed parameters {parameter_name: parameter_val, ...}
    changing = dictionary with the changing parameters {[p1, p2]:[[v1, v2], [v1.0, v2.0]], ...}'''

    def write_parameters(dict_param, no_comb):
        ''' Writes a dictionary of parameters to a file indexed by no_comb.
        '''
        current_path = os.path.join(path, 'config' + str(no_comb) + '.csv')
        with open(current_path, 'w') as f:
            for key in dict_param.keys():
                f.write("%s,%s\n" % (key, dict_param[key]))

    config = {}
    no = start_config_no

    # put the base parameters into config
    for var in base_config:
        config[var] = base_config[var]

    # generate all combinations for the changing parameters
    choice_changing = [0 for i in var_lists]
    constructed_all = False

    while not constructed_all:
        # set all variables according to the current choice of parameter combinations
        for var_list_order_no in changing:
            var_list = var_lists[var_list_order_no]
            for var_pos in range(len(var_list)):
                var = var_list[var_pos]
                alternative_no = choice_changing[var_list_order_no]
                config[var] = changing[var_list_order_no][alternative_no][var_pos]

        # save the current config file
        no += 1
        write_parameters(config, no)

        # iterate twards the next choice_changing
        constructed_all = True
        for var_list_order_no in range(len(var_lists)):
            no_options = len(changing[var_list_order_no])

            # if we can increase
            if choice_changing[var_list_order_no] < (no_options - 1):
                # increase the choce option
                choice_changing[var_list_order_no] += 1
                # the ones before are reseted to 0
                for j in range(var_list_order_no):
                    choice_changing[j] = 0
                # we didn't construct all - there is a new choice
                constructed_all = False
                break

    no += 1

    # create no_folders folders with the names of the generated files
    for i in range(no_folders):
        current_path = os.path.join(path, "file_names" + str(i+1) + ".txt")
        f = open(current_path, mode='w')
        for j in range(start_config_no + 1, no):
            if j % no_folders == i:
                print_path = os.path.join(path, 'config' + str(j) + '.csv\n')
                f.writelines((print_path))

    return no
