import csv
import numpy as np
from copy import deepcopy
import scipy.stats as stat

# if the debug is on, you can see agent 0 thought the simulation
debug = False
# shows intermediate prints for the matching algorithm (and RS)
debug_matching = False


class Attributes:
    def __init__(self, config):
        # Evaluation attributes
        self.eval = np.zeros(config['no_eval_attributes'])

        self.config = config

    def __eq__(self, other):
        ''' Two attributes are eqaul if all of their entries are.'''

        if isinstance(other, Attributes):
            return (self.eval == other.eval).all()
        else:
            return False

    def __hash__(self):
        max_rating = self.config['scale_eval_attributes'] + 1

        enc = 0

        for i in range(len(self.eval)):
            enc += self.eval[i]*(max_rating**i)

        return hash(enc)

    def id(self):
        enc = ''

        for i in range(len(self.eval)):
            enc += '-' + str(int(self.eval[i]))

        return enc

    def is_above(self, attr):
        '''Checks if all evaluation entries are above the ones in attr.'''

        return (self.eval >= attr.eval).all()

    def is_below(self, attr):
        '''Checks if all evaluation entries are below the ones in attr.'''

        return (self.eval <= attr.eval).all()

    def is_closer(self, to_attr, other_attr):
        '''Checks that self is closer to the to_attr than other_attr is to to_attr.'''

        distance_to_self = (abs(to_attr.eval - self.eval)).sum()
        distance_to_other = (abs(to_attr.eval - other_attr.eval)).sum()
        return distance_to_self <= distance_to_other

    def max_distance_below(self, other_attr):
        '''Finds the how much below self goes on the evaluation criteria from other_attr. '''

        return -min(min(self.eval - other_attr.eval), 0)

    def distance(self, other_attr):
        '''Finds the euclidean distance between two attributes '''

        return np.linalg.norm(self.eval - other_attr.eval)

    def improving_potential(self, school_attr):
        '''Finds how much can the school improve the student
        - if a student with attribute self
        - is at a school with attributes school_attr'''

        distance = 0
        for i in range(len(self.eval)):
            if self.eval[i] < school_attr.eval[i]:
                distance += (school_attr.eval[i] - self.eval[i])

        return distance

    def generate(self):
        '''Generates random attributes to agents.
        They are normally distributed, and rounded to the nearest integer on the scale.'''

        # Generate the random values
        self.eval = np.random.normal(
            self.config['mu'], self.config['sigma'], self.config['no_eval_attributes'])

        # Make a function that converts the values to a rating
        max_rating = self.config['scale_eval_attributes']

        def to_rating(n):
            '''Converts float values to a rating.'''
            n = round(n)
            if n < 0:
                n = 0
            if n > max_rating:
                n = max_rating
            return n

        # Change the values to be according to the rating
        self.eval = np.array([to_rating(n) for n in self.eval])

    def eval_score(self):
        ''' The evaluation policy. How do the evaluation criteria get aggregated.'''

        if self.config['eval_criteria'] == 'sum':
            return sum(self.eval)

    def get_all_attr_combinations(self):
        '''Returns a list with all possible attribute combinations.'''

        no_attr = self.config['no_eval_attributes']
        max_rating = self.config['scale_eval_attributes']

        all_comb = []

        i = no_attr - 1
        while(i >= 0):
            if i == no_attr - 1:
                all_comb.append(deepcopy(self))
            if self.eval[i] < max_rating:
                self.eval[i] += 1
                if i != no_attr - 1:
                    i = i+1
            else:
                self.eval[i] = 0
                i = i-1

        return all_comb


class Student:
    def __init__(self, config):
        # the level of knowledge on each evaluation criteria (before being assigned to a school)
        self.attr = Attributes(config)
        # Initial preference (by weights) on schools
        self.pref = {}

        self.config = config

    def investigate_school(self, prestige):
        ''' The observed value of a school, by the student, after investigating it.
        prestige = true prestige of the school
        '''

        noise = self.config["student_obs_noise"]
        score = prestige + (np.random.random()*2-1)*noise

        # we add some extra small noise to differentiate between possible equal values
        epsilon = np.random.random() * min(noise/10, 0.000001)

        return min(max(score, 0), self.config['max_eval_score']) + epsilon

    def generate(self, known_schools):
        ''' Generates attributes for students and initialises preferences.
        known_schools = dictionary {school_index: prestige_of_school}'''

        self.attr.generate()

        # the preference of students is given by the prestige of the school + noise
        # the noise is unif. at rand. in (-noise, +noise)
        for school_index in known_schools:
            self.pref[school_index] = self.investigate_school(known_schools[school_index])

    def update_pref(self, rec_schools, schools):
        ''' Updates the preferences of students after seeing the recommendation.
        rec_schools = recommended schools = {shool_index: expected level at graduation (grade)}
           it includes all the schools in st.pref
        schools = list of all schools
        '''

        # for each recommended school
        for school_index in rec_schools:
            # student goes and investigates that school: their observation is based on prestige
            self.pref[school_index] = self.investigate_school(schools[school_index].prestige)

            if debug_matching:
                print("for school", school_index, "student", self.attr. eval, "RS",
                      rec_schools[school_index], "observed prestige", self.pref[school_index],
                      'true prestige', schools[school_index].prestige)

            # the observation is infuenced by the RS
            rec_weight = self.config['recommendation_weight']
            self.pref[school_index] = rec_weight * rec_schools[school_index] + \
                (1-rec_weight) * self.pref[school_index]

    def eval_score(self):
        return self.attr.eval_score()


class School:
    def __init__(self, config):
        self.attr = Attributes(config)
        self.size = 0
        self.strategy = 'standard'
        self.misbehaviour_amount = config['misbehaviour_%'] * config['max_eval_score']

        # the prestige of a school given by the average evaluation score of alumna in the past year
        self.prestige = 0

        # the level of fitting for each type of student (given by average score of alumna of that
        # type) -- the history of the school
        self.history = {}

        # the utility received so far by the school
        self.utility = 0

        self.config = config
        self.cheap_bound = config['cheap_bound']
        self.all_strategies = config['all_strategies']

        # statstics for the school: contribution to welfare by year, utility for each strategy
        self.stats = {('utility_strategy_' + strat): [] for strat in self.all_strategies}
        self.stats['welfare'] = []
        self.stats['average_student'] = []
        for a in config['all_attr_types']:
            self.stats['welfare'+a.id()] = []

    def generate(self):
        ''' Generates attribute values for the school, and gives it capacity. '''

        if self.config['school_attribute_values'] == 'all_max':
            self.attr.eval = np.ones(self.config['no_eval_attributes']) * \
                self.config['scale_eval_attributes']
        elif self.config['school_attribute_values'] == 'random':
            self.attr.generate()

        self.size = self.config['school_capacity']

        # to generate the pre-RS history we assume each school receives each year random students
        # according to their capacity. The school does the best for the student
        self.prestige = self.attr.eval_score()
        no_years = self.config['no_years_rs']
        for i in range(no_years):
            self.history[i] = {}
            # generate attributes of school_capacity many students
            for j in range(self.size):
                st_attr = Attributes(self.config)
                st_attr.generate()
                # the school does the best for each of them
                val_and_cost = self.decision(st_attr)

                # add observation for this history
                if st_attr in self.history[i]:
                    self.history[i][st_attr][1] += 1
                else:
                    self.history[i][st_attr] = [val_and_cost['best_for_student'][0], 1]

        # now, we define the best_for_cheap strategy (depending on the value of cheap_bound
        # -1 --> allocation based on distribution, other_value --> that value)
        mu = self.config['mu']
        sigma = self.config['sigma']
        if self.cheap_bound == -1:
            # 1) find the distribution of student evaluation attributes & find the distance
            attr_pr_dist = {}
            max_dist = 0
            for a in self.config['all_attr_types']:
                # find the probability
                pr = 1
                for e in a.eval:
                    if e == 0:
                        pr *= stat.norm.cdf(0.5, mu, sigma)
                    elif e == self.config['scale_eval_attributes']:
                        pr *= 1 - stat.norm.cdf(e - 0.5, mu, sigma)
                    else:
                        pr *= stat.norm.cdf(e + 0.5, mu, sigma) - stat.norm.cdf(e - 0.5, mu, sigma)

                # find the distance
                d = int(a.improving_potential(self.attr))
                if d > max_dist:
                    max_dist = d

                # save to dictionary
                attr_pr_dist[a] = (pr, d)

            # 2) find the distribution of distance
            dist_pr = {}
            for d in range(max_dist + 1):
                dist_pr[d] = 0
                for a in attr_pr_dist:
                    if attr_pr_dist[a][1] == d:
                        dist_pr[d] += attr_pr_dist[a][0]

            # 3) find the percentage of students the school wants to get
            # if the utility is positive --> as many a possible, else --> as few as possible
            if self.config['utility_composition'][0:3] == 'pos':
                wanted = min(self.size, self.config['no_students'])
            else:
                wanted = max(0, self.config['no_students'] - (self.config['no_schools'] - 1)
                             * self.size)
                wanted = min(wanted, self.size)
            # a student that is attracted to them could, in theory go to any school; so, aim or more
            wanted *= self.config['no_schools'] / (2 * self.config['no_students'])
            # wanted /= self.config['no_students']

            # 4) find the cheap bound for that percentage of students
            pr_so_far = 0
            for d in range(max_dist + 1):
                if pr_so_far < wanted:
                    pr_so_far += dist_pr[d]
                    self.cheap_bound = d
                else:
                    break

    def eval_score(self):
        return self.attr.eval_score()

    def decision(self, student_attributes, return_utility_if_best=False):
        '''The school decides how it treats a certain student under different strategies.
        Returns the evaluation score for the student after graduation and the encountered cost
           for each strategy.
        Alteratively, this function can be used to find the utility of the school if it treats the
           student in the best way (if return_utility_if_best = True).
        '''

        end_attributes = Attributes(self.config)
        val_and_cost = {}

        # the utility composition is pos/neg + _ + strategy for standard
        offset = 0
        if self.config['utility_composition'][0:3] == 'neg':
            offset = self.config['no_eval_attributes'] * self.config['scale_eval_attributes']
        standard_utility = self.config['utility_composition'][4:]

        # 1 - for the 'standard' strategy
        # the student gets to the average/min of their level and the school's, 0 cost for the school
        def reset_standard_attributes():
            if standard_utility == 'average':
                end_attributes.eval = (student_attributes.eval + self.attr.eval)/2
            elif standard_utility == 'min':
                end_attributes.eval = np.minimum(student_attributes.eval, self.attr.eval)

        reset_standard_attributes()
        val_and_cost['standard'] = (end_attributes.eval_score(), offset)
        # print('standard:', self.attr.eval, student_attributes.eval, val_and_cost)

        # 2 - for the 'best_for_student' strategy
        # the student gets to the best the school can bring it to
        # the cost for the school is its effort = alpha*(the improvement of student)
        cost = 0
        for i in range(len(self.attr.eval)):
            if student_attributes.eval[i] < self.attr.eval[i]:
                cost += (self.attr.eval[i] - end_attributes.eval[i])*self.config['alpha']
                end_attributes.eval[i] = self.attr.eval[i]
        val_and_cost['best_for_student'] = (end_attributes.eval_score(), cost + offset)

        # print('best_for_student:', student_attributes.eval, val_and_cost)
        # input()
        if return_utility_if_best:
            return val_and_cost['best_for_student'][0] - val_and_cost['best_for_student'][1]

        # 3 - for the 'best_for_cheap' starategy
        # the student is treated as in 'best_for_student' if the cost is small
        # otherwise, they are left at standard
        if student_attributes.improving_potential(self.attr) <= self.cheap_bound:
            val_and_cost['best_for_cheap'] = val_and_cost['best_for_student']
        else:
            # do almost best
            reset_standard_attributes()
            cost = 0
            for i in range(len(self.attr.eval)):
                if student_attributes.eval[i] < (self.attr.eval[i]-self.misbehaviour_amount):
                    cost += (self.attr.eval[i] - self.misbehaviour_amount -
                             end_attributes.eval[i])*self.config['alpha']
                    end_attributes.eval[i] = self.attr.eval[i]-self.misbehaviour_amount
            val_and_cost['best_for_cheap'] = (end_attributes.eval_score(), cost + offset)
            # val_and_cost['best_for_cheap'] = val_and_cost['standard']

        # print(student_attributes.eval, self.attr.eval, end_attributes.eval, cost)
        return val_and_cost

    def new_school_year(self, student_types, year_no):
        '''School interacts with the students it was allocated with.
        The history gets updated (for year_no). Same for the utility of the school.
        students_types - the types allocated {type_student: no_allocated_students_of_that_type}
        year_no = is the year number modulo the number of years important for the RS'''

        utility = {str: 0 for str in self.all_strategies}
        self.history[year_no] = {}

        sum_results = 0
        no_students = 0
        average_type = np.zeros(len(self.attr.eval))

        # complete the statistics on the average student welfare by type
        for a in self.config['all_attr_types']:
            self.stats['welfare'+a.id()].append(None)

        for attr in student_types:
            dec = self.decision(attr)

            for s in self.all_strategies:
                value, cost = dec[s]
                utility[s] += (value - cost)*student_types[attr]

                # for the strategy that is the current strategy of the school:
                # the history and the prestige updates
                if self.strategy == s:
                    self.history[year_no][attr] = [value, student_types[attr]]
                    # for the prestige & welfare, keep track of results and number of students
                    sum_results += value*student_types[attr]
                    # add to statistics the welfare of the students of that type
                    self.stats['welfare'+attr.id()][-1] = (value, student_types[attr])

            no_students += student_types[attr]
            average_type += attr.eval*student_types[attr]

            # print(self.attr.eval)
            # print(attr.eval, student_types[attr])
            # print(utility, average_type)
            # input()

        # update utility and prestige
        if debug_matching:
            print(utility)
        self.utility += utility[self.strategy]
        self.prestige = sum_results/no_students

        # print(self.utility, self.prestige)

        # save statistics
        # 1 - record utility depending by starategy
        for strat in self.all_strategies:
            self.stats['utility_strategy_' + strat].append(utility[strat])

        # 2 - record welfare
        self.stats['welfare'].append(sum_results)

        # 3 - record average student type
        self.stats['average_student'].append(average_type/no_students)

        # print(self.stats)
        # input()


class Market:
    def __init__(self, config):
        self.students = []
        self.schools = []
        self.future_students = []

        # how many years the market has run with a RS
        self.age = 0

        # the chance of each school to be known to a student
        self.known_density = []

        self.config = config

    def update_known_density(self):
        ''' Based on the prestige of the schools, creates a density for the
        Pr(student knows each school)'''

        # The chance of each school to be known is given by its prestige
        # Good schools are exponentially as likely to be known to students
        def weight_fn(x): return 2 ** ((x/self.config['max_eval_score'])*10)

        self.known_density = [weight_fn(s.prestige) for s in self.schools]

        # transform to probability density function
        self.known_density /= sum(self.known_density)

    def generate_schools(self):
        ''' The procedure that generates the initial schools and their knowledge chance'''

        no = self.config['no_schools']
        for i in range(no):
            s = School(self.config)
            s.generate()
            self.schools.append(s)

        self.update_known_density()

        if debug:
            print('The attributes of each school and the prestige:')
            for s in self.schools:
                print(s.attr.eval, s.prestige)

    def generate_students_one_year(self):
        '''Generates a new wave of students (for a new matching round).'''

        self.students = []
        no_known_schools = self.config['no_known_schools']

        for i in range(self.config['no_students']):
            st = Student(self.config)

            # get the schools known by this student
            # pick indicies at random from the distribution
            known_schools_indices = np.random.choice(list(range(len(self.schools))),
                                                     no_known_schools, replace=False,
                                                     p=self.known_density)
            # record the school index and its prestige
            known_schools = {}
            for j in known_schools_indices:
                known_schools[j] = self.schools[j].prestige
            st.generate(known_schools)
            self.students.append(st)

            if debug and (i == 0):
                print('Attributes of i are: ', st.attr.eval)
                print('They initially know the following schools:')
                for j in st.pref:
                    print('They have pref.', st.pref[j], ' for school ',
                          j, ' with prestige ', self.schools[j].prestige)
                input()

    def generate_students(self):
        '''Generates students for all years.
        These will be kept in 'self.future_students', and will be taken from there in each new year.
        It ensures that for the same seed, the same students will arrive.
        '''

        no_years = self.config['no_years']
        for i in range(no_years):
            self.generate_students_one_year()
            self.future_students.append(self.students)

        self.students = []

    def generate(self):
        '''Generates schools and future students'''

        self.generate_schools()
        self.generate_students()

    def recommendation(self, st_attr, st_pref, year_no):
        '''This implements the recommender system.
        Given a student's attributes (st_attr) it returns a list of recommended school-indices.
        The recommender system also gives advice on the already known schools (st_pref)
        year_no = the current year --> year_no+1(mod no_years_rs) is the oldest'''

        rs = self.config['recommendation_strategy']
        # input()
        # print(st_attr.eval)

        def find_average_outcome(school):
            '''Finds the average expected outcome (grade) based on the history of the school.'''

            # print(school.strategy)
            if rs == 'default':
                # find the closest student in history
                # in case of equality, take the most recent data
                oldest_year = year_no + 1
                no_years_rs = self.config['no_years_rs']
                closest_attr = -1
                closest_eval = -1
                for i in range(no_years_rs):
                    year = (i+oldest_year) % no_years_rs
                    for attr in school.history[year].keys():
                        if closest_attr == -1 or attr.is_closer(st_attr, closest_attr):
                            closest_attr = attr
                            closest_eval = school.history[year][attr][0]

                # if evaluation of entry critera is lower, decrease by more than the noise
                # --> students get rather recommended school with exact/close match
                if closest_attr.eval_score() > st_attr.eval_score():
                    closest_eval -= self.config['rs_obs_noise']
                    # closest_eval -= 2*self.config['rs_obs_noise'] * \
                    #    (closest_attr.eval_score() - st_attr.eval_score())

                if debug_matching:
                    print("For school ", school.attr.eval, " and student ",
                          st_attr.eval, " the closest is ", closest_attr.eval, closest_eval)
                return closest_eval

            elif rs == 'perfect_info':
                # in this case, the recommender has perfect information about the future behaviour
                # this is the equivalent of an oracle

                # return the outcome the school will take
                val_cost = school.decision(st_attr)[school.strategy]
                return val_cost[0]

            elif rs in ['KNN', 'KNN+']:
                # choses the k nearest neighbours and estimates the average

                optimised = True if rs == 'KNN+' else False
                k = self.config['k']
                ordered_atts = self.config['atts_by_distance'][st_attr]
                no_years_rs = self.config['no_years_rs']

                no_nbhs = 0
                average = 0
                current_distance = -1
                while no_nbhs < k:
                    current_distance += 1
                    if current_distance >= len(ordered_atts):
                        break
                    current_nbhs_outcome = []
                    for attr_other in ordered_atts[current_distance]:
                        for year in range(no_years_rs):
                            if attr_other in school.history[year]:
                                no_students = school.history[year][attr_other][1]

                                val = school.history[year][attr_other][0]
                                # optimised version remove the noise if the attributes are lower
                                if optimised and attr_other.eval_score() > st_attr.eval_score():
                                    val -= self.config['rs_obs_noise']

                                for i in range(no_students):
                                    current_nbhs_outcome.append(val)

                    # if all the current neighbours will not get the total over k
                    # then add all
                    current_no = len(current_nbhs_outcome)
                    if current_no > 0:
                        if current_no <= (k-no_nbhs):
                            # average changes
                            average = average*no_nbhs + sum(current_nbhs_outcome)
                            no_nbhs += current_no
                            average /= no_nbhs
                        else:
                            # choose randomlly the rest to complete the number
                            chosen = np.random.choice(current_nbhs_outcome, k-no_nbhs)

                            # update the average
                            average = average*no_nbhs + sum(chosen)
                            no_nbhs = k
                            average /= no_nbhs

                if debug_matching:
                    print("For school ", school.attr.eval, " and student ",
                          st_attr.eval, " RS predicted score is ", average, current_nbhs_outcome)

                return average

            elif rs == 'default_old':
                outcome = 0
                no_years_records = 0

                # for each year in the history
                for i in range(self.config['no_years_rs']):
                    # if the student attribute is in the history for that year - add it to outcome
                    if st_attr in school.history[i]:
                        outcome += school.history[i][st_attr][0]
                        no_years_records += 1

                # if there are no records for that school
                if no_years_records == 0:
                    # print("Attributes not in history")
                    # try to find attribute combinations in the history closest (below and above)
                    str_attr_close = {'below': -1, 'above': -1,
                                      'below_outcome': -1, 'above_outcome': -1}
                    max_outcome = -1
                    for i in range(self.config['no_years_rs']):
                        for a in school.history[i].keys():
                            if st_attr.is_above(a):
                                if str_attr_close['below'] == -1 or a.is_closer(
                                        st_attr, str_attr_close['below']):
                                    str_attr_close['below'] = a
                                    str_attr_close['below_outcome'] = school.history[i][a][0]
                            elif st_attr.is_below(a):
                                if str_attr_close['above'] == -1 or a.is_closer(
                                        st_attr, str_attr_close['above']):
                                    str_attr_close['above'] = a
                                    str_attr_close['above_outcome'] = school.history[i][a][0]
                            if school.history[i][a] > max_outcome:
                                max_outcome = school.history[i][a][0]
                    # if we found below --> estimate from that
                    if str_attr_close['below_outcome'] != -1:
                        # if there is also an above one --> outcome = mean between the two
                        # otherwise, the above_outcome is the maximum one
                        if str_attr_close['above_outcome'] == -1:
                            str_attr_close['above_outcome'] = max_outcome
                        # print("Found a below_outocme:", str_attr_close['below'].eval, (
                        #    str_attr_close['below_outcome'] + str_attr_close['above_outcome'])/2)
                        return (str_attr_close['below_outcome'] + str_attr_close['above_outcome'])/2

                    # if there is no such records --> average (prestige - student value)
                    # print("Attribute not in history. No below.",
                    #      (school.prestige + st_attr.eval_score())/2)
                    return (school.prestige + st_attr.eval_score())/2
                else:
                    # print("Attribute in history", outcome/no_years_records)
                    return outcome/no_years_records

        # Find the suitability of each school for the student
        noise = self.config['rs_obs_noise']

        def add_noise(x):
            score = x + (np.random.random()*2-1)*noise
            # add differentiation between potentially equal values
            epsilon = np.random.random() * min(noise/10, 0.000001)
            return min(max(score, 0), self.config['max_eval_score']) + epsilon
        average_outcome_by_school = np.array([add_noise(find_average_outcome(
            s)) for s in self.schools])

        # Find the top schools to recommend to the students
        rec_size = self.config['rec_size']
        if rec_size < len(self.schools):
            indicies_recommended_schools = list(np.argpartition(
                -average_outcome_by_school, rec_size)[: rec_size])
        else:
            indicies_recommended_schools = list(range(len(self.schools)))

        # add the indices of schools already known by the student
        indicies_recommended_schools += st_pref.keys()

        return {s: average_outcome_by_school[s] for s in indicies_recommended_schools}

    def matching_alg(self):
        '''Uses the preferences of students to produce a matching.'''

        # available places at each school - initially this is the capacity of the school
        free_places = {}
        for j in range(len(self.schools)):
            free_places[j] = self.schools[j].size

        # allocation: {student: assigned school}
        allocation = {}

        # mechanism and the order
        mechanism_comp = self.config['matching_algorithm'].split('-')
        mechanism = mechanism_comp[0]
        if len(mechanism_comp) > 1:
            mechanism_ordring = mechanism_comp[1]

        if mechanism == 'SD':
            # the matching alg is random searial dectatorship for stundents
            no_students = len(self.students)
            if mechanism_ordring == 'random':
                order = np.random.permutation(no_students)
            elif mechanism_ordring == 'by_result':
                students_with_score = [(i, self.students[i].attr.eval_score()) for i in range(
                    len(self.students))]
                order = [pair[0] for pair in sorted(students_with_score, key=lambda x: -x[1])]

            # takes the students in the order given by the random permutation
            for i in order:
                pref_i = self.students[i].pref
                # assignes the student at the first school that still has a place
                top_school_index = -1
                top_school_value = -1

                for j in pref_i:
                    if free_places[j] > 0:
                        if top_school_value < pref_i[j]:
                            top_school_value = pref_i[j]
                            top_school_index = j
                        elif top_school_value == pref_i[j] and j > top_school_index:
                            top_school_value = pref_i[j]
                            top_school_index = j

                allocation[i] = top_school_index
                if top_school_index != -1:
                    free_places[top_school_index] -= 1

                if debug:
                    print(pref_i)
                    print('The student', self.students[i].attr.eval,
                          ' has lottery number ', list(order).index(i))
                    print('They were allocated school ', allocation[i])
                    input()

        elif mechanism in ['Boston', 'DA']:
            # The matching alg is Boston or DA

            # 1) get the lotteries at each school
            no_students = len(self.students)
            students_indices = list(range(no_students))
            if mechanism_ordring == 'random':
                order = {s_id: np.random.permutation(students_indices) for s_id in free_places}
                order_with_st_no = {s_id: {order[s_id][i]: i for i in range(no_students)}
                                    for s_id in free_places}
            elif mechanism_ordring == 'by_result':
                students_with_score = [(i, self.students[i].attr.eval_score()) for i in range(
                    len(self.students))]
                univ_order = [pair[0] for pair in sorted(students_with_score, key=lambda x: -x[1])]
                order = {s_id: univ_order for s_id in free_places}
                order_with_st_no = {s_id: {order[s_id][i]: i for i in range(no_students)}
                                    for s_id in free_places}
            elif mechanism_ordring == 'by_utility':
                order = {}
                for s_id in free_places:
                    st_by_utility = [(i, self.schools[s_id].decision(self.students[i].attr, True))
                                     for i in range(no_students)]
                    order[s_id] = [pair[0] for pair in sorted(st_by_utility, key=lambda x: -x[1])]
                order_with_st_no = {s_id: {order[s_id][i]: i for i in range(no_students)}
                                    for s_id in free_places}

            # 2) get the preference ranking of students over the schools
            if debug_matching:
                print(self.schools[0].utility, self.schools[-1].utility)
                print(order)
                input()
            prefs = {}
            for i in students_indices:
                pref_i = self.students[i].pref
                prefs[i] = [k for k, v in sorted(pref_i.items(), key=lambda item: -item[1])]
                if debug_matching:
                    print(i, self.students[i].attr.eval, pref_i)
            if debug_matching:
                print(prefs)
                input()

            # 3) Boston: in each round k the remaining students apply to their kth choice
            if mechanism == 'Boston':
                unassigned_students = students_indices
                round_no = -1
                while len(unassigned_students):
                    round_no += 1
                    if debug_matching:
                        print(round_no)

                    # which students apply to which schools this round
                    assignment_this_round = {s_id: [] for s_id in free_places}
                    for i in unassigned_students:
                        # i applies to their k-th school
                        if len(prefs[i]) > round_no:
                            s_id = prefs[i][round_no]  # the school that i applies to
                            # the ranking of i in the pref of s_id
                            rank_i = order_with_st_no[s_id][i]
                            assignment_this_round[s_id].append((i, rank_i))
                        else:
                            # if there are no more options in the pref of students, then they can no
                            #    longer be assigned
                            unassigned_students.remove(i)

                    # each school accepts the top ordered students according to school's prefs
                    for s_id in free_places:
                        if free_places[s_id] and len(assignment_this_round[s_id]):
                            # can the school accept all?
                            if free_places[s_id] >= len(assignment_this_round[s_id]):
                                # all students get allocated to the school
                                for i, rank in assignment_this_round[s_id]:
                                    allocation[i] = s_id
                                    free_places[s_id] -= 1
                                    # the student is no longer unassigned
                                    unassigned_students.remove(i)
                                    if debug_matching:
                                        print(i, ' to ', s_id)
                            else:
                                # accept the students until places are filled
                                index_accepted = np.argpartition(
                                    [el[1] for el in assignment_this_round[s_id]], free_places[
                                        s_id])[:free_places[s_id]]
                                accepted = [assignment_this_round[s_id][i][0]
                                            for i in index_accepted]
                                for i in accepted:
                                    allocation[i] = s_id
                                    free_places[s_id] -= 1
                                    # the student is no longer unassigned
                                    unassigned_students.remove(i)
                                    if debug_matching:
                                        print(i, ' to ', s_id)
                        if debug_matching:
                            input()
            elif mechanism == 'DA':
                unassigned_students = students_indices
                current_assignment = {s_id: [] for s_id in free_places}
                round_no = -1

                while len(unassigned_students):
                    round_no += 1
                    if debug_matching:
                        print(round_no)

                    # which students apply to which schools this round
                    for i in unassigned_students:
                        # i applies to the first school they did not yet apply
                        if len(prefs[i]):
                            s_id = prefs[i][0]  # the school that i applies to
                            # the ranking of i in the pref of s_id
                            rank_i = order_with_st_no[s_id][i]
                            current_assignment[s_id].append((i, rank_i))
                            prefs[i].pop(0)

                    # each school tentatively accepts the top ordered students according to
                    #   school's prefs
                    unassigned_students = []
                    for s_id in free_places:
                        if len(current_assignment[s_id]):
                            # can the school accept all?
                            if self.schools[s_id].size >= len(current_assignment[s_id]):
                                # all students get temporarily allocated to the school
                                if debug_matching:
                                    for i, rank in current_assignment[s_id]:
                                        print(i, ' temporarily to ', s_id)
                            else:
                                # temporarily accept the students until places are filled
                                no_places_s = self.schools[s_id].size
                                index_partitioned = np.argpartition(
                                    [el[1] for el in current_assignment[s_id]], no_places_s)
                                index_accepted = index_partitioned[:no_places_s]
                                index_rejected = index_partitioned[no_places_s:]
                                accepted = [current_assignment[s_id][i]
                                            for i in index_accepted]
                                rejected = [current_assignment[s_id][i]
                                            for i in index_rejected]

                                # the temporarily accepted students are allocated
                                current_assignment[s_id] = []
                                for i in accepted:
                                    current_assignment[s_id].append(i)
                                    if debug_matching:
                                        print(i[0], ' temporarily to ', s_id)

                                # the rejected studetns are unassigned and can choose in another
                                for i in rejected:
                                    unassigned_students.append(i[0])
                                    if debug_matching:
                                        print(i[0], ' rejected ')

                        if debug_matching:
                            input()

                # the final current assignmetn is the allocation
                for s_id in current_assignment:
                    for i in current_assignment[s_id]:
                        allocation[i[0]] = s_id
                if debug_matching:
                    print('The allocation is:', allocation)

        elif self.config['matching_algorithm'] == 'Boston_same_lottery':
            # The matching alg is Boston with schools preferring students with same lottery
            students_indices = list(range(len(self.students)))
            order = np.random.permutation(students_indices)

            # order the preferences of students by value
            if debug_matching:
                print(self.schools[0].utility, self.schools[1].utility)
                input()
            prefs = {}
            for i in order:
                pref_i = self.students[i].pref
                prefs[i] = [k for k, v in sorted(pref_i.items(), key=lambda item: -item[1])]
                if debug_matching:
                    print(i, self.students[i].attr.eval, pref_i)
            if debug_matching:
                print(prefs)
                input()

            round = 0
            while len(order) > 0:
                # students get their 1st choices in the given order
                new_order = []  # keeps unmatched students
                round += 1
                if debug_matching:
                    print(round)
                for i in order:
                    # try to assign at most preferred school
                    if free_places[prefs[i][0]] > 0:
                        allocation[i] = prefs[i][0]
                        free_places[prefs[i][0]] -= 1
                        if debug_matching:
                            print(i, ' to ', prefs[i][0])
                    else:
                        prefs[i].pop(0)
                        if len(prefs[i]) > 0:
                            new_order.append(i)
                order = new_order
                if debug_matching:
                    input()

        return allocation

    def iterate_once(self, year_no):
        '''Carries out one year (i.e. one round of iteration).
        year_no = the year number of this iteration (useful for updating the history)
        1) New students are comming
        2) Each student gets a personalised recommendation
        3) The matching is carried out
        4) Schools act (using their strategies) --> added utility & history gets updated
        5) Update density for knowing schools'''

        # ------ #1 generate students -----
        self.students = self.future_students[year_no]

        # ------ #2 get recommendations for students ------
        for st in self.students:
            rec = self.recommendation(st.attr, st.pref, year_no)
            # students update their preference based on recommendations
            st.update_pref(rec, self.schools)

            if debug and (st == self.students[0]):
                print('The RS suggested for ', st.attr.eval, ': ')
                for j in rec:
                    print('School ', j, ' with expected result ', rec[j])
                input()
                print('The student with ', st.attr.eval, ' updates his preferences to: ')
                for j in st.pref:
                    print('The preference for school', j, ' is ', st.pref[j])
                input()

        # ------ #3 students report their preferences & matching is carried out ------
        allocation = self.matching_alg()

        # keep a dictionary {school_no: {type_student: no_allocated_students_of_that_type} }
        allocated_students = {}
        for j in range(len(self.schools)):
            allocated_students[j] = {}

        for i in allocation:
            # the school number i was allocated to
            school_i = allocation[i]
            # if student no. i was allocated to a school
            if school_i != -1:
                # the attributes of i
                attr_i = self.students[i].attr

                if attr_i in allocated_students[school_i]:
                    allocated_students[school_i][attr_i] += 1
                else:
                    allocated_students[school_i][attr_i] = 1

        # ------ #4 a new year - schools and students interact -------
        # this new year will replace the history of the one 'no_years_rs' years before
        year_no = year_no % self.config['no_years_rs']
        for j in allocated_students:
            if allocated_students[j] != {}:
                self.schools[j].new_school_year(allocated_students[j], year_no)
            else:
                for k in self.schools[j].stats:
                    self.schools[j].stats[k].append(-1)

        # ------- #5 update the density function with gives the pr. of a school to be known ------
        self.update_known_density()
