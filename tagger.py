#!/usr/bin/env python

import sys
import numpy as np
from math import log10
from math import inf
import copy
import time
import pickle

class NGramModel:

    def __init__(self, n=2, laplace=0):
        self.__N = n
        self.__laplace_value = laplace

        self.__list_voc_states = []
        self.__list_voc_obs = []

        self.__start_obs_state = (-1, -1)
        self.__end_obs_state = (-2, -2)

        # these attributes are the parameters of the HMM model
        # each index i of the dicts consist of the i-gram frequencies/probabilities
        self.__state_transition_frequencies = {}
        self.__state_transition_probabilities = {}

        # Has the opposite, the matrix of emission frequencies/probabilities are directly stored in these attributes.
        self.__observation_emission_frequencies = {}
        self.__observation_emission_probabilities = {}

    def build_state_transition_frequency_matrix(self, depth):
        if depth < 1:
            # This should never happen
            exit("NGramModel.build_state_transition_frequency_matrix has been called with depth = 0. It should be > 0.")

        if depth == 1:
            # Exiting condition
            dict_state_transition_frequency_matrix = {}
            # each leef of the tree is initialized with the init_value
            for state in self.__list_voc_states:
                dict_state_transition_frequency_matrix[state] = 0

        else:
            # calculate lower-hierarchy branches
            dict_state_transition_frequency_matrix = self.build_state_transition_frequency_matrix(depth=depth-1)
            terminus = dict_state_transition_frequency_matrix
            dict_state_transition_frequency_matrix = {}
            # plug lower-hierarchy branches to the current level
            for state in self.__list_voc_states:
                dict_state_transition_frequency_matrix[state] = copy.deepcopy(terminus)

        return dict_state_transition_frequency_matrix

    def build_observation_emission_frequency_matrix(self):
        # The observation emission matrix is always of dimension 2
        for observation in self.__list_voc_obs:
            self.__observation_emission_frequencies[observation] = {}
            for state in self.__list_voc_states:
                self.__observation_emission_frequencies[observation][state] = 0

    def counter(self, sequence, sub_sequence_delimiter):
        sequence_index = 0
        length_sequence = len(sequence)

        delimiting_obs = sub_sequence_delimiter[0]

        # For each element of the corpus
        while sequence_index < length_sequence:
            # get the observable and the state at the current index
            (obs, state) = sequence[sequence_index]

            # boolean which say "are we currently processing a sub-sequence?"
            new_sequence = False

            # each subsequence will be processed separatly
            sub_sequence_states = []
            # each sub-sequence starts with N-1 (N being the N of N-gram) "junk states"
            for i in range(self.__N - 1):
                sub_sequence_states.append(self.__start_obs_state[1])

            while obs != delimiting_obs:
                new_sequence = True
                self.__observation_emission_frequencies[obs][state] += 1
                sub_sequence_states.append(state)
                sequence_index += 1
                (obs, state) = sequence[sequence_index]
            # each sub-sequence ends with 1 "end of sub-sequence state"
            sub_sequence_states.append(self.__end_obs_state[1])

            # if we were actually processing a real subsequence and not an artifact produced by two consecutive
            # sequence separator in the corpus
            if new_sequence:
                # count every state transition in the subsequence (1-gram transition are also considered here)
                self.count_state_transition_frequencies(sub_sequence_states)
                self.__observation_emission_frequencies[self.__start_obs_state[0]][self.__start_obs_state[1]] += 1 * (self.__N - 1)
                self.__observation_emission_frequencies[self.__end_obs_state[0]][self.__end_obs_state[1]] += 1

            sequence_index += 1

    def count_state_transition_frequencies(self, sequence):
        def increment_n_gram_frequency(state_transition_frequency, n_gram):
            # this function doesn't return anything because it works on the object level
            # any cleaner implementation is appreciated here
            if len(n_gram) == 1:
                # increment the state frequency counter given all the previous sates encountered
                state_transition_frequency[n_gram[0]] += 1
            else:
                # if the n_gram size is > 1, we are not at the lowest level of the tree. We have to keep "remembering"
                # the encountered states till we reach the last state of the N-gram
                increment_n_gram_frequency(state_transition_frequency[n_gram[-1]], n_gram[:-1])

        for n in range(self.__N):
            # we store the index in the sequence of the currently processed n_gram
            n_gram_indexes = range(n+1)
            length_sequence = len(sequence)

            # we stop processing till the last state of the n_gram has not reach the "end of sequence" state
            while n_gram_indexes[-1] < length_sequence:
                n_gram_states = [sequence[n] for n in n_gram_indexes]
                increment_n_gram_frequency(self.__state_transition_frequencies[n+1], n_gram_states)
                n_gram_indexes = list(map(lambda x: x+1, n_gram_indexes))

    def compute_state_transition_probabilities_matrix(self, n_gram_frequencies, n_minus_1_gram_frequencies, depth, first=False, smoothing=None):
        # remember the matrix of n-gram frequencies is formated like "frequency to get one state (level of hierarchy 1)
        # given the first before (level 2) ... given the n-1th before (level n-1)".
        # so when the depth is equals 1, it means that we have reached the last level of hierarchy (n-1) and the
        # returned value is an int instead of dict in n_minus_1_gram_frequencies[state] and n_gram_frequencies[state]
        if depth == 1:
            tmp_dict = {}
            if n_minus_1_gram_frequencies == {}:
                for state in self.__list_voc_states:
                    n_minus_1_gram_frequencies[state] = len(self.__list_voc_states)
            for state in self.__list_voc_states:
                tmp_dict[state] = -log10((n_gram_frequencies[state] + self.__laplace_value) /
                                         (n_minus_1_gram_frequencies[state] + (len(self.__list_voc_states) * self.__laplace_value)))
            return tmp_dict

        # if the depth is >= 2, then the returned value by n_gram_frequencies[state] is an other dict (a branch) which
        # we have to walk in order to get the frequencies (leefs)
        elif depth >= 2:
            tmp_dict = {}
            for state1 in self.__list_voc_states:
                # the first call of this recursive function (in the 'train' method) has an alternate behavior:
                # in order to stick to the definition of the n-gram probabilities, we want to get the frequency of the
                # n-1-gram at time t-1: look at this example:
                #
                # we are looking for the probability of the 3-gram "abc". p("abc") = freq("abc") / freq("ab")
                # -> at first call, we start walking through the 3-gram_frequency_matrix by entering in level "a"
                #    but we don't start walking through the 2-gram_frequency matrix
                # -> at second call, we start walking through both trees to get:
                #    - "c" given "b" given "a" frequency of the 3-gram
                #    - "b" given "a" frequency of the 2-gram at time t-1
                if first:
                    tmp_dict[state1] = self.compute_state_transition_probabilities_matrix(n_gram_frequencies[state1],
                                                                                          n_minus_1_gram_frequencies,
                                                                                          depth=depth-1)
                else:
                    tmp_dict[state1] = self.compute_state_transition_probabilities_matrix(n_gram_frequencies[state1],
                                                                                          n_minus_1_gram_frequencies[state1],
                                                                                          depth=depth-1)
            return tmp_dict

    def ngram_training(self, corpus, list_voc_obs, list_voc_states, sub_sequence_delimiter=(0, 0)):

        if len(self.__list_voc_obs) == 0:
            self.__list_voc_obs = list(list_voc_obs) + [self.__start_obs_state[0]] + [self.__end_obs_state[0]]

        if len(self.__list_voc_states) == 0:
            self.__list_voc_states = list(list_voc_states) + [self.__start_obs_state[1]] + [self.__end_obs_state[1]]

        # setup the frequencies matrix before processing the corpus in order to prevent any "key-missing" error
        # for each n, build the frequency matrix associated with the n-grams
        for i in range(self.__N):
            self.__state_transition_frequencies[i+1] = self.build_state_transition_frequency_matrix(
                depth=i+1)
        self.build_observation_emission_frequency_matrix()

        # actual count of transition and emission frequencies
        self.counter(corpus, sub_sequence_delimiter)

        # for each n, fill the frequency matrix associated with the n-grams
        for i in range(self.__N):
            # the probability to observe an n-gram at position t is the frequency of this n-gram divided by the
            # frequency of the n-1 gram at state t-1... more presitions in the
            # compute_state_transition_probabilities_matrix method
            n_gram_frequencies = self.__state_transition_frequencies[i+1]
            if i > 0:
                n_minus_1_gram_frequencies = self.__state_transition_frequencies[i]
            else:
                n_minus_1_gram_frequencies = {}
            self.__state_transition_probabilities[i+1] = \
                self.compute_state_transition_probabilities_matrix(n_gram_frequencies, n_minus_1_gram_frequencies, depth=i+1, first=True)

        self.compute_emission_probabilities()
        return

    def viterbir(self, sequence_obs):
        def init_pi_matrix(voc_states, depth, start_state, visited_states):
            """
            Initialize the score matrix for the position '0' of the sequence which will always be associated with
            special 'start' state.

            This is a recursive function: the matrix is of dimension N (N being the size of the N-grams)

            :param voc_states: The list of possible state
            :param depth: The depth of the pi matrix.
            :param start_state: The code of the 'start state"
            :param visited_states: The value of the matrix are set one by one. They all take 'inf' value but the value
            of the N-gram 'start start start' at position 0.

            :return: The initialized matrix.
            """
            if depth == 1:
                # at depth == 1, we set the score
                tmp_dict = {}
                only_start_in_visited_states = True

                for s in visited_states:
                    if s!= start_state:
                        only_start_in_visited_states = False
                        break
                for u in voc_states:
                    # the only box with the best value is the one of the N-gram 'start start start' which means that all
                    # visited states are 'start'
                    if only_start_in_visited_states and u == start_state:
                        tmp_dict[u] = 0
                    else:
                        tmp_dict[u] = inf
                return tmp_dict

            if depth >= 2:
                # if depth >= 2, we have to go deeper in the score matrix
                tmp_dict = {}
                for v in voc_states:
                    tmp_dict[v] = init_pi_matrix(voc_states, depth=depth-1, start_state=start_state, visited_states=tuple([v] + [s for s in visited_states]))
                return tmp_dict

        def get_score_viterbi(visited_states, score_viterbi):
            """
            Get the score of a specific N-gram.

            The position in the word should be already set when entering this function.

            :param visited_states: The N-gram: last visited states should be at the end of the sequence.
            :param score_viterbi: The viterbi scores of a given position.

            :return: The score of the N-gram
            """
            if len(visited_states) == 1:
                return score_viterbi[visited_states[0]]
            else:
                return get_score_viterbi(visited_states[:-1], score_viterbi[visited_states[-1]])

        def fill_pi_backtrack(pi, bp, visited_states, best_score, best_label):
            """
            Set the score to observe the N-gram of visited_states at the current observation.

            :param pi: The scoring matrix at the current observation.
            :param bp: The backpointer matrix at the current observation.
            :param visited_states: The N-gram
            :param best_score: The best obtained score for this N-gram
            :param best_label: The best_label which has given this best score.
            """
            if len(visited_states) == 1:
                pi[visited_states[0]] = best_score
                bp[visited_states[0]] = best_label
            else:
                if visited_states[-1] not in pi:
                    pi[visited_states[-1]] = {}
                if visited_states[-1] not in bp:
                    bp[visited_states[-1]] = {}
                fill_pi_backtrack(pi[visited_states[-1]], bp[visited_states[-1]], visited_states[:-1], best_score, best_label)

        def compute_best_pi_value(state_transition_probabilities, emission_probability, score_viterbi_last_obs,
                                  pi, bp, visited_states,
                                  voc_state, depth):
            """
            Compute best pi value for a given state and position of the sequence.

            This function is not atomic since it computes the best pi-value then store it it in the scoring matrix
            and the backpointer matrix. (better implementations appreciated here)

            :param state_transition_probabilities: The transition probability, given the visited states.
            :param emission_probability:
            :param score_viterbi_last_obs: The score in the pi_matrix for the last observable.
            :param pi: The score matrix at the current observation. Will be filled by "fill_pi_backtrack' function.
            :param bp: The backpointer matrix at the current observation. Will be filled by "fill_pi_backtrack' function
            :param visited_states: The states of the N-gram.
            :param voc_state:
            :param depth:
            """
            if depth == 1:
                best_label = None
                best_score = None
                for w in voc_state:
                    score_viterbi_last_obs_for_visited_states = get_score_viterbi(list(visited_states + [w])[1:],
                                                                                  score_viterbi_last_obs)
                    tmp_score = state_transition_probabilities[w] + \
                        emission_probability + \
                        score_viterbi_last_obs_for_visited_states

                    if best_score is None or tmp_score < best_score:
                        best_score = tmp_score
                        best_label = w

                fill_pi_backtrack(pi, bp, visited_states, best_score, best_label)

            elif depth >= 2:
                for v in voc_state:
                    compute_best_pi_value(
                        state_transition_probabilities[v],
                        emission_probability,
                        score_viterbi_last_obs,
                        pi,
                        bp,
                        visited_states + [v],
                        voc_state,
                        depth=depth - 1)


        def init_output(pi, state_transition_probabilities, voc_states, depth):
            """
            Get the states used to initialize output.

            :param pi: The scores at a given position. Usually, the last position of the sequence is used.
            :param state_transition_probabilities: The state transition probability. Usually initialized with the first
            state is the 'end state' special state.
            :param voc_states: The list of possible states.
            :param depth:
            :return: tuple(best_score, [best_labels])
            """
            if depth == 1:
                best_v_score = None
                best_v = None
                for v in voc_states:
                    v_score = pi[v] + state_transition_probabilities[v]
                    if best_v_score is None or v_score < best_v_score:
                        best_v_score = v_score
                        best_v = v
                return (best_v_score, [best_v])
            else:
                best_scores_labels = []
                for u in voc_states:
                    result = init_output(pi[u], state_transition_probabilities[u], voc_states, depth=depth-1)
                    best_scores_labels.extend([(result[0], [u] + result[1])])
                the_best_score = None
                the_best_labels = None
                for score in best_scores_labels:
                    if the_best_score is None or score[0] < the_best_score:
                        the_best_score = score[0]
                        the_best_labels = score[1]
                return (the_best_score, the_best_labels)

        def get_output_for_k(bp, k, output, depth):
            """
            Return the (k-1)th output.
            """
            if depth == 1:
                return bp[output[k]]
            else:
                return get_output_for_k(bp[output[k]], k+1, output, depth=depth-1)

        # pi is the matrix that will store the score for each position of the sequence and each state
        pi = []
        pi.append(init_pi_matrix(self.__list_voc_states, self.__N - 1, start_state=self.__start_obs_state[1], visited_states=[]))

        # bp is the backpointer which will be used to get back the states from the calculated scores
        bp = []
        bp.append({})

        # the actual observable sequence to compute
        sequence_obs = list(sequence_obs)
        n = len(sequence_obs)

        # a reformated version of the observable sequence in order to make it consistent with the score matrix indexes.
        formated_sequence_obs = [self.__start_obs_state[0]] + sequence_obs + [self.__start_obs_state[0]]
        n_formated = len(formated_sequence_obs)
        # the k only browse the actual obs_sequence
        for k in range(1, n_formated - 1):
            pi.append({})
            bp.append({})
            # u is the state associated with the obs
            for u in self.__list_voc_states:
                # compute the score for this u in terms of:
                # - The state transition probabilities leading to this u
                # - The observation emission probability for this u to generate this observable
                # - The scores obtained at the last position of the word (k-1)
                # - pi[k] and bp[k] are the scores and backpointers values to set
                # - [u] will be used for the setting of pi[k] and bp[k]
                # - again, the list of possible states and the depth are some 'utilitaries' for the recursive function
                compute_best_pi_value(self.__state_transition_probabilities[self.__N][u],
                                      self.__observation_emission_probabilities[formated_sequence_obs[k]][u],
                                      pi[k-1],
                                      pi[k],
                                      bp[k],
                                      [u],
                                      self.__list_voc_states,
                                      depth=self.__N-1)

        output = [None] * n
        best_score_labels = list(init_output(pi[n],
                                             self.__state_transition_probabilities[self.__N][self.__end_obs_state[1]],
                                             self.__list_voc_states, depth=self.__N -1)[1])
        # first, set the last N-1 output
        k = n
        while k > 0 and len(best_score_labels) > 0:
            output[k-1] = best_score_labels[0]
            best_score_labels = best_score_labels[1:]
            k -= 1

        while k > 0:
            output[k - 1] = get_output_for_k(bp[k+(self.__N-1)], k, output, depth=self.__N - 1)
            k -= 1

        return output

    def viterbi2(self, sequence_obs):
        pi = []
        pi.append({})

        for u in self.__list_voc_states:
            pi[0][u] = inf

        pi[0][self.__start_obs_state[1]] = 0

        bp = []
        bp.append({})

        sequence_obs = list(sequence_obs)
        n = len(sequence_obs)

        formated_sequence_obs = [self.__start_obs_state[0]] + sequence_obs + [self.__start_obs_state[0]]
        n_formated = len(formated_sequence_obs)
        for k in range(1, n_formated - 1):
            pi.append({})
            bp.append({})
            for u in self.__list_voc_states:
                best_pi_value = None
                best_third_gram = None
                for v in self.__list_voc_states:
                    pi_value = pi[k-1][v] + \
                               self.__state_transition_probabilities[2][u][v] + \
                               self.__observation_emission_probabilities[formated_sequence_obs[k]][u]
                    if best_pi_value is None or pi_value < best_pi_value:
                        best_pi_value = pi_value
                        best_third_gram = v
                pi[k][u] = best_pi_value
                bp[k][u] = best_third_gram

        output = [None] * n

        best_u = None
        best_u_score = inf
        for u in self.__list_voc_states:
            u_score = pi[n][u] + \
                      self.__state_transition_probabilities[2][self.__end_obs_state[1]][u]
            if u_score < best_u_score:
                best_u_score = u_score
                best_u = u
        output[n-1] = best_u

        k = n-1
        while k > 0:
            output[k-1] = bp[k+1][output[k]]
            k -= 1
        return output

    def viterbi3(self, sequence_obs):
        pi = []
        pi.append({})

        for u in self.__list_voc_states:
            pi[0][u] = {}
            for v in self.__list_voc_states:
                pi[0][u][v] = inf

        for u in self.__list_voc_states:
            pi[0][u][self.__start_obs_state[1]] = inf

        pi[0][self.__start_obs_state[1]][self.__start_obs_state[1]] = 0

        bp = []
        bp.append({})

        n = len(sequence_obs)

        formated_sequence_obs = [self.__start_obs_state[0]] + sequence_obs + [self.__start_obs_state[0]]
        n_formated = len(formated_sequence_obs)
        for k in range(1, n_formated - 1):
            pi.append({})
            bp.append({})
            for u in self.__list_voc_states:
                for v in self.__list_voc_states:
                    best_pi_value = None
                    best_third_gram = None
                    for w in self.__list_voc_states:
                        pi_value = pi[k-1][w][v] + \
                                   self.__state_transition_probabilities[3][u][v][w] + \
                                   self.__observation_emission_probabilities[formated_sequence_obs[k]][u]
                        if best_pi_value is None or pi_value < best_pi_value:
                            best_pi_value = pi_value
                            best_third_gram = w
                    if v not in pi[k]:
                        pi[k][v] = {}
                    pi[k][v][u] = best_pi_value
                    if v not in bp[k]:
                        bp[k][v] = {}
                    bp[k][v][u] = best_third_gram

        output = [None] * n

        best_u = None
        best_v = None
        best_uv_score = inf
        for u in self.__list_voc_states:
            for v in self.__list_voc_states:
                uv_score = pi[n][u][v] + \
                           self.__state_transition_probabilities[3][self.__end_obs_state[1]][u][v]
                if uv_score < best_uv_score:
                    best_uv_score = uv_score
                    best_u = u
                    best_v = v
        output[n-1] = best_u
        output[n-2] = best_v

        k = n-2
        while k > 0:
            output[k-1] = bp[k+2][output[k]][output[k+1]]
            k -= 1
        return output

    def process_observable_sequence(self, sequence_obs, computing_function, sub_sequence_delimiter=(0, 0)):
        sequence_obs = list(sequence_obs)

        output = []
        i = 0
        start = time.clock()
        while i < len(sequence_obs):

            sub_sequence = []
            while i < len(sequence_obs) and sequence_obs[i] != sub_sequence_delimiter[0]:
                if i % 10000 == 0:
                    print("%s ième mot: %.2fs" % (i, (time.clock() - start)))
                sub_sequence.append(sequence_obs[i])
                i += 1
            if bool(sub_sequence):
                output.extend(computing_function(sub_sequence))

            if i < len(sequence_obs) and sequence_obs[i] == sub_sequence_delimiter[0]:
                output.append(sub_sequence_delimiter[1])
            i += 1

        return output

    def test(self, corpus):
        sequence_obs = corpus[:, 0]
        sequence_tags = corpus[:, 1]

        result = self.process_observable_sequence(sequence_obs, self.viterbir)

        print("Calcul du résultat terminé.")
        length_result = len(result)
        i = 0
        error_count = 0
        while i < length_result:
            if result[i] != sequence_tags[i]:
                error_count += 1
            i += 1

        print("%.2f %% d'erreurs pour n=%s" % ((float(error_count) / float(length_result) * float(100)), self.__N))

    def compute_emission_probabilities(self, smoothing=None):
        for observable in self.__list_voc_obs:
            self.__observation_emission_probabilities[observable] = {}
            for state in self.__list_voc_states:

                self.__observation_emission_probabilities[observable][state] = \
                    -log10((self.__observation_emission_frequencies[observable][state] + self.__laplace_value) /
                           (self.__state_transition_frequencies[1][state] + (len(self.__list_voc_obs) * self.__laplace_value)))

def open_encoded_corpus_obs_state(s_filename):
    """
    Open the specified file and return the array containing its informations.

    Each line of the input file should contain 2 members splitted by a spacing character.
    Sequence should be separated by one line. If there is multiple blank lines between sequence, only one will be taken
    into account.

    :param s_filename: The path to the input file.
    :return: Array os shape (n, 2) n is the number of lines of the file.
    """
    f = None
    try:
        f = open(s_filename, 'r')
    except FileNotFoundError:
        exit("The specified corpus: " + str(s_filename) + "does not exist.")
    list_tuples_obs_state = []
    s_line = f.readline()
    while s_line != "":
        s_stripped_line = s_line.strip()
        if s_stripped_line != "":
            tuple_obs_state = tuple(map(int, s_stripped_line.split()))
            list_tuples_obs_state.append(tuple_obs_state)
        else:
            # if the line is blank, it means it is the end of a sequence
            if list_tuples_obs_state[-1] != tuple((0, 0)):
                # I don't want to consider multiple blank lines
                tuple_obs_state = tuple((0, 0))
                list_tuples_obs_state.append(tuple_obs_state)
        s_line = f.readline()
    return np.array(list_tuples_obs_state)

def open_vocabulary(s_filename):
    """
    Open a vocabulary file and return the list of words.

    Each line of the vocabulary file should contain a word.

    :param s_filename: The path to the vocabulary file
    :return: list of words
    """
    f = None
    try:
        f = open(s_filename, 'r')
    except FileNotFoundError:
        exit("The specified corpus: " + str(s_filename) + "does not exist.")
    # code inspired from the encoding script of Carlos Ramisch encode.py
    dict_decode_encode = {}
    dict_encode_decode = {}
    k = 1
    for line in f.readlines():
        stripped_line = line.strip()
        if stripped_line != "":
            dict_decode_encode[stripped_line] = k
            dict_encode_decode[k] = stripped_line
            k += 1
    return dict_decode_encode, dict_encode_decode

def main():
    pass

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

if __name__ == "__main__":
    # main()
    start = time.clock()
    n=2
    print("Démarage pour n=%s." % n)

    try:
        new = load_object('dump.save' + str(n))
        print("Modèle chargé.")
    except FileNotFoundError:
        new = NGramModel(n=n, laplace=1)
        path_corpus_train = sys.argv[1]
        path_voc_obs = sys.argv[2]
        path_voc_state = sys.argv[3]
        dict_obs_decode_encode, dict_obs_encode_decode = open_vocabulary(path_voc_obs)
        dict_state_decode_encode, dict_state_encode_decode = open_vocabulary(path_voc_state)

        new.ngram_training(open_encoded_corpus_obs_state(path_corpus_train), dict_obs_encode_decode.keys(), dict_state_encode_decode.keys())
        print("Entrainement du modèle terminé.")
        save_object(new, 'dump.save' + str(n))

    path_corpus_test = sys.argv[4]
    new.test(open_encoded_corpus_obs_state(path_corpus_test))

    print("L'execution pour n=%s a duré %.4fs." % (n, (time.clock() - start)))