#!/usr/bin/env python

import sys
import numpy as np
from math import log10
import copy


class MorphoSyntacticTagger:

    def __init__(self):
        self.__list_voc_states = []
        self.__list_voc_obs = []

        # these attributes will be filled according to the content of the actual corpus and shouldn't be altered
        self.__state_frequencies = {}
        self.__state_transition_frequencies = {}
        self.__observation_emission_frequencies = {}

        # these attributes are the parameters of the HMM model
        self.__state_transition_probabilities = {}
        self.__observation_emission_probabilities = {}
        self.__state_pi_probabilities = {}

    def build_state_frequency_matrix(self, init_value=0):
        for state in self.__list_voc_states:
            self.__state_frequencies[state] = init_value

    def build_state_transition_frequency_matrix(self, depth, init_value=0):
        if depth == 1:
            for state in self.__list_voc_states:
                self.__state_transition_frequencies[state] = init_value

        elif depth > 1:
            self.build_state_transition_frequency_matrix(depth=depth-1)
            terminus = self.__state_transition_frequencies
            self.__state_transition_frequencies = {}
            for state in self.__list_voc_states:
                self.__state_transition_frequencies[state] = copy.deepcopy(terminus)
        return

    def build_observation_emission_frequency_matrix(self, init_value=0):
        for observation in self.__list_voc_obs:
            self.__observation_emission_frequencies[observation] = {}
            for state in self.__list_voc_states:
                self.__observation_emission_frequencies[observation][state] = init_value

    def count_observation_emission_frequencies(self, tuple_sequence):
        for obs_state in tuple_sequence:
            self.__observation_emission_frequencies[obs_state[0]][obs_state[1]] += 1

    def count_state_transition_frequencies(self, sequence, N):
        def increment_n_gram_frequency(state_transition_frequency, n_gram):
            if len(n_gram) == 1:
                state_transition_frequency[n_gram[0]] += 1
            else:
                increment_n_gram_frequency(state_transition_frequency[n_gram[-1]], n_gram[:-1])

        n_gram_indexes = range(N)

        length_sequence = len(sequence)
        while n_gram_indexes[-1] < length_sequence:
            increment_n_gram_frequency(self.__state_transition_frequencies, [sequence[n] for n in n_gram_indexes])
            a = [sequence[n] for n in n_gram_indexes]
            n_gram_indexes = list(map(lambda x: x+1, n_gram_indexes))

    def count_state_frequencies(self, sequence):
        for state in sequence:
            self.__state_frequencies[state] += 1

    def hmm_training(self, corpus, N=2, smoothing=None):

        if len(self.__list_voc_obs) == 0:
            self.set_voc_obs(list(set(corpus[:, 0])))

        if len(self.__list_voc_states) == 0:
            self.set_voc_states(list(set(corpus[:, 1])))

        # setup the frequencies matrix before processing the corpus in order to prevent any "key-missing" error
        self.build_state_transition_frequency_matrix(depth=N)
        self.build_state_frequency_matrix()
        self.build_observation_emission_frequency_matrix()

        # counting state frequencies and transition frequencies could have been done at once but I wanted to keep
        # the code clear
        self.count_state_frequencies(corpus[:, 1])
        self.count_state_transition_frequencies(corpus[:, 1], N)
        self.count_observation_emission_frequencies(corpus)



        self.compute_state_transition_probabilities_matrix(self.__state_transition_frequencies, N)
        c = 0
        for state1 in self.__state_transition_probabilities:
            for state2 in self.__state_transition_probabilities[state1]:
                # for state3 in self.__state_transition_probabilities[state1][state2]:
                if state2 == 8:
                    c += self.__state_transition_probabilities[state1][state2]
        exit()

        if smoothing == "Laplace":
            # Laplace smoothing consists on adding a weight to each N-gram probability.
            # todo: at the moment, this weight is 1, it might be interesting to parametrize this
            pass



    def predict(self):
        pass

    def compute_state_transition_probabilities_matrix(self, state_transition_frequencies, depth):
        """
        Return the matrix of transitions as -log probabilities between token formatted as a dict of dict.

          - The first level of hierarchy is the token at position t (qt)
          - The second is the token at position t-1 (qt-1)
          - The value is -log(P(qt|qt-1))

        :param token_sequence: the sequence of tokens
        :return: the matrix
        """

        if depth == 1:
            tmp_dict = {}
            for state in self.__list_voc_states:
                tmp_dict[state] = state_transition_frequencies[state] / self.__state_frequencies[state]
            return tmp_dict

        elif depth > 1:
            for state1 in self.__list_voc_states:
                self.__state_transition_probabilities[state1] = \
                    self.compute_state_transition_probabilities_matrix(state_transition_frequencies[state1], depth=depth-1)
            return self.__state_transition_probabilities

        # def transition_prob(dict_token_occurences, dict_dict_transition_occurences):
        #     dict_dict_transition_log_probs = {}
        #
        #     for token in dict_dict_transition_occurences:
        #         dict_dict_transition_log_probs[token] = {}
        #         for last_token in dict_dict_transition_occurences[token]:
        #             dict_dict_transition_log_probs[token][last_token] = \
        #                 -log10(dict_dict_transition_occurences[token][last_token] / dict_token_occurences[last_token])
        #             # todo a tester -> ca a l'air ok
        #
        #     return dict_dict_transition_log_probs
        #
        # for state in self.__state_transition_frequencies:

    def compute_emission_probabilities(self):


        pass

    def compute_pi_probabilities(self):


        pass

    def set_voc_states(self, list_voc_states):
        self.__list_voc_states = list_voc_states

    def set_voc_obs(self, list_voc_obs):
        self.__list_voc_obs = list_voc_obs

def open_encoded_corpus_obs_state(s_filename):
    """
    Open the specified file and return the array containing its informations.

    Each line of the input file should contain 2 members splitted by a spacing character.

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


def emission_prob(token_couples, dict_state_occurences, dict_dict_emission_occurences):
    """
    Return the matrix of emissions as -log probabilities between token formatted as a dict of dict.

      - The first level of hierarchy is the observable at position t
      - The second is the state at position t
      - The value is -log(P(observable|state))

    :param token_couples: the sequence of couples observable/state
    :return: the matrix
    """
    for couple in token_couples:
        observable = couple[0]
        state = couple[1]
        if state not in dict_state_occurences:
            dict_state_occurences[state] = 1
        else:
            dict_state_occurences[state] += 1

        if observable not in dict_dict_emission_occurences:
            dict_dict_emission_occurences[observable] = {}

        if state not in dict_dict_emission_occurences[observable]:
            dict_dict_emission_occurences[observable][state] = 1
        else:
            dict_dict_emission_occurences[observable][state] += 1

    dict_dict_emission_log_probs = {}
    for observable in dict_dict_emission_occurences:
        dict_dict_emission_log_probs[observable] = {}
        for state in dict_dict_emission_occurences[observable]:
            dict_dict_emission_log_probs[observable][state] = \
                -log10(dict_dict_emission_occurences[observable][state] / dict_state_occurences[state])

    return dict_dict_emission_log_probs


def pi_prob(token_sequence, delimiter, dict_starting_token_occurence):
    int_nb_starts = 0
    start = True
    for token in token_sequence:
        if token == delimiter:
            start = True
            continue
        elif start == False:
            continue
        else:
            start = False
            int_nb_starts += 1
            if token not in dict_starting_token_occurence:
                dict_starting_token_occurence[token] = 1
            else:
                dict_starting_token_occurence[token] += 1

    dict_pi = {}
    for token in dict_starting_token_occurence:
        dict_pi[token] = -log10(dict_starting_token_occurence[token] / int_nb_starts)

    return dict_pi


def build_two_elm_combinatorial(token_list1, token_list2, init_value=0):
    combinatorial = {}
    for token1 in token_list1:
        combinatorial[token1] = {}
        for token2 in token_list2:
            combinatorial[token1][token2] = init_value
    return combinatorial


def viterbi(sequence_obs, labels, transition_matrix, emission_matrix, pi_probs):

    # for p in pi_probs:
    #     print(str(p) + "," + str(pi_probs[p]))
    #
    # for l in labels:
    #     print("," + str(l), end="")
    # print()
    # for k in labels:
    #     print(str(k), end="")
    #     for l in labels:
    #         print("," + str(transition_matrix[k][l]), end="")
    #     print()
    # print()
    # for i in emission_matrix:
    #     print(i, end=",")

    backtrack = {}
    sequence_obs_size = len(sequence_obs)
    score = [{}]

    # initialization
    for y in labels:
        score[0][y] = pi_probs[y] + emission_matrix[sequence_obs[0]][y]

    # for each observable in the sequence
    for i in range(1, sequence_obs_size):
        # setup the score dict for this observable
        score.append({})
        # we'll calculate the score for each possible label
        for y_current in labels:
            best_label_score = None
            best_label = None
            # the score is calculated in term of the label of the last position
            for y_last in labels:
                # for each label at position i-1, we calculate the score at position i
                score_y_last = score[i-1][y_last]
                score_transition = transition_matrix[y_current][y_last]
                score_emission = emission_matrix[sequence_obs[i]][y_current]
                y_last_score = score_y_last + score_transition + score_emission
                # we are in log probs so we have to min the score
                if best_label_score is None or y_last_score < best_label_score:
                    best_label = y_last
                    best_label_score = y_last_score

            # store the best score for each position of the observable sequence and each possible label
            score[i][y_current] = best_label_score
            # store the label of the position i-1 which gave the best score for position i
            if i not in backtrack:
                backtrack[i] = {}
            backtrack[i][y_current] = best_label

    i = sequence_obs_size - 1
    output = [None] * sequence_obs_size
    best_last_score = min(score[i].values())
    y = None
    for label, score in score[i].items():
        if score == best_last_score:
            y = label
            output[i] = y
            break

    while i > 0:
        output[i-1] = backtrack[i][y]
        y = backtrack[i][y]
        i -= 1
    return output

def test(corpus, dict_encode_state, transition_probabilities, emission_probabilities, pi_probabilities):
    sequence_obs = corpus[:, 0]
    sequence_tags = corpus[:, 1]

    result = viterbi(sequence_obs,
        dict_encode_state.keys(),
        transition_probabilities,
        emission_probabilities,
        pi_probabilities)

    length_result = len(result)
    i = 0
    error_count = 0
    while i < length_result:
        if result[i] != sequence_tags[i]:
            error_count += 1
        i += 1

    print("%.2f %% d'erreurs" % (float(error_count)/float(length_result)*float(100)))

def main():
    path_corpus_train = sys.argv[1]
    path_corpus_test = sys.argv[4]
    path_voc_observable = sys.argv[2]
    path_voc_states = sys.argv[3]

    dict_decode_obs, dict_encode_obs = open_vocabulary(path_voc_observable)
    dict_decode_state, dict_encode_state = open_vocabulary(path_voc_states)

    # lissage de laplace + 1 au compte du bigramme / + V au compte de l'unigramme

    transition_occurence_matrix_init = build_two_elm_combinatorial(dict_encode_state.keys(),
                                                                   dict_encode_state.keys(),
                                                                   init_value=1)
    state_occurence_init = {}
    length_voc_state = len(dict_encode_state)
    for token in dict_encode_state.keys():
        state_occurence_init[token] = length_voc_state

    transition_probabilities = transition_prob(open_encoded_corpus_obs_state(path_corpus_train)[:, 1],
                                               state_occurence_init,
                                               transition_occurence_matrix_init)

    state_occurence_init = {}
    length_voc_state_plus_length_voc_lex = len(dict_encode_obs)
    for token in dict_encode_state.keys():
        state_occurence_init[token] = length_voc_state_plus_length_voc_lex

    emission_occurence_matrix_init = build_two_elm_combinatorial(dict_encode_obs.keys(),
                                                                 dict_encode_state.keys(),
                                                                 init_value=1)

    emission_probabilities = emission_prob(open_encoded_corpus_obs_state(path_corpus_train),
                                           state_occurence_init,
                                           emission_occurence_matrix_init)

    state_occurence_init = {}
    length_voc_state = len(dict_encode_state)
    for token in dict_encode_state.keys():
        state_occurence_init[token] = length_voc_state

    pi_probabilities = pi_prob(open_encoded_corpus_obs_state(path_corpus_train)[:, 1],
                               dict_decode_state["PONCT"],
                               state_occurence_init)
    #
    # print("\n".join([dict_encode_state[i] for i in viterbi([dict_decode_obs[i] for i in['Une',
    #                                                                                     'regrettable',
    #                                                                                     'erreur',
    #                                                                                     'nous',
    #                                                                                     'a',
    #                                                                                     'fait',
    #                                                                                     'écrire']],
    #                dict_encode_state.keys(),
    #                transition_probabilities,
    #                emission_probabilities,
    #                pi_probabilities)]))

    # test(open_encoded_corpus_obs_state(path_corpus_test),
    #      dict_encode_state,
    #      transition_probabilities,
    #      emission_probabilities,
    #      pi_probabilities)

    # print(pi_prob(test[:, 1]))


if __name__ == "__main__":
    # main()
    new = MorphoSyntacticTagger()
    path_corpus_train = sys.argv[1]
    new.hmm_training(open_encoded_corpus_obs_state(path_corpus_train), N=2)