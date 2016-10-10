#!/usr/bin/env python

import sys
import numpy as np
from math import log10
import copy


class NGramModel:

    def __init__(self, n=2, smoothing="Laplace"):
        self.__N = n
        self.__smoothing = smoothing

        self.__list_voc_states = []
        self.__list_voc_obs = []

        # these attributes are the parameters of the HMM model
        # each index i of the dicts consist of the i-gram frequencies/probabilities
        self.__state_transition_frequencies = {}
        self.__state_transition_probabilities = {}

        # Has the opposite, the matrix of emission frequencies/probabilities are directly stored in these attributes.
        self.__observation_emission_frequencies = {}
        self.__observation_emission_probabilities = {}

    def build_state_transition_frequency_matrix(self, depth, init_value=0):
        if depth < 1:
            # This should never happen
            exit("NGramModel.build_state_transition_frequency_matrix has been called with depth = 0. It should be > 0.")

        if depth == 1:
            # Exiting condition
            dict_state_transition_frequency_matrix = {}
            # each leef of the tree is initialized with the init_value
            for state in self.__list_voc_states:
                dict_state_transition_frequency_matrix[state] = init_value

        else:
            # calculate lower-hierarchy branches
            dict_state_transition_frequency_matrix = self.build_state_transition_frequency_matrix(depth=depth-1, init_value=init_value)
            terminus = dict_state_transition_frequency_matrix
            dict_state_transition_frequency_matrix = {}
            # plug lower-hierarchy branches to the current level
            for state in self.__list_voc_states:
                dict_state_transition_frequency_matrix[state] = copy.deepcopy(terminus)

        return dict_state_transition_frequency_matrix

    def build_observation_emission_frequency_matrix(self, init_value=0):
        # The observation emission matrix is always of dimension 2
        for observation in self.__list_voc_obs:
            self.__observation_emission_frequencies[observation] = {}
            for state in self.__list_voc_states:
                self.__observation_emission_frequencies[observation][state] = init_value

    def counter(self, sequence):
        sequence_index = 0
        length_sequence = len(sequence)

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
                sub_sequence_states.append(-1)

            while obs != 0:
                new_sequence = True
                self.__observation_emission_frequencies[obs][state] += 1
                sub_sequence_states.append(state)
                sequence_index += 1
                (obs, state) = sequence[sequence_index]
            # each sub-sequence ends with 1 "end of sub-sequence state"
            sub_sequence_states.append(-2)

            # if we were actually processing a real subsequence and not an artifact produced by two consecutive
            # sequence separator in the corpus
            if new_sequence:
                # count every state transition in the subsequence (1-gram transition are also considered here)
                self.count_state_transition_frequencies(sub_sequence_states)
                # Counts the number of sub-sequence separations
                self.__observation_emission_frequencies[0][0] += 1

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

    def compute_state_transition_probabilities_matrix(self, state_transition_frequencies, depth, smoothing=None):
        if depth == 1:
            tmp_dict = {}
            for state in self.__list_voc_states:
                if smoothing == 'Laplace':
                    tmp_dict[state] = state_transition_frequencies[state] / (self.__state_frequencies[state] + len(self.__list_voc_states))
                else:
                    tmp_dict[state] = state_transition_frequencies[state] / self.__state_frequencies[state]
            return tmp_dict

        elif depth > 1:
            tmp_dict = {}
            for state1 in self.__list_voc_states:
                tmp_dict[state1] = self.compute_state_transition_probabilities_matrix(state_transition_frequencies[state1], depth=depth-1)
            return tmp_dict


    def ngram_training(self, corpus):

        if len(self.__list_voc_obs) == 0:
            self.set_voc_obs(list(set(corpus[:, 0])))

        if len(self.__list_voc_states) == 0:
            self.set_voc_states(list(set(corpus[:, 1])) + [-1] + [-2])

        if self.__smoothing == "Laplace":
            # Laplace smoothing consists on adding a weight to each N-gram probability.
            # todo: at the moment, this weight is 1, it might be interesting to parametrize this
            init_value = 1
        else:
            init_value = 0

        # setup the frequencies matrix before processing the corpus in order to prevent any "key-missing" error
        for i in range(self.__N):
            self.__state_transition_frequencies[i+1] = self.build_state_transition_frequency_matrix(
                depth=i+1,
                init_value=init_value)
        self.build_observation_emission_frequency_matrix(init_value=init_value)

        self.counter(corpus)

        self.__state_transition_probabilities = \
            self.compute_state_transition_probabilities_matrix(self.__state_transition_frequencies, N)
        self.compute_emission_probabilities()

    def viterbi(self, sequence_obs):
        # def viterbi(, labels, transition_matrix, emission_matrix, pi_probs):

        backtrack = {}
        sequence_obs_size = len(sequence_obs)
        score = [{}]

        # initialization
        for y in self.__list_voc_states:
            score[0][y] = self.__[y] + emission_matrix[sequence_obs[0]][y]

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
                    score_y_last = score[i - 1][y_last]
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
            output[i - 1] = backtrack[i][y]
            y = backtrack[i][y]
            i -= 1
        return output

    def predict(self):
        pass


    def compute_emission_probabilities(self, smoothing=None):
        for observable in self.__list_voc_obs:
            self.__observation_emission_probabilities[observable] = {}
            for state in self.__list_voc_states:
                if smoothing == 'Laplace':
                    self.__observation_emission_probabilities[observable][state] = \
                        self.__observation_emission_frequencies[observable][state] / (
                        self.__state_frequencies[state] + len(self.__list_voc_obs))
                else:
                    self.__observation_emission_probabilities[observable][state] = \
                        self.__observation_emission_frequencies[observable][state] / self.__state_frequencies[state]

    def set_voc_states(self, list_voc_states):
        self.__list_voc_states = list_voc_states

    def set_voc_obs(self, list_voc_obs):
        self.__list_voc_obs = list_voc_obs

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
    list_tuples_obs_state = [tuple((0, 0))]
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
    if list_tuples_obs_state[-1] != tuple((0, 0)):
        list_tuples_obs_state.append(tuple((0, 0)))
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

def build_two_elm_combinatorial(token_list1, token_list2, init_value=0):
    combinatorial = {}
    for token1 in token_list1:
        combinatorial[token1] = {}
        for token2 in token_list2:
            combinatorial[token1][token2] = init_value
    return combinatorial


def viterbi(sequence_obs, labels, transition_matrix, emission_matrix, pi_probs):

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
    # path_corpus_test = sys.argv[4]
    # path_voc_observable = sys.argv[2]
    # path_voc_states = sys.argv[3]

    # dict_decode_obs, dict_encode_obs = open_vocabulary(path_voc_observable)
    # dict_decode_state, dict_encode_state = open_vocabulary(path_voc_states)

    # lissage de laplace + 1 au compte du bigramme / + V au compte de l'unigramme


    # state_occurence_init = {}
    # length_voc_state_plus_length_voc_lex = len(dict_encode_obs)
    # for token in dict_encode_state.keys():
    #     state_occurence_init[token] = length_voc_state_plus_length_voc_lex
    #
    # emission_occurence_matrix_init = build_two_elm_combinatorial(dict_encode_obs.keys(),
    #                                                              dict_encode_state.keys(),
    #                                                              init_value=1)

    # emission_probabilities = emission_prob(open_encoded_corpus_obs_state(path_corpus_train),
    #                                        state_occurence_init,
    #                                        emission_occurence_matrix_init)
    #
    # state_occurence_init = {}
    # length_voc_state = len(dict_encode_state)
    # for token in dict_encode_state.keys():
    #     state_occurence_init[token] = length_voc_state
    #
    # pi_probabilities = pi_prob(open_encoded_corpus_obs_state(path_corpus_train)[:, 1],
    #                            dict_decode_state["PONCT"],
    #                            state_occurence_init)
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


if __name__ == "__main__":
    # main()
    new = NGramModel(n=3)
    path_corpus_train = sys.argv[1]
    new.ngram_training(open_encoded_corpus_obs_state(path_corpus_train))