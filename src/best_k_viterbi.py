import numpy as np
import copy

START_STATE_KEY = "START"
STOP_STATE_KEY = "STOP"
UNK_KEY = "#UNK#"
LOG_ZERO_PROB = float("-inf")

def viterbi_best_k(word_seq,log_emission_params,log_transmission_params,stateset,k):
    ''' Get list of at most top k tag sequences for a given word sequence.
    Note that it is not guaranteed for k sequences to be returned.
    Less will be returned if there are simply not enough possible sequences.
    
    :param word_seq: List of words in sentence
    :type ls: list

    :param log_emission_params: Dictionary of log-emission probabilities from Part 3
    :type log_emission_params: dict

    :param log_transmission_params: Dictionary of log-emission probabilities from Part 3
    :type log_transmission_params: dict

    :param stateset: set of states
    :type stateset: set

    :param wordset: set of words
    :type wordset: set

    :param ls: k
    :type ls: int
    
    :return: List of top k (or less) (log-probability, sequence as list of tags) 
    :rtype: list
    
    '''

    probabilities = {START_STATE_KEY: [(0,[START_STATE_KEY])]}
    
    for i in range(len(word_seq)):
        word = word_seq[i]
        probabilities = viterbi_best_k_step(probabilities,log_emission_params,log_transmission_params,word,stateset,k)
        
    probabilities = viterbi_best_k_step(probabilities,log_emission_params,log_transmission_params,STOP_STATE_KEY,stateset,k)
    
    return probabilities[STOP_STATE_KEY]

def filter_best_k(ls, k):
    ''' Filter and return best k (log-probability, sequence pairs)
    
    :param ls: List of (log-probability, sequence pairs)
    :type ls: list

    :param ls: k
    :type ls: int

    :return: List of top k (log-probability, sequence pairs)
    :rtype: list
    
    '''

    ls_sorted = sorted(ls,key=lambda i: -1*i[0])
    return ls_sorted[:k]

def viterbi_best_k_step(probabilities,log_emission_params,log_transmission_params,word,stateset,k):
    ''' Iteration step for best-k viterbi algorithm
    
    :param probabilities: dictionary indexed by previous possible tags, with values being list of top-k or less (log-probability, sequence list)
    :type ls: dict

    :param log_emission_params: Dictionary of log-emission probabilities from Part 3
    :type log_emission_params: dict

    :param log_transmission_params: Dictionary of log-emission probabilities from Part 3
    :type log_transmission_params: dict

    :param word: Current word in step (or STOP_STATE_KEY if ending sequence on this step)
    :type word: str

    :param stateset: set of states
    :type stateset: set

    :param ls: k
    :type ls: int
    
    :return: dictionary indexed by current possible tags, with values being list of top-k or less (log-probability, sequence list)
    :rtype: dict
    
    '''

    target_stateset = None
    output_proba_dict = dict()
    target_stateset = None
    if word==STOP_STATE_KEY:
        target_stateset = set([STOP_STATE_KEY])
    else:
        target_stateset = copy.deepcopy(stateset)
        target_stateset.remove(START_STATE_KEY)
        target_stateset.remove(STOP_STATE_KEY)
    
    for target_state in target_stateset:
        log_emission_prob = 0 if STOP_STATE_KEY in target_stateset else log_emission_params[target_state].get(word,log_emission_params[target_state][UNK_KEY])
        sum_new_list = []
        for source_state in probabilities:
            current_list = probabilities[source_state]
            log_transmission_prob = log_transmission_params[source_state][target_state]
            for i in current_list: sum_new_list.append((i[0]+log_transmission_prob+log_emission_prob,i[1]+[target_state]))
        output_proba_dict[target_state] = filter_best_k(sum_new_list,k)
    
    return output_proba_dict

def get_stateset_and_wordset(part_1_emission_dict,part_2_transmission_dict):
    ''' Parses emission and transmission dictionaries from parts 1 and 2, and returns sets of all states and of all words respectively.
    
    :param part_1_emission_dict: Part 1 emission dictionary
    :type part_1_emission_dict: dict 

    :param part_2_transmission_dict: Part 2 transmission dictionary
    :type part_2_transmission_dict: dict

    :return: stateset, wordset (sets of strings)
    :rtype: set, set
    
    '''
    stateset = set()
    wordset = set()
    for i in part_1_emission_dict:
        stateset.add(i[1])
        wordset.add(i[0])
    for i in part_2_transmission_dict:
        stateset.add(i[0])
        stateset.add(i[1])
    return stateset, wordset



def convert_to_log_nonsparse_emission_dict(part_1_emission_dict,stateset,wordset):
    ''' Parses emission dictionary from part 1, and returns newly-formatted dictionary that is non-sparse.
    
    :param part_1_emission_dict: Part 1 emission dictionary
    :type part_1_emission_dict: dict 

    :return: output_dict: dictionary containing emission log-probabilities and indexed by [source state][word]
    :rtype: dict
    
    '''

    output_dict = dict()
    stateset = copy.deepcopy(stateset)
    stateset.remove(START_STATE_KEY)
    stateset.remove(STOP_STATE_KEY)

    for i in stateset:
        add_dict = dict()
        for j in wordset:
            if (j,i) in part_1_emission_dict:
                add_dict[j] = part_1_emission_dict[(j,i)]
            else:
                add_dict[j] = LOG_ZERO_PROB
        output_dict[i] = add_dict
    return output_dict

def convert_to_log_nonsparse_tranmission_dict(part_2_transmission_dict,stateset):
    ''' Parses emission dictionary from part 2, and returns newly-formatted dictionary that is non-sparse.
    
    :param part_2_transmission_dict: Part 2 transmission dictionary
    :type part_2_transmission_dict: dict 

    :param stateset: set of states
    :type stateset: set

    :return: output_dict: dictionary containing tranmission log-probabilities and indexed by [source state][destination state]
    :rtype: dict
    
    '''

    output_dict = dict()
    origin_stateset = copy.deepcopy(stateset)
    origin_stateset.remove(STOP_STATE_KEY)
    destination_stateset = copy.deepcopy(stateset)
    destination_stateset.remove(START_STATE_KEY)

    for i in origin_stateset:
        add_dict = dict()
        for j in destination_stateset:
            if (i,j) in part_2_transmission_dict:
                add_dict[j] = part_2_transmission_dict[(i,j)]
            else:
                add_dict[j] = LOG_ZERO_PROB
        output_dict[i] = add_dict
    return output_dict