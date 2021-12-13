import copy
from .best_k_viterbi import START_STATE_KEY, STOP_STATE_KEY, UNK_KEY, LOG_ZERO_PROB
from .dataset_utils import parse_dataset
import numpy as np

def get_log_dest_state_freq_dict(stateset,training_sentences):
    ''' Get log frequency of destination states (states to which transmission occurs)
    
    :param stateset: set of states
    :type stateset: set
    
    :param training_sentences: list of list of [word,state]
    :type training_sentences: list[list[str]]

    :Output: dictionary of key: destination state, value: log_proba
    :rtype: dict
    '''
    count_dict = {i:0 for i in stateset if not i==START_STATE_KEY}
    total_count = 0
    
    count_dict[STOP_STATE_KEY] = len(training_sentences)
    total_count += len(training_sentences)

    for i in training_sentences:
        for j in i:
            count_dict[j[1]] += 1
            total_count += 1
    
    return {i:np.where(count_dict[i]!=0,np.log(count_dict[i])-np.log(total_count),LOG_ZERO_PROB) for i in count_dict}

def get_skipgram_log_transmission_params_dict_list(stateset,training_sentences,log_dest_state_freq_dict,k,regularization_count=0):
    ''' Get (regularized) log-transmission parameters dict list.
    Each list index corresponds to a dict of probabilities of stateseq -> destination state similar to in part 3 (first axis: origin stateseq, second axis: destination state, value: log-prob).

    :param stateset: set of all states
    :type stateset: set

    :param training_sentences: dataset of training sentences
    :type training_sentences: list[list[str]]

    :param log_dest_state_freq_dict: dictionary of log-destination state frequencies
    :type log_dest_state_freq_dict: dict
    
    :param k: maximum origin stateseq length
    :type k: int

    :param regularization_count: number of word counts to add for each origin stateseq
    :type regularization_count: int

    :Output: list of dictionary of dictionary of log-transmission params
    :rtype: dict
    '''
    count_dict_list = []
    total_count_dict_list = []
    origin_stateset = copy.deepcopy(stateset)
    origin_stateset.remove(STOP_STATE_KEY)
    destination_stateset = copy.deepcopy(stateset)
    destination_stateset.remove(START_STATE_KEY)
    origin_stateseq_list = None
    for i in range(k): 
        if i == 0:
            origin_stateseq_list = [(j,) for j in origin_stateset]
        else:
            new_origin_stateseq_list = []
            for j in origin_stateseq_list:
                new_origin_stateseq_list += [tuple(list(j)+[k]) for k in destination_stateset if not k==STOP_STATE_KEY]
            origin_stateseq_list = new_origin_stateseq_list
        count_dict_entry = {j:{k:regularization_count*(np.e**log_dest_state_freq_dict[k]) for k in destination_stateset} for j in origin_stateseq_list}
        count_dict_list.append(count_dict_entry)
        total_count_dict_entry = {j:regularization_count for j in origin_stateseq_list}
        total_count_dict_list.append(total_count_dict_entry)
    
    for sentence in training_sentences:
        stateseq = [START_STATE_KEY] + [i[1] for i in sentence] + [STOP_STATE_KEY]
        for i in range(1,len(stateseq)):
            dependency_len = min(i,k)
            dependency = tuple(stateseq[i-dependency_len:i])
            curr_state = stateseq[i]
            count_dict_list[dependency_len-1][dependency][curr_state] += 1
            total_count_dict_list[dependency_len-1][dependency] += 1
    
    out_dict_list = []
    for i in range(k):
        out_dict_entry = dict()
        for j in count_dict_list[i]:
            if total_count_dict_list[i][j]==0:
                out_dict_entry[j] = {l:log_dest_state_freq_dict[l] for l in destination_stateset}
            else:
                out_dict_entry[j] = {l:np.log(count_dict_list[i][j][l])-np.log(total_count_dict_list[i][j]) if not count_dict_list[i][j][l]==0 else LOG_ZERO_PROB for l in destination_stateset}
        out_dict_list.append(out_dict_entry)
    return out_dict_list

def part_4_naffins_viterbi(word_seq,log_emission_params,log_transmission_params_list,stateset,k):
    ''' Viterbi algorithm based on k-length skipgram

    :param word_seq: list of str listing words in sentence
    :type word_seq: list[str]

    :param log_emission_params: dict of dict of log-emission prob
    :type log_emission_params: dict

    :param log_transmission_params_list: list of dict of dict of log-transmission prob
    :type log_transmission_params_list: list[dict]

    :param stateset: set of states
    :type stateset: set

    :param k: maximum length of origin stateseq determining next state
    :type k: int
    
    :Output: most likely tuple of (log-proba, state sequence)
    :rtype: tuple
    '''

    probabilities = {(START_STATE_KEY,):(0,[START_STATE_KEY])}

    for i in range(len(word_seq)):
        word = word_seq[i]
        probabilities = part_4_naffins_viterbi_step(probabilities,log_emission_params,log_transmission_params_list,word,stateset,min(i,k-1),k)
    
    probabilities = part_4_naffins_viterbi_step(probabilities,log_emission_params,log_transmission_params_list,STOP_STATE_KEY,stateset,min(len(word_seq),k-1),k)
    probabilities = [probabilities[i] for i in probabilities]
    
    max_pair = None
    max_score = None
    for i in probabilities:
        if max_score==None:
            max_score = i[0]
            max_pair = i
            continue
        if i[0]>max_score:
            max_score = i[0]
            max_pair = i
    return max_pair

def part_4_naffins_viterbi_step(probabilities,log_emission_params,log_transmission_params_list,word,stateset,transmission_params_list_index,k):
    ''' Perform a single step of k-length stepgram-oriented Viterbi algorithm
    
    :param probabilities: dict of (ending portion of stateseq (max k length) as tuple)->(score,stateseq)
    :type probabilities: dict
    
    :param log_emission_params: dict of dict of log-emission probabilities
    :type log_emission_params: dict

    :param log_transmission_params_list: list of dict of dict of log-transmission probabilities
    :type log_transmission_params_list: list[dict]

    :param word: current word
    :type word: str

    :param stateset: set of all states
    :type stateset: set

    :param transmission_params_list_index: current length of preceding stateseq determining current state
    :type transmission_params_list_index: int

    :param k: maximum length of preceding stateseq determining current state
    :type k:int

    :Output: same format as probabilities dictionary, advanced by 1 step
    :rtype: dict
    '''
    log_transmission_params = log_transmission_params_list[transmission_params_list_index]
    target_stateset = None
    if word==STOP_STATE_KEY:
        target_stateset = set([STOP_STATE_KEY])
    else:
        target_stateset = copy.deepcopy(stateset)
        target_stateset.remove(START_STATE_KEY)
        target_stateset.remove(STOP_STATE_KEY)
    
    output_proba_dict = dict()
    for target_state in target_stateset:
        log_emission_prob = 0 if STOP_STATE_KEY in target_stateset else log_emission_params[target_state].get(word,log_emission_params[target_state][UNK_KEY])
        for source_statelist in probabilities:
            new_statelist = tuple((list(source_statelist) + [target_state])[-1*k:])
            original_pair = probabilities[source_statelist]
            new_log_proba = original_pair[0] + log_transmission_params[source_statelist][target_state] + log_emission_prob
            new_stateseq = original_pair[1] + [target_state]
            if new_statelist in output_proba_dict:
                if new_log_proba > output_proba_dict[new_statelist][0]:
                    output_proba_dict[new_statelist] = (new_log_proba,new_stateseq)
            else:
                output_proba_dict[new_statelist] = (new_log_proba,new_stateseq)
    return output_proba_dict