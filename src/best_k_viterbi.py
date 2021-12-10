import numpy as np

START_STATE_KEY = "START"
STOP_STATE_KEY = "STOP"
UNK_KEY = "#UNK#"
LOG_ZERO_PROB = float("-inf")

# word_seq: list of [word_1,word_2,word_3,...]
# log_emission_params: NON-SPARSE dict of {
#   "source_state": {
#     "target_word": logarithmic probability
#   }
# }
# log_transmission_params follows same format
# k: number of best candidates to output
def output_function(word_seq,log_emission_params,log_transmission_params,k):
    probabilities = {START_STATE_KEY: [(0,[START_STATE_KEY])]}
    
    for i in range(len(word_seq)):
        word = word_seq[i]
        probabilities = viterbi_best_k(probabilities,log_emission_params,log_transmission_params,word,k)
        
    probabilities = viterbi_best_k(probabilities,log_emission_params,log_transmission_params,STOP_STATE_KEY,k)
    
    return probabilities[STOP_STATE_KEY]

def filter_best_k(ls, k):
    ls_sorted = sorted(ls,key=lambda i: -1*i[0])
    return ls_sorted[:k]

def viterbi_best_k(probabilities,log_emission_params,log_transmission_params,word,k):
    target_stateset = None
    output_proba_dict = dict()
    target_stateset = [STOP_STATE_KEY] if word==STOP_STATE_KEY else [i for i in list(log_transmission_params.keys()) if not i==START_STATE_KEY]
    
    for target_state in target_stateset:
        log_emission_prob = 0 if STOP_STATE_KEY in target_stateset else log_emission_params[target_state].get(word,None)
        sum_new_list = []
        for source_state in probabilities:
            current_list = probabilities[source_state]
            log_transmission_prob = log_transmission_params[source_state][target_state]
            for i in current_list: sum_new_list.append((i[0]+log_transmission_prob+log_emission_prob,i[1]+[target_state]))
        output_proba_dict[target_state] = filter_best_k(sum_new_list,k)
    
    return output_proba_dict

# Convert to format admissible for output_function
def convert_to_log_emission_params(emission_dict):
    output_dict = dict()
    source_state_set = set()
    target_word_set = set()
    for i in emission_dict:
        source_state_set.add(i[1])
        target_word_set.add(i[0])
    for i in list(source_state_set):
        add_dict = dict()
        for j in list(target_word_set):
            if (j,i) in emission_dict:
                add_dict[j] = np.log(emission_dict[(j,i)]) if emission_dict[(j,i)]>0 else LOG_ZERO_PROB
            else:
                add_dict[j] = 0
        output_dict[i] = add_dict
    return output_dict