from .best_k_viterbi import *
from .dataset_utils import parse_dataset
import copy

def part_3(nonsparse_emission_dict,nonsparse_transmission_dict,stateset,test_dir):
    ''' Perform testing for part 3. Saves dataset to test_dir's immediate folder as dev.p3.out
    
    :param nonsparse_emission_dict: dictionary of log-emission probabilities from part 3
    :type: dict
    
    :param nonsparse_transmission_dict: dictionary of log-transmission probabilities from part 3
    :type: dict

    :param stateset: set of states
    :type: set

    :param test_dir: filename of training file
    :type: string
    
    '''
    output_dir = test_dir[:test_dir.index("dev.in")] + "dev.p3.out"
    k = 5

    test_dataset = parse_dataset(test_dir,False)
    output_lines = []

    for i in test_dataset:
        pred_seqs = viterbi_best_k(i,nonsparse_emission_dict,nonsparse_transmission_dict,stateset,k)
        pred_seqs = [(i[0],i[1][1:-1]) for i in pred_seqs]
        while len(pred_seqs)<k:
            pred_seqs.append(copy.deepcopy(pred_seqs[-1]))
        output_lines_interm = []
        for j in range(len(i)):
            output_lines_interm.append(" ".join([i[j]]+[pred_seqs[a][1][j] for a in range(k)]))
        output_lines.append("\n".join(output_lines_interm))
    output = "\n\n".join(output_lines) + "\n\n"
    
    with open(output_dir,"w", encoding="utf-8") as f:
        f.write(output)