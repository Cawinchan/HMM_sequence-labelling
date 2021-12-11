import numpy as np

from src.emission import MLE_emission_parameters, new_MLE_emission_parameters_with_unknown, predict_y
from src.transition import MLE_transition_parameters
from src.viterbi import viterbi
from src.best_k_viterbi import get_stateset_and_wordset, convert_to_log_nonsparse_emission_dict, convert_to_log_nonsparse_tranmission_dict
from src.part_3 import part_3

folder_dir = "ES"
train_dir = f"data/{folder_dir}/train"
test_dir = f"data/{folder_dir}/dev.in"

if __name__ == "__main__":
    # Part 1

    # Find emission params
    count_y_dict, count_y_to_x_dict, emission_dict = MLE_emission_parameters(train_dir)
    emission_dict = new_MLE_emission_parameters_with_unknown(count_y_dict, count_y_to_x_dict, emission_dict, k=1)

    p1_output_dir = f"data/{folder_dir}/dev.p1.out"

    # Predict Sequence labels 
    predict_y(emission_dict,test_dir,p1_output_dir) # Output in output_dir

    # Part 2

    # Find transition paramters
    count_y_dict, count_y_to_y_dict, transition_dict = MLE_transition_parameters(train_dir)

    p2_output_dir = f"data/{folder_dir}/dev.p2.out"

    # Perform Viterbi
    viterbi(emission_dict, transition_dict, test_dir, p2_output_dir)

    # Part 3 
    stateset, wordset = get_stateset_and_wordset(emission_dict,transition_dict)
    nonsparse_emission_dict = convert_to_log_nonsparse_emission_dict(emission_dict,stateset,wordset)
    nonsparse_transmission_dict = convert_to_log_nonsparse_tranmission_dict(transition_dict,stateset)
    part_3(nonsparse_emission_dict,nonsparse_transmission_dict,stateset,test_dir)
    



