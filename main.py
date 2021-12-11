import numpy as np

from src.emission import MLE_emission_parameters, new_MLE_emission_parameters_with_unknown, predict_y
from src.transition import MLE_transition_parameters
from src.viterbi import viterbi
from src.best_k_viterbi import output_function
from src.test_paramters import test_emission_proba_dict,test_transmission_proba_dict
from src.test_paramters import epsilon, START_STATE_KEY, STOP_STATE_KEY

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

    # Tests to see validity of best K viterbi algorithm  
    test_1 = output_function(["b","c"],test_emission_proba_dict,test_transmission_proba_dict,1)
    test_1_expected_sequences = [START_STATE_KEY,"Z","Y",STOP_STATE_KEY]
    assert abs((np.e**test_1[0][0])-0.0864)<=epsilon
    assert len(test_1_expected_sequences)==len(test_1[0][1])
    for i in range(len(test_1_expected_sequences)):
        assert test_1[0][1][i]==test_1_expected_sequences[i]
        
    test_2 = output_function(["b","c"],test_emission_proba_dict,test_transmission_proba_dict,3)
    test_2_expected_scores = [0.0864,0.0576,0]
    test_2_expected_sequences = [[START_STATE_KEY,"Z","Y",STOP_STATE_KEY],[START_STATE_KEY,"X","Y",STOP_STATE_KEY]]
    for i in range(len(test_2_expected_scores)):
        assert abs((np.e**test_2[i][0])-test_2_expected_scores[i])<=epsilon
    for i in range(len(test_2_expected_sequences)):
        for j in range(len(test_2_expected_sequences[i])):
            assert test_2[i][1][j]==test_2_expected_sequences[i][j]



