import numpy as np
from best_k_viterbi import viterbi_best_k, LOG_ZERO_PROB, START_STATE_KEY, STOP_STATE_KEY, UNK_KEY

# Testing
if __name__=="__main__":
    epsilon = 10**-5

    test_emission_proba_dict = {
        "X": {
            "a": np.log(0.4),
            "b": np.log(0.6),
            "c": LOG_ZERO_PROB,
            UNK_KEY: LOG_ZERO_PROB
        },
        "Y": {
            "a": np.log(0.2),
            "b": LOG_ZERO_PROB,
            "c": np.log(0.8),
            UNK_KEY: LOG_ZERO_PROB
        },
        "Z": {
            "a": np.log(0.2),
            "b": np.log(0.6),
            "c": np.log(0.2),
            UNK_KEY: LOG_ZERO_PROB
        }
    }

    test_transmission_proba_dict = {
        START_STATE_KEY: {
            "X": np.log(0.5),
            "Y": LOG_ZERO_PROB,
            "Z": np.log(0.5),
            STOP_STATE_KEY: LOG_ZERO_PROB
        },
        "X": {
            "X": LOG_ZERO_PROB,
            "Y": np.log(0.4),
            "Z": np.log(0.4),
            STOP_STATE_KEY: np.log(0.2)
        },
        "Y": {
            "X": np.log(0.2),
            "Y": LOG_ZERO_PROB,
            "Z": np.log(0.2),
            STOP_STATE_KEY: np.log(0.6)
        },
        "Z": {
            "X": np.log(0.4),
            "Y": np.log(0.6),
            "Z": LOG_ZERO_PROB,
            STOP_STATE_KEY: LOG_ZERO_PROB
        }
    }

    stateset = set(["X","Y","Z",START_STATE_KEY,STOP_STATE_KEY])

    test_1 = viterbi_best_k(["b","c"],test_emission_proba_dict,test_transmission_proba_dict,stateset,1)
    test_1_expected_sequences = [START_STATE_KEY,"Z","Y",STOP_STATE_KEY]
    assert abs((np.e**test_1[0][0])-0.0864)<=epsilon
    assert len(test_1_expected_sequences)==len(test_1[0][1])
    for i in range(len(test_1_expected_sequences)):
        assert test_1[0][1][i]==test_1_expected_sequences[i]
        
    test_2 = viterbi_best_k(["b","c"],test_emission_proba_dict,test_transmission_proba_dict,stateset,3)
    test_2_expected_scores = [0.0864,0.0576,0]
    test_2_expected_sequences = [[START_STATE_KEY,"Z","Y",STOP_STATE_KEY],[START_STATE_KEY,"X","Y",STOP_STATE_KEY]]
    for i in range(len(test_2_expected_scores)):
        assert abs((np.e**test_2[i][0])-test_2_expected_scores[i])<=epsilon
    for i in range(len(test_2_expected_sequences)):
        for j in range(len(test_2_expected_sequences[i])):
            assert test_2[i][1][j]==test_2_expected_sequences[i][j]

    print("All tests passed for best k Viterbi algorithm")