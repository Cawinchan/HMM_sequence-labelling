import numpy as np

# Testing
START_STATE_KEY = "START"
STOP_STATE_KEY = "STOP"
epsilon = 10**-5
LOG_ZERO_PROB = float("-inf")

test_emission_proba_dict = {
    "X": {
        "a": np.log(0.4),
        "b": np.log(0.6),
        "c": LOG_ZERO_PROB
    },
    "Y": {
        "a": np.log(0.2),
        "b": LOG_ZERO_PROB,
        "c": np.log(0.8)
    },
    "Z": {
        "a": np.log(0.2),
        "b": np.log(0.6),
        "c": np.log(0.2)
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