import math

# Testing
START_STATE_KEY = "START"
STOP_STATE_KEY = "STOP"
epsilon = 10**-5
LOG_ZERO_PROB = float("-inf")

test_emission_proba_dict = {
    "X": {
        "a": math.log(0.4),
        "b": math.log(0.6),
        "c": LOG_ZERO_PROB
    },
    "Y": {
        "a": math.log(0.2),
        "b": LOG_ZERO_PROB,
        "c": math.log(0.8)
    },
    "Z": {
        "a": math.log(0.2),
        "b": math.log(0.6),
        "c": math.log(0.2)
    }
}

test_transmission_proba_dict = {
    START_STATE_KEY: {
        "X": math.log(0.5),
        "Y": LOG_ZERO_PROB,
        "Z": math.log(0.5),
        STOP_STATE_KEY: LOG_ZERO_PROB
    },
    "X": {
        "X": LOG_ZERO_PROB,
        "Y": math.log(0.4),
        "Z": math.log(0.4),
        STOP_STATE_KEY: math.log(0.2)
    },
    "Y": {
        "X": math.log(0.2),
        "Y": LOG_ZERO_PROB,
        "Z": math.log(0.2),
        STOP_STATE_KEY: math.log(0.6)
    },
    "Z": {
        "X": math.log(0.4),
        "Y": math.log(0.6),
        "Z": LOG_ZERO_PROB,
        STOP_STATE_KEY: LOG_ZERO_PROB
    }
}