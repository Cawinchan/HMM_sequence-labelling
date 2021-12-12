import numpy as np

# Answer for Part 2a)

def  MLE_transition_parameters(train_dir = "data/ES/train"):
    ''' Calculates the transition parameters by count(y->x)/count(y)
    
    :param train_dir: our train file path to either ES or RU
    :type train_dir: str

    :return: count_y_dict, Count(yi-1), keys are word '!', value MLE
    :rtype: dict

    :return: count_y_to_y_dict, Count(yi-1,yi), keys are tuples of word and label ('!', 'O'), value MLE
    :rtype: dict    

    :return: transition_dict, Count(yi-1, yi)/Count(yi-1), keys are tuples of word and label ('!', 'O'), value MLE
    :rtype: dict

    '''
    count_y_dict = {}
    count_y_to_y_dict = {}
    transition_dict = {}
    prev_label = ""

    with open(train_dir, "r", encoding="utf8") as f:
        for line in f:
            # Parse each line
            if len(line.split(" ")) == 2:
                word, label = line.replace("\n","").split(" ")
            else:
                label = ''
            if label == '' and prev_label != '':
                count_y_dict["STOP"] = count_y_dict.get("STOP") + 1 if count_y_dict.get("STOP") else 1
            elif label !='':
                if prev_label == '':
                    count_y_dict["START"] = count_y_dict.get("START") + 1 if count_y_dict.get("START") else 1
                if label in count_y_dict:
                    count_y_dict[label] = count_y_dict.get(label)+1
                else:
                    count_y_dict[label] = 1
            if prev_label == '' and label != '':
                if ("START", label) in count_y_to_y_dict:
                    count_y_to_y_dict[("START", label)] = count_y_to_y_dict.get(("START", label)) + 1
                else:
                    count_y_to_y_dict[("START", label)] = 1
            elif label == '' and prev_label != '':
                if (prev_label, "STOP") in count_y_to_y_dict:
                    count_y_to_y_dict[(prev_label, "STOP")] = count_y_to_y_dict.get((prev_label, "STOP")) + 1
                else:
                    count_y_to_y_dict[(prev_label, "STOP")] = 1
            elif label != '' and prev_label != '':
                if (prev_label, label) in count_y_to_y_dict:
                    count_y_to_y_dict[(prev_label, label)] = count_y_to_y_dict.get((prev_label, label)) + 1
                else:
                    count_y_to_y_dict[(prev_label, label)] = 1
            prev_label = label
    # print("count(y): \n", count_y_dict, "\n")
    # print("count(y->x): \n",list(count_y_to_y_dict.items()), len(count_y_to_y_dict), "\n")
    # Calculate our transition
    for key, value in count_y_to_y_dict.items(): # Default is iterate keys()
        prev_label = key[0]
        label = key[1]
        prob =  value / count_y_dict.get(prev_label)
        transition_dict[key] = np.where(prob != 0, np.log(prob), float("-inf"))
    # print("MLE: \n",list(transition_dict.items()), len(transition_dict) ,"\n")

    return count_y_dict, count_y_to_y_dict, transition_dict