import numpy as np 


# Answer for Part 1a)

def MLE_emission_parameters(train_dir = "data/ES/train"):
    ''' Calculates the emission parameters by count(y->x)/count(y)
    
    :param train_dir: our train file path to either ES or RU
    :type train_dir: str

    :return: count_y_dict, Count(y), keys are word '!', value MLE
    :rtype: dict

    :return: count_y_to_x_dict, Count(y->x), keys are tuples of word and label ('!', 'O'), value MLE
    :rtype: dict    

    :return: emission_dict, Count(y->x)/Count(y), keys are tuples of word and label ('!', 'O'), value MLE
    :rtype: dict
    
    '''
    count_y_dict = {}
    count_y_to_x_dict = {}
    emission_dict = {}

    with open(train_dir, "r", encoding="utf8") as f:
        for line in f:
            # Parse each line
            if len(line.split(" ")) == 2:
                word, label = line.replace("\n","").split(" ")
            else:
                # skip lines with space 
                continue
            if label in count_y_dict:
                count_y_dict[label] = count_y_dict.get(label) + 1
            else:
                count_y_dict[label] = 1
            if (word,label) in count_y_to_x_dict:
                count_y_to_x_dict[(word,label)] = count_y_to_x_dict.get((word,label)) + 1
            else:
                count_y_to_x_dict[(word,label)] = 1
    # print("count(y): \n", count_y_dict, "\n")
    # print("count(y->x): \n",list(count_y_to_x_dict.items())[0:5], len(count_y_to_x_dict), "\n")
    # Calculate our emission
    for key, value in count_y_to_x_dict.items(): # Default is iterate keys()
        word = key[0]
        label = key[1]
        prob =  value / count_y_dict.get(label)
        emission_dict[key] = np.where(prob != 0, np.log(prob), float("-inf"))
    # print("MLE: \n",list(emission_dict.items())[0:5],len(emission_dict) ,"\n")

    return count_y_dict, count_y_to_x_dict, emission_dict

# Answer for Part 1b)

def new_MLE_emission_parameters_with_unknown(count_y_dict, count_y_to_x_dict, emission_dict, k=1):
    ''' Adds the unknown_word_token to our dictionary and finds our new emission paramters 

    :param count_y_dict: count y dictionary
    :type count_y_dict: dict

    :param count_y_to_x_dict: count y -> x dictionary
    :type count_y_to_x_dict: dict
    
    :param emission_dict: Emission dictionary
    :type emission_dict: dict

    :param k: we assume we have observed that there are k occurrences of such an event.
    :type k: int

    :return: emission_plus_unknown_dict, keys are tuple of word and label ('!', 'O'), value MLE
    :rtype: dict
    
    '''
    # Calculate our new emission
    for key, value in count_y_to_x_dict.items(): # Default is iterate keys()
        label = key[1]
        prob =  value / (count_y_dict.get(label) + k)
        emission_dict[key] = np.where(prob != 0, np.log(prob), float("-inf"))

    # print("#UNK# values:")
    for key in count_y_dict:
        prob = k / (count_y_dict.get(key) + k)
        emission_dict[("#UNK#",key)] = np.where(prob != 0, np.log(prob), float("-inf"))
        # print(("#UNK#",key),emission_dict.get(("#UNK#",key)))
    return emission_dict

# Answer for Part 1c)

def predict_y(emission_dict, test_dir="data/ES/dev.in", output_dir="data/ES/dev.p1.out"):
    ''' Finds our predicted_y with our emission_dict through argmax[y] p(x|y)
     
    :param test_dir: our test file in either ES or RU
    :type test_dir: str

    :param output_dir: our output file for either ES or RU
    :type test_dir: str

    :Output: "dev.p1.out" by default to the directory given
    :rtype: .out file
    '''
    
    emission_word_set = set(i[0] for i in list(emission_dict.keys()))

    emission_label_lst = list(set(i[1] for i in list(emission_dict.keys())))

    with open(output_dir,'w', encoding="utf-8") as f:
        with open(test_dir,'r',encoding="utf-8") as file:
            for line in file:
                if len(line.replace("\n","")) > 0:
                    word = line.replace("\n","")
                else:
                    f.write("\n")                    
                    continue
                if word not in emission_word_set: # If there is no such word in emission set word as unknown
                    word = "#UNK#"

                label_arr = np.zeros((len(emission_label_lst)))
                for idx, label in enumerate(emission_label_lst):
                    if emission_dict.get((word,label)):
                        label_arr[idx] = emission_dict.get((word,label))
                    else:
                        label_arr[idx] = float("-inf")
                predicted_y_idx = np.argmax(label_arr,axis=0)
                predicted_y = emission_label_lst[predicted_y_idx] # Convert argmax index to predicted name
                f.write(f"{word} {predicted_y}\n") # Write in our original word