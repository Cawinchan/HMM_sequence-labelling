import numpy as np

# Answer for Part 2b)

def viterbi(emission_dict, transition_dict, test_dir, output_dir = "data/ES/dev.p2.out"):
    ''' Perform Viterbi

    :param emission_dict: emission_dict dictionary
    :type emission_dict: dict

    :param transition_dict: transition dictionary
    :type transition_dict: dict
    
    :param test_dir: our test file in either ES or RU
    :type test_dir: str

    :param output_dir: our output file for either ES or RU
    :type test_dir: str

    :Output: "dev.p2.out" by default to the directory given
    :rtype: .out file
    
    '''


    test_array = []
    viterbi_array = [{"word": "", 'O': float("-inf"), "START": (1, ''), 'STOP': float("-inf"), 'B-positive': float("-inf"), 'B-negative': float("-inf"), 'B-neutral': float("-inf"), 'I-positive': float("-inf"), 'I-negative': float("-inf"), 'I-neutral': float("-inf")}]
    labels = ['O', 'START', 'STOP', 'B-positive', 'B-negative', 'B-neutral', 'I-positive', 'I-negative', 'I-neutral']

    emission_word_set = set(i[0] for i in list(emission_dict.keys()))

    with open(test_dir, 'r',encoding="utf-8") as file:
        for line in file:
            test_array += [line.replace("\n","")]

    count = 1
    #Parse each line, calculate the probabilities for each label and create a dictionary for each line with the label probabilities.
    for word in test_array:
        temp_dict = {'word': word}
        if word == '':
            temp_list = []
            for prev_y in labels:
                if viterbi_array[count - 1].get(prev_y) != float("-inf"):
                    if transition_dict.get((prev_y, "STOP")):
                        temp_list.append(np.longdouble(viterbi_array[count - 1].get(prev_y)[0] + transition_dict.get((prev_y, "STOP"))))
                    else:
                        temp_list.append(float("-inf"))
                else:
                    temp_list.append(float("-inf"))

            max_index = np.argmax(temp_list)
            for y in labels:
                if y == "STOP":
                    temp_dict[y] = (temp_list[max_index], labels[max_index])
                elif y == "START":
                    temp_dict[y] = (1, '')
                else:
                    temp_dict[y] = float("-inf")
            viterbi_array.append(temp_dict)
        else:
            if word not in emission_word_set:
                word = "#UNK#"
            for t in labels:
                if emission_dict.get((word, t)):
                    temp_list = []
                    for prev_y in labels:
                        if viterbi_array[count - 1].get(prev_y) != float("-inf"):
                            if transition_dict.get((prev_y, t)):
                                temp_list.append(viterbi_array[count - 1].get(prev_y)[0] + transition_dict.get((prev_y, t)) + emission_dict.get((word, t)))
                            else:
                                temp_list.append(float("-inf"))
                        else:
                            temp_list.append(float("-inf"))
                    max_index = np.argmax(temp_list)
                    temp_dict[t] = (temp_list[max_index], labels[max_index])
                else:
                    temp_dict[t] = float("-inf")
                
            viterbi_array.append(temp_dict)
        count += 1
    #Backtrack the viterbi array to find the highest probability paths
    result_array = [""]*len(viterbi_array)
    for i in range(len(viterbi_array) - 1, 0, -1):
        if i == len(viterbi_array) - 1:
            result_array[i] = viterbi_array[i].get("word")
        tmp_list = []
        if viterbi_array[i].get('word') == '':
            result_array[i] = ''
            prev_label = viterbi_array[i].get('STOP')[1]
        else:
            result_array[i] = viterbi_array[i].get("word") + " " + prev_label
            try:
                prev_label = viterbi_array[i].get(prev_label)[1]
            except:
                prev_label = 'O'
    #Write the results into a text file
    with open(output_dir,'w', encoding="utf-8") as f:
        for i in result_array[1:]:
            f.write(i + '\n')


