import sys
import re
from copy import copy
from collections import defaultdict
from optparse import OptionParser

separator = ' '

#Read entities from predcition
def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict)), outputColumnIndex=1):

    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    answers[example] = []
    for line in predicted:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                answers[example].append(list(entity))
                entity = []

            example += 1
            answers[example] = []
            word_index = 0
            last_ne = "O"
            continue
        else:
            split_line = line.split(separator)
            #word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]


            last_entity = []

            #check if it is start of entity
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = list(entity)

                entity = [sent]
                    
                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity =list(entity)
                entity = []


            if last_entity:
                answers[example].append(list(last_entity))
                last_entity = []

        last_sent = sent
        last_ne = ne
        word_index += 1

    if entity:
        answers[example].append(list(entity))


    return answers



#Read entities from gold data
def get_observed(observed,outputColumnIndex=1):


    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_sent = ""
    last_entity = []

    observations=defaultdict(defaultdict)
    observations[example] = []

    for line in observed:
        line = line.strip()
        if line.startswith("##"):
            continue
        elif len(line) == 0:
            if entity:
                observations[example].append(list(entity))
                entity = []

            example += 1
            observations[example] = []
            word_index = 0
            last_ne = "O"
            continue

        else:
            split_line = line.split(separator)
            word = split_line[0]
            value = split_line[outputColumnIndex]
            ne = value[0]
            sent = value[2:]


            last_entity = []

            #check if it is start of entity, suppose there is no weird case in gold data
            if ne == 'B' or (ne == 'I' and last_ne == 'O') or (last_ne != 'O' and ne == 'I' and last_sent != sent):
                if entity:
                    last_entity = entity

                entity = [sent]
                    
                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = entity
                entity = []


            if last_entity:
                observations[example].append(list(last_entity))
                last_entity = []


        last_ne = ne
        last_sent = sent
        word_index += 1

    if entity:
        observations[example].append(list(entity))

    return observations

#Print Results and deal with division by 0
def compile_dict(num_gold,num_pred, num_correct):
    prec = num_correct / num_pred
    rec = num_correct / num_gold
    if abs(prec + rec ) < 1e-6:
        f = 0
    else:
        f = 2 * prec * rec / (prec + rec)
    return {
        "correct": num_correct,
        "total_actual": num_gold,
        "total_predicted": num_pred,
        "precision": prec,
        "recall": rec,
        "f": f
    }


#Compare results bewteen gold data and prediction data
def compare_observed_to_predicted(observed, predicted):

    correct_sentiment = 0
    correct_entity = 0

    total_observed = 0.0
    total_predicted = 0.0

    #For each Instance Index example (example = 0,1,2,3.....)
    for example in observed:
        observed_instance = observed[example]
        predicted_instance = predicted[example]

        #Count number of entities in gold data
        total_observed += len(observed_instance)
        #Count number of entities in prediction data
        total_predicted += len(predicted_instance)

        #For each entity in prediction
        for span in predicted_instance:
            span_begin = span[1]
            span_length = len(span) - 1
            span_ne = (span_begin, span_length)
            span_sent = span[0]

            #For each entity in gold data
            for observed_span in observed_instance:
                begin = observed_span[1]
                length = len(observed_span) - 1
                ne = (begin, length)
                sent = observed_span[0]

                #Entity matched
                if span_ne == ne:
                    correct_entity += 1
                    

                    #Entity & Sentiment both are matched
                    if span_sent == sent:
                        correct_sentiment += 1

    return {
        "Entity": compile_dict(total_observed,total_predicted,correct_entity),
        "Sentiment": compile_dict(total_observed,total_predicted,correct_sentiment)
    }

def evaluate_predictions(actual_dir,predicted_dir,outputColumnIndex=1):
    gold = open(actual_dir, "r", encoding='UTF-8')
    prediction = open(predicted_dir, "r", encoding='UTF-8')

    #Read Gold data
    observed = get_observed(gold,outputColumnIndex=outputColumnIndex)

    #Read Predction data
    predicted = get_predicted(prediction,outputColumnIndex=outputColumnIndex)

    gold.close()
    prediction.close()

    #Compare
    return compare_observed_to_predicted(observed, predicted)
