def parse_dataset(file,is_training=True):
    ''' Parse training/testing file to give list of sentences as lists of [word,state] (for training) or word (for testing)

    :param file: filename of training/testing file
    :type file: str

    :is_training: boolean to indicate whether file is a training dataset
    :type is_training: boolean
    
    :rtype: list[list[list[str]]] | list[list[str]]
    '''
    with open(file, "r", encoding="utf8") as f:
        data = f.read()
    sentences = data.split('\n\n')
    sentences = [i.strip().split('\n') for i in sentences if not i=='']
    if is_training:
        sentences = [[j.strip().split(' ') for j in i] for i in sentences]
        sentences = [[["".join(j[:-1]),j[-1]] for j in i] for i in sentences]
    return sentences