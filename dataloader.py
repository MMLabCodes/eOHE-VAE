"""
This file is meant to go from various representations to 1HOT and back
"""
import numpy as np
# from GrammarVAE_codes import to_one_hot, prods_to_eq
# import GrammarVAE_grammar as zinc_grammar
from itertools import combinations
import selfies as sf
from tqdm import tqdm

def unique_chars_iterator(smile):
     """
     """
     atoms = []
     for i in range(len(smile)):
         atoms.append(smile[i])
     return atoms

def grammar_one_hot_to_smile(one_hot_ls):
    _grammar = zinc_grammar
    _productions = _grammar.GCFG.productions()
    
    # This is the generated grammar sequence
    grammar_seq = [[_productions[one_hot_ls[index,t].argmax()] 
                        for t in range(one_hot_ls.shape[1])] 
                        for index in range(one_hot_ls.shape[0])]
    #print(grammar_seq)
    smile = [prods_to_eq(prods) for prods in grammar_seq]
    
    return grammar_seq, smile

def integer_encode_smiles(smiles, dictionary,largest_smile_len):
    test_smiles = ''
    encoding = []

    for i in range(len(smiles)):
        # print('dictionary',dictionary)
        test_smiles=test_smiles + smiles[i]
        try:
            value_encoded = dictionary[test_smiles]
            # print('Vals',test_smiles,value_encoded)
            encoding.append(value_encoded)
            test_smiles = ''
        except:
            encoding.append('Not_complete_item')
    encoding_list = [i for i in encoding if i != 'Not_complete_item']
    # Add padding element if the lenght of array was reduced to smaller value than largest_smile_len
    if len(encoding_list) < largest_smile_len:
        # print('Len encoding_list', len(encoding_list),encoding_list)
        encoding_list += [dictionary[' ']]*(largest_smile_len-len(encoding_list))
        # print('Len encoding_list', len(encoding_list),encoding_list)
    return encoding_list

def smile_to_hot(smile, largest_smile_len, alphabet, type_of_encoding,dictionary=None):
    if dictionary:
        char_to_int = dictionary
    else:
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    if type_of_encoding==0:
        for _ in range(largest_smile_len-len(smile)):
            smile+=' ' 
    elif type_of_encoding==1: 
        for _ in range(largest_smile_len-len(smile)):
            smile+=' '    
    elif type_of_encoding==2: 
        for _ in range(largest_smile_len-len(list(sf.split_selfies(smile)))):
            smile+=' '        

    integer_encoded = integer_encode_smiles(smile, char_to_int,largest_smile_len)
    # print('integer_encoded',integer_encoded)
        
    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

def multiple_smile_to_hot(smiles_list, largest_smile_len, alphabet, type_of_encoding,dictionary=None):
    """
    Convert a list of smile strings to a one-hot encoding
    
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in tqdm(smiles_list):
        if dictionary:
            _, onehot_encoded = smile_to_hot(smile, largest_smile_len, alphabet, type_of_encoding,dictionary)
        else:
            _, onehot_encoded = smile_to_hot(smile, largest_smile_len, alphabet, type_of_encoding)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def smile_to_hot_v1(smile, largest_smile_len, alphabet, type_of_encoding,dictionary=None):
    """
    Go from a single smile string to a one-hot encoding.
    """
    if dictionary:
        char_to_int =dictionary
    else:
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    if type_of_encoding==0:
        for _ in range(largest_smile_len-len(smile)):
            smile+=' ' 
    elif type_of_encoding==1: 
        for _ in range(largest_smile_len-len(smile)):
            smile+=' '    
    elif type_of_encoding==2: 
        # print('Lenght selfie',len(list(sf.split_selfies(smile))))
        for _ in range(largest_smile_len-len(list(sf.split_selfies(smile)))):
            smile+=' '       
    integer_encoded = integer_encode_smiles(smile, char_to_int,largest_smile_len)
    # print('integer_encoded',integer_encoded)
    pairs = get_nearest_pairs(len(alphabet))[0]
    # print('Pairs',pairs)

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(pairs[0])]
        new_position_index = value//pairs[1]
        equivalent_value = (value + 1 - pairs[1] * new_position_index)/pairs[1]
        letter[new_position_index] = equivalent_value
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)    

def multiple_smile_to_hot_v1(smiles_list, largest_smile_len, alphabet, type_of_encoding,dictionary=None):
    """
    Convert a list of smile strings to a one-hot encoding
    
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in tqdm(smiles_list):
        if dictionary:
            _, onehot_encoded = smile_to_hot_v1(smile, largest_smile_len, alphabet, type_of_encoding,dictionary)
        else:
            _, onehot_encoded = smile_to_hot_v1(smile, largest_smile_len, alphabet, type_of_encoding)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)    

def smile_to_hot_v2(smile, largest_smile_len, alphabet, type_of_encoding,dictionary):
    """
    Go from a single smile string to a one-hot encoding.
    """
    if dictionary:
        char_to_int =dictionary
    else:
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    if type_of_encoding==0:
        for _ in range(largest_smile_len-len(smile)):
            smile+=' ' 
    elif type_of_encoding==1: 
        for _ in range(largest_smile_len-len(smile)):
            smile+=' '    
    elif type_of_encoding==2: 
        for _ in range(largest_smile_len-len(list(sf.split_selfies(smile)))):
            smile+=' '        
        
    integer_encoded = integer_encode_smiles(smile, char_to_int,largest_smile_len)
    # integer_encoded = [char_to_int[char] for char in unique_chars_iterator(smile)]
    pairs = get_nearest_pairs(len(alphabet))[0]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        # letter = [0 for _ in range(len(alphabet))]
        letter = [0 for _ in range(pairs[0])]
        new_position_index = value//pairs[1]
        equivalent_value = 2**(value - pairs[1]*new_position_index)/(2**(pairs[1]-1 ))
        letter[new_position_index] = equivalent_value
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)    

def multiple_smile_to_hot_v2(smiles_list, largest_smile_len, alphabet, type_of_encoding,dictionary):
    """
    Convert a list of smile strings to a one-hot encoding
    
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in tqdm(smiles_list):
        # _, onehot_encoded = smile_to_hot_v2(smile, largest_smile_len, alphabet, type_of_encoding)
        if dictionary:
            _, onehot_encoded = smile_to_hot_v2(smile, largest_smile_len, alphabet, type_of_encoding,dictionary)
        else:
            _, onehot_encoded = smile_to_hot_v2(smile, largest_smile_len, alphabet, type_of_encoding)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)    
        

def hot_to_smile(onehot_encoded,alphabet):
    """
    Go from one-hot encoding to smile string
    """    
    # From one-hot to integer encoding
    integer_encoded = onehot_encoded.argmax(1)
    
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    # integer encoding to smile
    regen_smile = "".join(int_to_char[x] for x in integer_encoded)
    regen_smile = regen_smile.strip()
    return regen_smile


def check_conversion_bijection(smiles_list, largest_smile_len):
    """
    This function should be called to check successful conversion to and from 
    one-hot on a data set.
    """
    for i, smile in enumerate(smiles_list):
        _, onehot_encoded = smile_to_hot(smile, largest_smile_len)
        regen_smile = hot_to_smile(onehot_encoded)
#        print('Original: ', smile, ' shape: ', len(smile))
#        print('REcon: ', regen_smile , ' shape: ', len(regen_smile))
#        return
        if smile != regen_smile:
            print('Filed conversion for: ', smile, ' @index: ', i)
            break
    print('All conditions passed!')


"""MOD: timehms added"""


def timehms(time_seconds):
    """
    Convert time in second to hours minutes and seconds

    Args:
        timehms (float): var that contains time in seconds

    Returns:
        hours, minutes and seconds (h,m,s) printed on terminal
    """
    h = time_seconds // 3600
    m = (time_seconds % 3600) // 60
    s = time_seconds % 60
    return int(h), int(m), s

def list2dictionary(lst):
    vals = [i for i in range(len(lst))]
    return dict(zip(lst, vals))


def get_pair_factors_of_number(length_of_vocabulary):
    """
    Given the length_of_vocabulary, this function return 
    the pairs of factors that multiply them together be 
    equal to length_of_vocabulary
    The factors are in the list 'given_list'

    Args:
        length_of_vocabulary (int): Length of vocabulary
    Returns:
        List with pairs of factors of length_of_vocabulary
    """

    pairs = []
    given_list = [i for i in range(2, 12)] + [13, 17, 19]
    given_list = 2*given_list
    for number1, number2 in combinations(given_list, 2):
        if number1 * number2 == length_of_vocabulary:
            pairs.append([number1, number2])
    return pairs

def get_nearest_pairs(length_of_vocabulary):
    """
    Given the length_of_vocabulary, this function return nearest 
    pairs of factors that multiply them together be equal to 
    length_of_vocabulary. If it is not possible to find pairs of factors
    a new class is added to vocabulary, to increase the length_of_vocabulary
    this process is repeated in a while loop until it will be possible to 
    find at least one pair of factors

    Args:
        length_of_vocabulary (int): Length of vocabulary
    Returns:
        List with pairs of factors of length_of_vocabulary (list)
        Number of classes to add to the vocabulary (int)
    """

    number = length_of_vocabulary
    pads_to_add = 0
    pairs = get_pair_factors_of_number(number)
    """
    if the length of vocabulary is not factorizable, increase
    artificially the length_of vocabulary until factorization 
    be possible, 
    Alternatives to avoid increase lenght_of_vocabulary: increase 
    top_number in given_list of function 'get pair_factors_of_number'
    """
    
    if len(pairs) == 0:
        while len(pairs) == 0:
            number = number + 1
            pads_to_add = pads_to_add + 1
            pairs = get_pair_factors_of_number(number)
    return pairs[0],pads_to_add