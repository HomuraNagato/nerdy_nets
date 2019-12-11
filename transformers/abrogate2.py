
import string
import json
import numpy as np
import pandas as pd


if __name__ == '__main__':

    file_path = '../data/cleaned_data/cleaned_data.json'
    batch_size = 100
    reader = pd.read_json(file_path, precise_float=True, dtype=False, lines=False)

    i = 0

    json_out = {}
    
    for section in reader:

        print("train\n", section[i])
        break;
        '''
        train = section[i]['train']
        test = section[i]['test']

        
        for train, test in zip(train_batch, test_batch):

            #print("train\n", train, "test\n", test)
        
            
            i += 1
            json_out[i] = { 'train': train, 'test': test }
        '''

    '''
    with open('cleaned_data.json', 'w') as json_file:
        json.dump(json_out, json_file)
    '''
