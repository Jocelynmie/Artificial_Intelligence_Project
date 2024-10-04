import random 
import numpy as np 
from collections import defaultdict
import os
import string

class CharNGramLanguageModel:
    def __init__(self, n,dataset):
        # the length of n-gram 
        self.n = n
        #use dict to save the count of n-gram 
        #self.ngram_counts -> {{}}
        # {
        #     "hel":{"l":1}, .....
        # }
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        # the probabilities of next char
        self.ngram_probabilities = {}
        # to store all unique characters in the dataset
        # self.all_chars = set(string.ascii_letters + string.digits + string.punctuation)  
        self.all_chars = set(string.ascii_letters)   
        self.train(dataset)
      

        
    '''
    Get the dataset from specific files
    return the dataset
    '''
    def get_dataset_from_txt(self, directory):
        dataset = []

        if os.path.isfile(directory):  # check if it's a file
            with open(directory, 'r', encoding='utf-8') as file:
                dataset.append(file.read().strip() + '\0')
        else:  # loop all files in this dircectory
            for filename in os.listdir(directory):
                # if the file is txt:
                if filename.endswith('.txt'):
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                        dataset.append(file.read().strip() + '\0')
        return dataset
    
    def train(self, dataset):
        # for each line:
        for sequence in dataset:
            #remove space and add‘/0’ to end the sequence
            # sequence = sequence.strip() + '\0'
            self.all_chars.update(set(sequence))

    
            for i in range(len(sequence) - self.n):
                # get n-gram and the next char
                ngram = sequence[i:i+self.n]
                next_char = sequence[i+self.n]
                #update
                self.ngram_counts[ngram][next_char] += 1
        # calculate the probability for each char
        for ngram, char_counts in self.ngram_counts.items():
            total_count = sum(char_counts.values())
            self.ngram_probabilities[ngram] = {
                char: count / total_count for char, count in char_counts.items()
            }
        # self.calculate_char_frequencies(dataset)
        
    # def calculate_char_frequencies(self, dataset):
    #     total_chars = 0 
    #     for sequence in dataset:
    #         sequence = sequence.strip() + '\0'
    #         for char in sequence:
    #             self.char_counts[char] += 1
    #             total_chars += 1
    #     self.char_probabilities = {
    #         char: count / total_chars for char, count in self.char_counts.items()
    #     }
    

    ''' Generate next char by prompt '''
    def generate_character(self, prompt):
        # get the last n chars
        ngram = prompt[-self.n:]
        if ngram in self.ngram_probabilities:
            # choose next char by probability
            chars, probs = zip(*self.ngram_probabilities[ngram].items())
            next_char = np.random.choice(chars, p=probs)
            return next_char
        else:
            # if there is no n-gram , choose the next character randomly based 
            # on the single previous character
            next_char = random.choice(list(self.all_chars))
            return next_char

    # generate text 
    def generate(self, prompt):
        #initialize
        generated_text = prompt
        for _ in range(100):  # limit to prevent infinite loop
            next_char = self.generate_character(generated_text[-self.n:])
            #  if next char is \0
            # which means we need to end generation 
            if next_char == '\0':
                break
            # update text
            generated_text += next_char
        return generated_text
 

def main():
    data_path = '/Users/yangwenyu/Desktop/CS5100 /HW1/Data'
    
    model_instance = CharNGramLanguageModel(3, [])
    dataset = model_instance.get_dataset_from_txt(data_path)

    if not dataset:
        print("Invalid file path.")
        return

    while True:
        try:
            n = int(input("Please enter the value of n for the n-gram model: "))
            if n <= 0:
                print("n must be a positive integer.")
                continue
            
            model = CharNGramLanguageModel(n, dataset)
            break
        except ValueError:
            print("Please enter a valid integer.")

    while True:
        prompt = input(f"Enter a prompt (at least {n} characters) or 'End' to quit: ")
        if prompt.lower() == 'end':
            break
        if len(prompt) < n:
            print(f"Prompt must be at least {n} characters long.")
            continue
        generated_text = model.generate(prompt)
        print("Generated text:", generated_text)

if __name__ == "__main__":
    main()