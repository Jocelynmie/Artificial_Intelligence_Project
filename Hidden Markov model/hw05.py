import string
from collections import defaultdict

def read_dataset_from_file(file_path):
    #Open and read entire file content
    with open(file_path, 'r') as file:
        data_string = file.read()
    return parse_dataset(data_string)

def parse_dataset(data_string):
    #Initialize empty list to store pairs
    #(misspelling, correct) 
    dataset = []
    #Each line in the input string
    for line in data_string.split('\n'):
        if ':' in line:
            #Split line into correct word and misspellings
            correct, misspellings = line.split(':')
            correct = correct.strip()
            #Create pairs for each misspelling 
            for misspelling in misspellings.split():
                dataset.append((misspelling.strip(), correct))
    return dataset

def calculate_emission_probabilities(dataset):
    #Defaultdict to store char emission counts
    emissions = defaultdict(lambda: defaultdict(int))
    #Track total emissions
    total_emissions = defaultdict(int)
    #Count emissions from typos to correct 
    for typo, correct in dataset:
        for t, c in zip(typo, correct):
            emissions[c][t] += 1
            total_emissions[c] += 1
            
     #Calculate emission probabilities
    emission_probs = defaultdict(lambda: defaultdict(float))
    for correct, typos in emissions.items():
        for typo, count in typos.items():
            emission_probs[correct][typo] = count / total_emissions[correct]
        
        #Add bias for correct char emission
        emission_probs[correct][correct] = max(emission_probs[correct].values()) + 0.1

    return emission_probs

def calculate_transition_probabilities(dataset):
    #Store transition counts
    transitions = defaultdict(lambda: defaultdict(int))
    #Track total transitions
    total_transitions = defaultdict(int)
    #Count transitions in correct words
    for x, correct in dataset:
        # from START state
        transitions['START'][correct[0]] += 1
        total_transitions['START'] += 1
        # between chars
        for i in range(len(correct) - 1):
            transitions[correct[i]][correct[i+1]] += 1
            total_transitions[correct[i]] += 1
        # to END state
        transitions[correct[-1]]['END'] += 1
        total_transitions[correct[-1]] += 1

    #Calculate transition probabilities 
    transition_probs = defaultdict(lambda: defaultdict(float))
    for state, next_states in transitions.items():
        for next_state, count in next_states.items():
            transition_probs[state][next_state] = count / total_transitions[state]

    return transition_probs

def viterbi_decode(word, emission_probs, transition_probs):
    #Get all possible states
    states = list(string.ascii_lowercase)
    V = [{}]
    path = {}
    #first step
    for state in states:
        V[0][state] = transition_probs['START'][state] * emission_probs[state].get(word[0], 1e-10)
        path[state] = [state]
    #Run Viterbi algorithm 
    for t in range(1, len(word)):
        V.append({})
        newpath = {}
        #Calculate probabilities for each state
        for state in states:
            (prob, state0) = max((V[t-1][y0] * transition_probs[y0].get(state, 1e-10) * emission_probs[state].get(word[t], 1e-10), y0) for y0 in states)
            V[t][state] = prob
            newpath[state] = path[state0] + [state]
        #return final path
        path = newpath

    (prob, state) = max((V[len(word) - 1][y], y) for y in states)
    return ''.join(path[state])

def correct_text(text, emission_probs, transition_probs):
    words = text.split()
    corrected_words = [viterbi_decode(word.lower(), emission_probs, transition_probs) for word in words]
    return ' '.join(corrected_words)



def main():
    file_path = "/Users/yangwenyu/Desktop/CS5100 /HW5/dataset.txt"


    dataset = read_dataset_from_file(file_path)

    emission_probs = calculate_emission_probabilities(dataset)
    transition_probs = calculate_transition_probabilities(dataset)

    print("Please enter a word :")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        corrected_text = correct_text(user_input, emission_probs, transition_probs)
        print(f"correct: {corrected_text}")
        print()
        
        
if __name__ =='__main__':
    main()