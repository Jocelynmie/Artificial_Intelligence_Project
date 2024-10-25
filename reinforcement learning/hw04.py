import random 
import numpy as np 
from collections import defaultdict
import os
import string
from scipy.special import softmax

class CharNGramLanguageModel:
    def __init__(self, n, dataset):
        self.n = n
        self.ngram_counts = defaultdict(lambda: defaultdict(float))
        self.all_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ')
        self.train(dataset)

    def get_dataset_from_txt(self, directory):
        dataset = []
        if os.path.isfile(directory):
            with open(directory, 'r', encoding='utf-8') as file:
                dataset = file.read().strip().split('$')
        else:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                        dataset.extend(file.read().strip().split('$'))
        return [text.strip() for text in dataset if text.strip()]
    
   

    def train(self, dataset):
        for text in dataset:
            sequence = '\0' * self.n + text + '\0'
            self.all_chars.update(set(sequence))

            for i in range(len(sequence) - self.n):
                ngram = sequence[i:i+self.n]
                next_char = sequence[i+self.n]
                self.ngram_counts[ngram][next_char] += 1

    def generate_character(self, prompt):
        ngram = prompt[-self.n:]
        if ngram in self.ngram_counts:
            chars, weights = zip(*self.ngram_counts[ngram].items())
            # Using softmax to handle negative weights
            probs = softmax(weights)
            next_char = np.random.choice(chars, p=probs)
            return next_char
        else:
            return random.choice(list(self.all_chars))

    def generate(self, prompt, max_length=1000):
        generated_text = prompt
        for _ in range(max_length - len(prompt)):
            next_char = self.generate_character(generated_text[-self.n:])
            if next_char == '\0':
                break
            generated_text += next_char
        return generated_text

class ReinforcementLearning:
    def __init__(self, char_ngram_model, alpha, gamma):
        self.model = char_ngram_model
        self.alpha = alpha
        self.gamma = gamma
        
    #Q_learn function: criteria argument is used correctly
    def Q_learn(self, criteria, num_prompts=1, iterations_per_prompt=30):
        # Q_learn function asks ad correctly uses `num_prompts` number of prompt
        for _ in range(num_prompts):
            prompt = input(f"Enter a prompt (at least {self.model.n} characters): ")
            if len(prompt) < self.model.n:
                print(f"Prompt must be at least {self.model.n} characters long.")
                continue
            
            # Q_learn function correctly generates `iterations_per_prompt` number of generations per prompt
            for _ in range(iterations_per_prompt):
                generated_text = self.model.generate(prompt)
                reward = criteria(generated_text)

                # Update weights using Q-learning
                for i in range(len(generated_text) - self.model.n):
                    ngram = generated_text[i:i+self.model.n]
                    next_char = generated_text[i+self.model.n]
                    
                    current_q = self.model.ngram_counts[ngram][next_char]
                    max_future_q = max(self.model.ngram_counts[generated_text[i+1:i+1+self.model.n]].values(), default=0)
                    
                    new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
                    self.model.ngram_counts[ngram][next_char] = new_q

                print(f"Generated text: {generated_text}")
                print(f"Reward: {reward}")
def manual_criteria(text):
    print("\nGenerated text:")
    print(text)
    print("\nEvaluation criteria:")
    print("9-10: Perfect match to preferences")
    print("7-8: Good match with minor issues")
    print("5-6: Partially matches preferences")
    print("3-4: Mostly doesn't match, but has some good elements")
    print("1-2: Completely misses the mark")
    
    while True:
        try:
            score = float(input("\nRate this text from 1 to 10: "))
            if 1 <= score <= 10:
                return score
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
def main():
    data_path = '/Users/yangwenyu/Desktop/CS5100 /HW1/Data'
    
    # Create a new instance of CharNGramLanguageModel
    model = CharNGramLanguageModel(4, [])
    dataset = model.get_dataset_from_txt(data_path)

    if not dataset:
        print("Invalid file path or empty dataset.")
        return

    model.train(dataset)

    # Get user input for the prompt
    prompt = input(f"Enter a prompt (at least {model.n} characters): ")
    while len(prompt) < model.n:
        print(f"Prompt must be at least {model.n} characters long.")
        prompt = input(f"Enter a prompt (at least {model.n} characters): ")

    # Generate 10 pieces of text and calculate average length before Q-learning
    total_length = 0
    for i in range(10):
        generated_text = model.generate(prompt)
        total_length += len(generated_text)
        print(f"Sample generated text before Q-learning {i+1}: {generated_text}")
        print(f"Length: {len(generated_text)}")
    
    avg_length_before = total_length / 10
    print(f"Average length before Q-learning: {avg_length_before}")

    # Create ReinforcementLearning instance and run Q-learning with lambda x: -len(x)
    rl = ReinforcementLearning(model, alpha=0.1, gamma=0.9)
    print("\nRunning Q-learning with lambda x: -len(x)")
    rl.Q_learn(criteria=lambda x: -len(x), num_prompts=1, iterations_per_prompt=30)

    # Generate 10 more pieces of text and calculate average length after Q-learning with lambda x: -len(x)
    total_length = 0
    for i in range(10):
        generated_text = model.generate(prompt)
        total_length += len(generated_text)
        print(f"Sample generated text after Q-learning (lambda) {i+1}: {generated_text}")
        print(f"Length: {len(generated_text)}")
    
    avg_length_after_lambda = total_length / 10
    print(f"Average length after Q-learning (lambda): {avg_length_after_lambda}")






    # Reset the model
    model = CharNGramLanguageModel(4, [])
    model.train(dataset)

    # Generate 10 pieces of text and calculate average score before Q-learning
    total_score = 0
    for i in range(10):
        generated_text = model.generate(prompt)
        score = manual_criteria(generated_text)
        total_score += score
        print(f"Sample generated text before Q-learning (manual) {i+1}: {generated_text}")
        print(f"Score: {score}")
    
    avg_score_before = total_score / 10
    print(f"Average score before Q-learning (manual): {avg_score_before}")

    # Create ReinforcementLearning instance and run Q-learning with manual_criteria
    rl = ReinforcementLearning(model, alpha=0.1, gamma=0.9)
    print("\nRunning Q-learning with manual_criteria")
    rl.Q_learn(criteria=manual_criteria, num_prompts=1, iterations_per_prompt=30)

    # Generate 10 more pieces of text and calculate average score after Q-learning with manual_criteria
    total_score = 0
    for i in range(10):
        generated_text = model.generate(prompt)
        score = manual_criteria(generated_text)
        total_score += score
        print(f"Sample generated text after Q-learning (manual) {i+1}: {generated_text}")
        print(f"Score: {score}")
    
    avg_score_after = total_score / 10
    print(f"Average score after Q-learning (manual): {avg_score_after}")

if __name__ == "__main__":
    main()


