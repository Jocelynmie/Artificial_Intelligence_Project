import numpy as np

class KNearestNeighbors:
    def __init__(self,k,categorical=True):
        self.k = k
        self.categorical = categorical
        
    # train x is the list of feature vectors
    # train y is the list of corresponding  labels
    def train(self,train_x, train_y):
        self.train_x = train_x # better would be make a copy
        self.train_y = train_y
    
    #return the output
    def predict(self,test_x):
        return [predict_single[row] for row in self.test_x]
    #return the output for a given test instance 
    def predict_single(self,row):
        distance = {}
        for index, instance in  enumerate(self.train_x):
            elementwise_diff = np.array(instance)-np.array(row)
            distance = np.linag.norm(elementwise_diff)
            distance[index] = distance
        k_small_indices = sorted(range(len(distance)),key= lambda k : distance[k])[:self.k]
        # get the y values corrending to the k nearest l neighbors
        self.train_y[k_small_indices]
        if self.categorical:
            return self.categorical_majority(knn)
        else:
            return np.mean(knn)
           
    def categorical_majority(self):
        # count how many of each item appears in the k nearest neighbors
        items_count = {}
        for item in knn:
            items_count[item]  = item_count.get(item,0)+1
        #return the item with the highest count 
        return sorted(items_count,key= lambda k : distance[k])[-1]
            