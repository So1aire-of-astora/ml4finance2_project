import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import random

"""
----------User Manual for the Scorer----------
This is an improved, and of course, cooler version of the original scorer for the fake news challange. 
Developed by Paul Zhang. 

During the project, we had to get rid of some not-so-cool news samples that are NOT EVEN IN ENGLISH, 
and as a result, the original scorer just wouldn't work. Well, here comes the new scorer :)

We have also planned for some customizations on the scoring criterion, so that our models always get the highest 
possible scores. But since this is apparently immoral, we have kept the original criterion:
    +0.25 for each correct unrelated
    +0.25 for each correct related (label is any of agree, disagree, discuss)
    +0.75 for each correct agree, disagree, discuss

The scorer will provide three scores: MAX, NULL, and TEST
    MAX  - the best possible score (100% accuracy)
    NULL - score as if all predicted stances were unrelated
    TEST - score based on the provided predictions

Another thing we choose to keep is the usage:

    $ python scorer.py test_labels pred_labels output_path
    
    params:
        test_labels - CSV file with TRUE stance labels
        pred_labels - CSV file with PREDICTED stance labels
        output_path - Optional, specifies the path of the output file if you prefer not to print to the console.

----------End of User Manual----------
"""

class CoolScorer:

    labels = ['agree', 'disagree', 'discuss', 'unrelated']

    def __init__(self, test_path, pred_path):
        y_true, y_pred = pd.read_csv(test_path)["Stance"], pd.read_csv(pred_path)["Stance"]
    
        self.score, self.cm = self.score_model(y_true, y_pred)
        self.accuracy = np.diag(self.cm).sum() / y_true.shape[0]
        self.null_score, self.best_score = self.score_default(y_true)
        self.relative_score = self.score / self.best_score * 100

    @classmethod
    def score_model(cls, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels = cls.labels)
        base_score = np.zeros((4, 4))
        np.fill_diagonal(base_score, np.array([.75, .75, .75, .25]))
        score_matrix = base_score + np.pad(.25*np.ones((3, 3), dtype=int), pad_width=((0, 1), (0, 1)), mode='constant', constant_values=0)
        return np.sum(cm * score_matrix), pd.DataFrame(cm, index=[f"Actual {label}" for label in cls.labels], columns=[f"Predicted {label}" for label in cls.labels])
    
    @staticmethod
    def score_default(y_true):
        """
        Compute the "all false" baseline (all labels as unrelated) and the max possible score
        params:
        - y_true: pandas DataFrame containing the true labels
        return: 
            (null_score, best_score)
        """
        null_score = (y_true[y_true == "unrelated"].shape[0]) * .25
        return null_score, null_score + y_true.shape[0] - 3 * null_score
    
    def __str__(self):
        return "Confusion Matrix:\n{}\nMax Score: {:.2f}\tNull Score: {:.2f}\tModel Score: {:.2f}\tRelative Score: {:.2f}\nModel Accuracy:{:.6f}".format(self.cm, 
            self.best_score, self.null_score, self.score, self.relative_score, self.accuracy)

# class WrongUsageException(Exception):
#     def __init__(self):
#         self.messages = [
#             "WHY HAVEN'T YOU READ THE DOCS? WHY?",
#             "TOLD YOU YOU ARE GONNA GET YELLED AT!",
#             "THAT'S NOT HOW YOU USE THE SCORER! MY SCORER!",
#             "NICE TRY!",
#             "REEEEEAD THE DOOOOOOOCS!",
#             "I DON'T THINK SO!"
#         ]
#         self.message = random.choice(self.messages)
#         super().__init__(self.message)

if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        # raise WrongUsageException
        print("Wrong usage. Please check the instructions.")
        sys.exit(1)

    scorer = CoolScorer(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 3:
        print(scorer)
    else:
        with open(sys.argv[-1], 'w') as f:
            print(scorer, file = f)