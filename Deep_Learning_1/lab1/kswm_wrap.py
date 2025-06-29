
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics

from sklearn.metrics import average_precision_score

import data

'''
Metode:
  __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
    Konstruira omotač i uči RBF SVM klasifikator
    X, Y_:           podatci i točni indeksi razreda
    param_svm_c:     relativni značaj podatkovne cijene
    param_svm_gamma: širina RBF jezgre

  predict(self, X)
    Predviđa i vraća indekse razreda podataka X

  get_scores(self, X):
    Vraća klasifikacijske mjere
    (engl. classification scores) podataka X;
    ovo će vam trebati za računanje prosječne preciznosti.

  support
    Indeksi podataka koji su odabrani za potporne vektore
'''

class KSWMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto', param_kernel="rbf"):
        self.svm = SVC(C=param_svm_c, gamma=param_svm_gamma, kernel=param_kernel)
        
        self.svm.fit(X, Y_)  

    def predict(self, X):
        return self.svm.predict(X)
    
    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_
    
    def eval(self, X, Y_):
        Y = self.predict(X)

        scores = self.get_scores(X)
        sorted_indices = np.argsort(-abs(scores))
        ranked_labels = Y_[sorted_indices]

        return (data.eval_perf_multi(Y, Y_)) #, data.eval_AP(ranked_labels))
    

if __name__=="__main__":
    np.random.seed(100)

    K=6
    C=2
    N=10

    X, Y_ = data.sample_gmm_2d(K, C, N)

    model = KSWMWrap(X, Y_, 1, "auto")

    special = model.support()
    
    Y = model.predict(X)

    print(model.eval(X, Y_))

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: model.predict(x), rect, offset=0)

    data.graph_data(X, Y_, Y, special) 

    plt.show()