import numpy as np
from datetime import datetime
from symlearn.core.methods import Methods
from sklearn.metrics import r2_score

class Model:
    def __init__(self,
                 max_evaluations=10000,
                 max_generations=-1,
                 max_time=None,
                 verbose=True
                 ) -> None:
        self.model = None
        self.max_evaluations = max_evaluations
        self.current_evaluation = 0
        self.max_generations = max_generations
        self.current_generation = 0
        self.max_time = max_time
        self.start_time = None
        self.end_time = None
        self.verbose = verbose

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def predict(self, X):
        return self.model.output(X)

    def must_terminate(self):
        terminate = False
        if self.max_time and datetime.now() > self.end_time:
            terminate = True
        elif self.max_evaluations > -1 and self.current_evaluation > self.max_evaluations:
            terminate = True
        elif self.max_generations > -1 and self.current_generation > self.max_generations:
            terminate = True
        elif self.model.fitness < self.target_error:
            terminate = True
        return terminate

    def export_best(self, export_path='images/', filename='Best'):
        if self.model:
            label = "Best error: "
            label += str(round(self.model.fitness, 3))
            Methods.export_graph(self.model, export_path + filename, label)
            print(self.model.equation())

    def test_model(self, X):
        zero_array = np.zeros(X.shape)
        x = np.vstack([zero_array, X])
        return self.model.output(x)[-1]
