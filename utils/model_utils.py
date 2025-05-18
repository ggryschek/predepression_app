# model_utils.py

class ThresholdModelWrapper:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_model(self):
        return self.model

    def __getattr__(self, attr):
        return getattr(self.model, attr)
