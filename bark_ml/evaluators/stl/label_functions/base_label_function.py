class BaseQuantizedLabelFunction():
    def __init__(self, robustness: float = float('-inf')):
        self.robustness = robustness    

    def get_current_robustness(self):
        return self.robustness