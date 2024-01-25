from bark.core.world.evaluation.ltl import EvaluatorLTL
from bark_ml.evaluators.stl.label_functions.base_label_function import BaseQuantizedLabelFunction

class EvaluatorSTL(EvaluatorLTL):    
    def __init__(self, agent_id: int, ltl_formula: str, label_functions, eval_return_robustness_only: bool = True):
        super().__init__(agent_id, ltl_formula, label_functions)
        self.robustness = float('inf')
        self.label_functions_stl = label_functions
        self.eval_return_robustness_only = eval_return_robustness_only        

    def Evaluate(self, observed_world):        
        eval_return = super().Evaluate(observed_world)        

        # print(f"Evaluate STL return: {eval_return}")

        if self.eval_return_robustness_only:
            eval_return = self.compute_robustness()
        else:
            eval_return = str(eval_return) + ";" + str(self.compute_robustness())
        # print(f"Evaluate return updated: {eval_return}")

        return eval_return
    
    def compute_robustness(self): 
        self.robustness = float('inf')

        for le in self.label_functions: 
            # print(le)                       
            if isinstance(le, BaseQuantizedLabelFunction):                 
                self.robustness = min(self.robustness, le.get_current_robustness())                                

        # print("------------------")
        if self.robustness == float('inf') or self.robustness == float('-inf'):
           self.robustness = 0.0
           
        # print(f'Robustness in EvaluatorSTL: {self.robustness}')
        return self.robustness