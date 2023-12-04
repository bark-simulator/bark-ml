from bark.core.world.evaluation.ltl import EvaluatorLTL
from bark_ml.evaluators.stl.label_functions.base_label_function import BaseQuantizedLabelFunction

class EvaluatorSTL(EvaluatorLTL):    
    def __init__(self, agent_id: int, ltl_formula_str: str, label_functions):
        super().__init__(agent_id, ltl_formula_str, label_functions)
        self.robustness = float('inf')

    def Evaluate(self, observed_world):        
        eval_return = super().Evaluate(observed_world)
        # print(f"Evaluate return: {eval_return}")
        # print(f"Evaluate safety_violations: {super().safety_violations}")
        # TODO: Should we remove the # of safety violations? We should subtract the robustness, shouldn't we?
        eval_return = eval_return - self.compute_robustness()        
        # print(f"Evaluate return updated: {eval_return}")
        return eval_return
    
    def compute_robustness(self): 
        self.robustness = float('inf')
               
        for le in self.label_functions:
            if isinstance(le, BaseQuantizedLabelFunction):                
                self.robustness = min(self.robustness, le.get_current_robustness())                                

        if self.robustness == float('inf') or self.robustness == float('-inf'):
           self.robustness = 0.0
           
        # print(f'Robustness in EvaluatorSTL: {self.robustness}')
        return self.robustness