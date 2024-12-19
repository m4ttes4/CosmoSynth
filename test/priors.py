'''
Concetto idea generale di come funziona un prior.
è un istanza di model (magari posso ereditare da una nuova classe "CallableModel" che permette di 
definire Model(grid, theta))
Il prior ritorna semplicemente Prior(point) = value
'''

class Prior:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, value):
        return self.evaluate(value)
    

class UniformPrior(Prior):
    def __init__(self, lower, upper, valid = 0.0, invalid = float('inf')):
        super().__init__('Uniform Prior')
        self.lower = lower
        self.upper = upper
        self.valid = valid
        self.invalid = invalid
    
    def evaluate(self, value):
        if value < self.lower or value > self.upper:
            return self.invalid
        return self.valid
        