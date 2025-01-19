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
    
    def evaluate(self, value):
        """
        Da overriddare nelle subclass. Deve restituire un numero:
        - 0.0 se 'valido'
        - float('inf') se 'invalido'
        """
        raise NotImplementedError
    
    def _get_str(self):
        raise NotImplementedError
    
    def __or__(self, other) -> 'CompositePrior':
        """
        Composizione in OR: se uno dei due prior è 'valido',
        allora il risultato è 0.0, altrimenti inf.
        """
        return CompositePrior(self, other, op="or")

    def __and__(self, other) -> 'CompositePrior':
        """
        Composizione in AND: i due prior devono essere entrambi validi
        (0.0) per avere output 0.0, altrimenti inf.
        """
        return CompositePrior(self, other, op="and")
    
    
class CompositePrior(Prior):
    def __init__(self, left: Prior, right: Prior, op: str):
        super().__init__(name=f"CompositePrior({op})")
        self.left = left
        self.right = right
        self.op = op.lower()  # normalizziamo, ad es. 'OR' -> 'or'

    def evaluate(self, value):
        # Evaluo i due prior figli
        left_val = self.left(value)
        right_val = self.right(value)

        if self.op == "or":
            #TODO logica è sbagliata ma per ora basta avere o e inf come valori corretti
            #
            # Logica "OR":
            # Se almeno uno dei due è "valido" (== 0.0),
            # ritorniamo 0.0, altrimenti inf.
            #
            # In pratica, 0.0 e inf => min(0.0, inf) = 0.0 => valido
            # inf e inf => min(inf, inf) = inf => invalido
            #
            return min(left_val, right_val)

        elif self.op == "and":
            #
            # Logica "AND":
            # Se entrambi sono validi (entrambi 0.0), allora 0.0
            # altrimenti inf.
            #
            # Possiamo usare max() (0.0 e inf => inf),
            # oppure sommarli (0.0 + inf => inf).
            #
            # Entrambe le scelte vanno bene. Esempio con max():
            #
            return max(left_val, right_val)

        else:
            raise ValueError(f"Operatore sconosciuto: {self.op}")


class UniformPrior(Prior):
    def __init__(self, lower, upper, valid = 0.0, invalid = -float('inf')):
        super().__init__('Uniform Prior')
        self.lower = lower
        self.upper = upper
        self.valid = valid
        self.invalid = invalid
    
    
    def evaluate(self, value):
        if value < self.lower or value > self.upper:
            return self.invalid
        return self.valid
    
    def _get_str(self) -> str:
        'Helper func to get string representative of the prior'
        def format_value(value):
            return f"{value:.1e}" if value > 1000 or value < 0.001 else f"{value:.1f}"
        
        lower_str = format_value(self.lower)
        upper_str = format_value(self.upper)
        return f"Uniform({lower_str}, {upper_str})"

        