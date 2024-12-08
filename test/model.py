import inspect
from copy import deepcopy
import warnings
from parameter import Parameter, ParameterHandler
from typing import List,  Tuple
from io import StringIO
from collections  import OrderedDict
from itertools import islice
import operator


def calcola_dimensioni(lista):
    # Variabili di controllo per la sequenza
    dimensioni = 0
    if lista[0] == "x" and lista[1] != "y":
        dimensioni += 1
    elif lista[0] == "x" and lista[1] == "y" and lista[2] != "z":
        dimensioni += 2
    elif lista[0] == "x" and lista[1] == "y" and lista[2] == "z":
        dimensioni += 3

    return dimensioni

def componemodels(op, **kwargs):
    return lambda left, right: CompositeModel(left, right, op, **kwargs)

class Model:
    # That means we can't optimize a method with **kwargs directly.
    # Also I think in the cover method we need to pass on the f(**z) method, right?
    # def g(f, x, z2):
    #   return f(z1=x[0], z2=z2, z3=x[1])
    _name = "SimpleModel"
    _parameters = ParameterHandler()
    _grid_variables = []    # args from evaluate that defines the grid
    _ndim = 1       #number of dimensions in wich the model is defined
    _ninputs = 1    # total number of inputs that defines the evaluate call
    _noutputs = 1   # number of outputs, i.e number of elements returned from evaluate call

    def __init__(self, func, parameters, ndim, ninputs, noutputs, name="SimpleModel") -> None:
        
        # NOTE rendere Model chiamabile direttamente come wrapper a funzione in stile LMFIT
        self._parameters = parameters
        self._ndim = ndim
        self._ninputs = ninputs
        self._noutputs = noutputs
        self._callable = func
        self._name = name
        self._grid_variables = []  # Inizializza _grid_variables nel costruttore
        self._cache = {}  # cache base

    def _update_cache(self, key, value) -> None:
        self._parameters._update_cache(key, value)

    # POINTER PROPRIETA
    @property
    def parameters(self):
        return self._parameters

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("New name must be of type string")
        self._name = value

    @property
    def n_dim(self):
        return self._ndim

    @property
    def n_inputs(self):
        return self._ninputs

    @property
    def n_outputs(self):
        return self._noutputs

    @property
    def grid_variables(self):
        return self._grid_variables

    @property
    def parameters_names(self) -> List[str]:  # cached
        return self.parameters.parameters_names

    @property
    def parameters_keys(self) -> List[str]:
        return self.parameters.parameters_keys

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameters_values(self) -> List[float]:  # cached
        return self.parameters.parameters_values

    @property
    def parameters_bounds(self) -> List[Tuple[float, float]]:
        return self.parameters.parameters_bounds

    @property
    def free_parameters(self) -> List[Parameter]:
        return self.parameters.free_parameters

    @property
    def parameters_values_dict(self):
        return self.parameters.parameters_values_dict

    @property
    def frozen_parameters(self) -> List[Parameter]:
        return self.parameters.frozen_parameters

    @property
    def n_free_parameters(self) -> int:
        return self.parameters.n_free_params

    @property
    def _binary_freeze_map(self) -> List[bool]:
        # possibile da cachare
        return self._parameters._binary_freeze_map
        #if "binary_freeze_map" in self._cache:
        #    return self._cache["binary_freeze_map"]
        #return [p.frozen for p in self]

    @property
    def _binary_melt_map(self) -> List[bool]:
        # possibile da cachare
        return self.parameters._binary_melt_map
        #if "binary_melt_map" in self._cache:
        #    return self._cache["binary_melt_map"]
        #return [not p.frozen for p in self]


    # Simple Model is a leaf
    @property
    def left(self):
        return None

    @property
    def right(self):
        return None

    @property
    def not_frozen_indeces(self):
        return self.parameters.not_frozen_indeces
        #if "not_frozen_indeces" in self._cache:
        #    return self._cache["not_frozen_indeces"]
        #return [
        #    i
        #    for i in range(len(self._binary_freeze_map))
        #    if self._binary_freeze_map[i] is False
        #]

    @property
    def frozen_indeces(self):
        return self.parameters.frozen_indeces
        #if "frozen_indeces" in self._cache:
        #    return self._cache["frozen_indeces"]
        #return [
        #    i
        #    for i in range(len(self._binary_freeze_map))
        #    if self._binary_freeze_map[i] is True
        #]

    def _map_name_to_index(self, name) -> int:
        return self.parameters._map_name_to_index(name)

    def set_parameters_values(self, args=None, **kwargs) -> None:
        """
        Imposta i valori dei parametri utilizzando argomenti posizionali o parole chiave.

        Args:
            args (list, opzionale): Una lista di valori per i parametri.
            kwargs (dict, opzionale): Un dizionario con nomi di parametri come chiavi e valori corrispondenti.



        Esempio:
            >>> obj.set_parameters_values([1, 2, 3])
            >>> obj.set_parameters_values(param1=1, param2=2)
        """
        if args:
            self.parameters.set_values(args)
        if kwargs:
            self.parameters.set_values(kwargs)

    def set_parameters_bounds(self, args=None, **kwargs) -> None:
        """
        Imposta i limiti dei parametri utilizzando argomenti posizionali o parole chiave.

        Args:
            args (list, opzionale): Una lista di limiti per i parametri.
            kwargs (dict, opzionale): Un dizionario con nomi di parametri come chiavi e limiti corrispondenti.


        Esempio:
            >>> obj.set_parameter_bounds([0, 10])
            >>> obj.set_parameter_bounds(param1=(0, 10), param2=(0, 5))
        """
        if args:
            self.parameters.set_bounds(args)
        if kwargs:
            self.parameters.set_bounds(kwargs)

    def _set_frozen_state(self, state: bool, *args) -> None:
        """
        Imposta lo stato di congelamento per i parametri specificati o per tutti i parametri.

        Args:
            state (bool): Stato di congelamento (True per congelare, False per scongelare).
            args (tuple): Una lista di nomi o indici dei parametri da congelare/scongelare.

        Esempio:
            >>> obj._set_frozen_state(True, 'param1', 'param2')
            >>> obj._set_frozen_state(False)
        """
        if not args:
            vals = self.parameters_keys

        else:
            vals = args
        for element in vals:
            name = element
            if isinstance(element, int):
                name = self.parameters._map_indices_to_names(element)
            self.parameters[name].frozen = state

    def freeze_parameters(self, *args, **kwargs) -> None:
        """
        Congela i parametri specificati o tutti i parametri se nessuno è specificato.

        Args:
            args (tuple): Una lista di nomi o indici dei parametri da congelare.
            kwargs (dict): Un dizionario con nomi di parametri come chiavi e valori corrispondenti per congelarli a determinati valori.

        Esempio:
            >>> obj.freeze_parameters('param1', 'param2')
            >>> obj.freeze_parameters(param1=1, param2=2)
        """
        if kwargs:
            # posso freezare un parametro ad un determinato valore
            self.set_parameters_values(kwargs)
            args = [*args, *list(kwargs.keys())]

        self._set_frozen_state(True, *args)

        # AGGIORNO CACHE
        self._update_cache(key="binary_freeze_map", value=[p.frozen for p in self])
        self._update_cache(key="binary_melt_map", value=[not p.frozen for p in self])
        self._update_cache(
            key="not_frozen_indeces",
            value=[
                i
                for i in range(len(self._binary_freeze_map))
                if self._binary_freeze_map[i] is False
            ],
        )
        self._update_cache(
            key="frozen_indeces",
            value=[
                i
                for i in range(len(self._binary_freeze_map))
                if self._binary_freeze_map[i] is True
            ],
        )

    def unfreeze_parameters(self, *args) -> None:
        """
        Scongela i parametri specificati o tutti i parametri se nessuno è specificato.

        Args:
            args (tuple): Una lista di nomi o indici dei parametri da scongelare.

        Esempio:
            >>> obj.unfreeze_parameters('param1', 'param2')
            >>> obj.unfreeze_parameters()
        """
        self._set_frozen_state(False, *args)

        # AGGIORNO CACHE
        self._update_cache(key="binary_freeze_map", value=[p.frozen for p in self])
        self._update_cache(key="binary_melt_map", value=[not p.frozen for p in self])
        self._update_cache(
            key="not_frozen_indeces",
            value=[
                i
                for i in range(len(self._binary_freeze_map))
                if self._binary_freeze_map[i] is False
            ],
        )
        self._update_cache(
            key="frozen_indeces",
            value=[
                i
                for i in range(len(self._binary_freeze_map))
                if self._binary_freeze_map[i] is True
            ],
        )

    @staticmethod
    def _extract_params(method, default_value=1, **kwargs):
        """
        Estrae i nomi e i valori di default dei parametri dal metodo evaluate.

        Parameters:
        -----------
        method : function
            Metodo evaluate della classe.

        Returns:
        --------
        tuple
            Lista dei nomi dei parametri, dei valori di default e dello stato frozen.
        """
        signature = inspect.signature(method)
        params = {}
        is_constant = []
        for param_name, param in signature.parameters.items():
            if param_name != "self":
                if param.default is inspect.Parameter.empty:
                    params[param_name] = default_value
                    is_constant.append(False)
                else:
                    params[param_name] = param.default
                    is_constant.append(True)
        if kwargs:
            print(kwargs)
            for key, value in kwargs.items():
                if key not in params and key != "name":
                    raise ValueError(f"Param {key} is not a function-key")

                params[key] = value

        return list(params.keys()), list(params.values()), is_constant

    @classmethod
    def wrap(cls, func, grid_variables=None, ndim=1, noutputs=1,default_values=1.0, name='SimpleModel'):
        names, values, frozen = cls._extract_params(func, default_values)
        _name = name
        if grid_variables is None:
            _grid = []
        else:
            _grid = grid_variables
            for name in _grid:
                if name not in names:
                    raise ValueError(f'Grid variable {name} is not present in function call')
            
        if ndim is None and grid_variables is not None:
            _ndim = len(_grid)
        else:
            _ndim = 1
                
        if ndim is not None and grid_variables is not None:
            if ndim != len(grid_variables):
                raise ValueError('Number of dimensions do not match number of grid variables')
            _ndim = len(grid_variables)
        
        _ninputs = len(names)   # numero totale di inputs del evaluate
        _noutputs = noutputs

        parameters = ParameterHandler()
        for i in range(len(names)):
            if names[i] not in _grid:
                parameters.add_parameter(
                    Parameter(name=names[i], value=values[i], frozen=frozen[i])
                )
        new_cls = cls(func, parameters, _ndim, _ninputs, _noutputs, _name)
        new_cls._grid_variables = _grid
                
        return new_cls

    
    @classmethod
    def from_callable(
        cls,
        func,
        ndim=None,
        ninputs=None,
        noutputs=1,
        default_value=1,
        name="SimpleModel",
        **kwargs,
    ):
        names, values, frozen = cls._extract_params(
            func, default_value=default_value, **kwargs
        )

        # check sul numero di dimensioni
        if ndim is None:
            ndim = calcola_dimensioni(names)

        if ninputs is None:
            ninputs = len(names) - calcola_dimensioni(names)

        parameters = ParameterHandler()

        for i in range(ndim, len(names)):
            parameters.add_parameter(
                Parameter(name=names[i], value=values[i], frozen=frozen[i])
            )
        # func, parameters, ndim, ninputs, noutputs, name="SimpleModel"

        new_cls = cls(func, parameters, ndim, ninputs, noutputs, name)
        new_cls._grid_variables = names[:ndim]  # Sovrascrive _grid_variables
        new_cls.HAS_GRID = (
            (len(new_cls.grid_variables) > 0)
            if new_cls._grid_variables is not None
            else False
        )
        if ndim == 0:
            new_cls._ndim = 1

        new_cls._ninputs = len(parameters) 
        # new_cls._update_callable()
        return new_cls

    def __str__(self):
        """
        Restituisce una rappresentazione testuale del modello.

        Returns:
            str: Una stringa che rappresenta il modello.
        """
        buffer = StringIO()
        
        # Scrittura delle informazioni generali
        buffer.write(f"MODEL NAME: {self.name} \n")
        buffer.write(f"FREE PARAMS: {self.n_free_parameters}\n")
        buffer.write(f"GRID VARIABLES: {self.grid_variables}\n")
        buffer.write(f"N-DIM: {self.n_dim}\n")
        buffer.write("-" * 60 + "\n")
        buffer.write(f"{'':<4} {'NAME':<15} {'VALUE':<10} {'IS-FROZEN':<10} {'BOUNDS':<20}\n")
        buffer.write("-" * 60 + "\n")
        
        # Scrittura dei parametri
        for i, param in enumerate(self._parameters):
            value_str = f"{param.value:.2f}"
            bounds_str = f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            frz = "Yes" if param.frozen else "No"
            buffer.write(f"{i:<4} {param.name:<15} {value_str:<10} {frz:<10} {bounds_str:<20}\n")
        
        return buffer.getvalue()

    def evaluate(self, *args, **kwargs):
        return self._callable(*args, **kwargs)
    
    
    def __call__(self, *grid, **kwargs):
        
        tmp = self.parameters_values_dict

        if kwargs:
            for key in kwargs:
                if self[key].frozen:
                    warnings.warn(
                        f"Parameter {key} is frozen, new value will be ignored"
                    )
                else:
                    tmp[key] = kwargs[key]

        return self._callable(*grid, **tmp)

    def __getitem__(self, name: str) -> Parameter:
        return self.parameters[name]

    def __setitem__(self, key, value: Parameter) -> None:
        return self.parameters.__setitem__(key, value)

    def __contains__(self, key: str) -> bool:
        return self.parameters.__contains__(key)

    def __iter__(self):
        return self.parameters.__iter__()

    def __len__(self) -> int:
        return self.parameters.__len__()

    def copy(self):
        return deepcopy(self)

    __add__ = componemodels("+")
    __mul__ = componemodels("*")
    __or__ = componemodels("|")
    __truediv__ = componemodels("/")
    __sub__ = componemodels("-")
    __pow__ = componemodels("**") 
    
    




class CompositeModel(Model):
    LINEAR_OPERATIONS = ["+", "-", "*", "/", "**"]
    COMPOSITE_OPERATION = "|"

    IS_COMPOSITE = True
    _parameters = OrderedDict()
    _name = "CompositeModel"
    _callable = None

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def __init__(self, left=None, right=None, op="+") -> None:
        self._left = left
        self._right = right
        self.op_str = op
        self._operator = self.map_operator(op)

        self._n_dim, self._n_inputs, self._noutputs = self._update_n_dim()
        self._parameters = self._init_parameters()
        self.submodels = self._collect_submodels()
        self._cache = {}

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def parameters(self):
        return self._parameters

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._noutputs

    @property
    def grid_variables(self):
        return self.left.grid_variables

    def map_operator(self, op):
        """
        Mappa l'operatore dato a una funzione corrispondente.

        Args:
            op (str): Operatore come stringa.

        Returns:
            function: Funzione corrispondente all'operatore.

        """
        if op == "+":
            val = operator.add
        elif op == "/":
            val = operator.truediv
        elif op == "*":
            val = operator.mul
        elif op == "-":
            val = operator.sub
        elif op == "**":
            val == operator.pow
        else:
            val = None
        return val

    def _update_n_dim(self):
        """
        Controlla gli inputs e gli outputs dei sottomodelli per essere sicuro
        che le operazioni binarie siano supportate.
        corregge il numero di inputs/outputs del modello composito di conseguenza
        """

        if self.op_str in self.LINEAR_OPERATIONS:
            if self.left.n_dim != self.right.n_dim:
                raise ValueError("Number of dimensions do not match!")

            n_dim = self.left.n_dim
            n_inputs = len(self.left.parameters_names) + len(self.right.parameters_names)+len(self.grid_variables)
            n_outputs = self.left.n_outputs

        elif self.op_str == self.COMPOSITE_OPERATION:
            if self.left.n_outputs != len(self.right.grid_variables):
                raise ValueError(
                    "Number of output for left must be equal to number of grid variables of right!"
                )
            # if self.left.n_dim != self.right.n_dim:
            #    raise ValueError("Number of dimensions do not match!")

            n_dim = self.left.n_dim
            n_inputs = self.left.n_inputs
            n_outputs = self.right.n_outputs

        return n_dim, n_inputs, n_outputs

    def _collect_submodels(self):
        param_map = OrderedDict()
        submodels = []

        def dfs(node):
            nonlocal param_map, submodels
            if not node:
                return

            if not node.left and not node.right:
                submodels.append(node.name)

                return

            dfs(node.left)
            dfs(node.right)

        dfs(self)

        return submodels

    def composite_structure(self):
        """
        Restituisce una stringa che rappresenta la logica con cui i sottomodelli sono uniti.

        Returns:
            str: Una stringa che rappresenta la struttura dell'albero binario del modello composito.
        """

        def helper(m, id_counter):
            if isinstance(m, CompositeModel):
                left_str, id_counter = helper(m.left, id_counter)
                right_str, id_counter = helper(m.right, id_counter)
                return f"({left_str} {m.op_str} {right_str})", id_counter
            else:
                return f"{m.name} [{id_counter}]", id_counter + 1

        structure, _ = helper(self, 0)
        return structure

    def _init_parameters(self):
        """Crea un Nuovo ParameterHandler con i nomi cambiati dei parametri ma mappati
        agli stessi parametri originali
        """

        parameters = ParameterHandler()
        n = 0

        def dfs(node):
            nonlocal n, parameters
            if node is None:
                return

            if not node.left and not node.right:
                for param in node:
                    name = param.name + f"_{n}"

                    parameters.add_parameter(param, name=name)
                n += 1

            dfs(node.left)
            dfs(node.right)

        dfs(self)

        return parameters

    def __str__(self):
        """
        Restituisce una stringa che rappresenta il modello composito e i suoi parametri.

        Returns:
            str: Una stringa che rappresenta il modello composito, i modelli contenuti e i parametri liberi.
        """
        buffer = StringIO()

        # Scrivi le informazioni generali del modello
        buffer.write(f"COMPOSITE MODEL NAME: {self.name} \n")
        buffer.write(f"CONTAINED MODELS: {self.submodels}\n")
        buffer.write(f"GRID VARIABLES: {self.grid_variables}\n")
        buffer.write(f"LOGIC: {self.composite_structure()}\n")
        buffer.write(f"FREE PARAMS: {self.n_free_parameters}\n")
        buffer.write("-" * 60 + "\n")
        buffer.write(f"{'':<4} {'NAME':<15} {'VALUE':<10} {'IS-FROZEN':<10} {'BOUNDS':<20} \n")
        buffer.write("-" * 60 + "\n")

        # Scrivi i dettagli dei parametri
        for i, (param_name, param) in enumerate(self.parameters.items()):
            value_str = f"{param.value:.2f}"
            bounds_str = f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            frz = "Yes" if param.frozen else "No"
            buffer.write(
                f"{i:<4} {param_name:<15} {value_str:<10} {frz:<10} {bounds_str:<20}\n"
            )

        # Ritorna il contenuto del buffer
        return buffer.getvalue()


        
    def evaluate(self, *args, **kwargs):
        """
        Valuta il modello utilizzando i valori forniti come input.

        Args:
            *args: Valori posizionali per le variabili della griglia.
            **kwargs: Valori chiave per i parametri.

        Returns:
            Il risultato dell'evaluazione del modello.
        """
        # Costruzione iniziale della griglia e del dizionario dei parametri
        grid = args[: len(self.grid_variables)]
        tmp = {**self.parameters_values_dict, **kwargs}

        # Prepara gli iteratori per dividere i valori di tmp
        tmp_values = iter(tmp.values())
        left_keys, right_keys = self.left.parameters_keys, self.right.parameters_keys

        # Creazione delle sottosezioni per left_vals e right_vals
        left_vals = dict(zip(left_keys, islice(tmp_values, self.left.n_inputs)))
        right_vals = dict(zip(right_keys, tmp_values))

        # Operazioni lineari
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, **left_vals),
                self.right.evaluate(*grid, **right_vals),
            )
        # Operazioni composite
        elif self.op_str in self.COMPOSITE_OPERATION:
            left_res = [self.left.evaluate(*grid, **left_vals)]
            return self.right.evaluate(*left_res)

        # Operazione sconosciuta
        raise ValueError(f"Unknown operation: {self.op_str}")

    def __call__(self, *args, **kwargs):
        #grid = {key: kwargs.pop(key) for key in self.grid_variables}
        grid = args[:len(self.grid_variables)]
        tmp = self.parameters_values_dict

        if kwargs:
            for key in kwargs:
                if self[key].frozen:
                    warnings.warn(
                        f"Parameter {key} is frozen, new value will be ignored"
                    )
                else:
                    tmp[key] = kwargs[key]

        # Pre-calcola i valori di tmp una volta sola
        tmp_values = list(tmp.values())

        # Usa un'unica lista per entrambi i dizionari
        left_vals = {
            key: val for key, val in zip(self.left.parameters_keys, tmp_values)
        }
        right_vals = {
            key: val
            for key, val in zip(
                self.right.parameters_keys, tmp_values[self.left.n_inputs :]
            )
        }

        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, **left_vals),
                self.right.evaluate(*grid, **right_vals),
            )

        elif self.op_str in self.COMPOSITE_OPERATION:
            left_res = [self.left.evaluate(*grid, **left_vals)]

            return self.right.evaluate(*left_res)


