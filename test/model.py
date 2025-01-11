import inspect
from copy import deepcopy
import warnings
from parameter import Parameter, ParameterHandler
from typing import Callable, List,  Tuple
#from io import StringIO
from collections  import OrderedDict
from itertools import islice
import operator
from tabulate import tabulate

'''
TODO: nuova gestione delle chiamate
__call__ prende args la griglia, kwargs i parametri
evaluate rimane così (ma da snellire per composite model)
call richiede tanti args quanti parametri liberi
'''

def componemodels(op, **kwargs):
    return lambda left, right: CompositeModel(left, right, op, **kwargs)

class Model:
   
    _name = "SimpleModel"
    _parameters = ParameterHandler()
    _grid_variables = []    # args from evaluate that defines the grid
    _n_dims = 1       #number of dimensions in wich the model is defined
    _n_inputs = 1    # total number of inputs that defines the evaluate call
    _n_outputs = 1   # number of outputs, i.e number of elements returned from evaluate call

    def __init__(self, 
                 func:Callable, 
                 parameters:ParameterHandler, 
                 ndim:int, ninputs:int, noutputs:int, name:str="SimpleModel") -> None:
        
        # TODO rendere Model chiamabile direttamente come wrapper a funzione in stile LMFIT
        # implementare self._initialize_from_callable e self._initilize_from_wrap per i due casi
        self._parameters = parameters
        self._n_dims = ndim
        self._n_inputs = ninputs
        self._n_outputs = noutputs
        self._callable = func
        self._name = name
        self._grid_variables = []  # Inizializza _grid_variables nel costruttore
        #self._cache = {}  # cache base
        

    def _update_cache(self, key, value) -> None:
        '''TODO: odio come ho implementato la cache,
        trova un modo migliore pls'''
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
        return self._n_dims

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

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
    def not_frozen_indeces(self) -> List[int]:
        return self.parameters.not_frozen_indeces
        #if "not_frozen_indeces" in self._cache:
        #    return self._cache["not_frozen_indeces"]
        #return [
        #    i
        #    for i in range(len(self._binary_freeze_map))
        #    if self._binary_freeze_map[i] is False
        #]

    @property
    def frozen_indeces(self) -> List[int]:
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
    def _extract_params(method, default_value=1, **kwargs) -> tuple[list[str], list[float], list[bool]]:
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
    def wrap(cls, func, grid_variables=None, params=None, ndim=None, noutputs=1, default_values=1.0, initial_values:dict|None = None, param_option:dict=None, name='SimpleModel'):
        """
        Wrap a given function and create a model class instance with specified parameters and grid variables.

        Parameters
        ----------
        func : callable
            The function to be wrapped. It should accept a set of parameters that will be defined
            as either grid variables or free parameters.
        grid_variables : iterable of str, optional
            The names of the variables that represent grid dimensions. If not provided and `params` is given,
            the grid variables will be inferred as all arguments of the function that are not in `params`.
        params : iterable of str, optional
            The names of the parameters that are non-grid variables (i.e., free parameters).
            If not provided and `grid_variables` is given, the parameters will be inferred as all arguments
            of the function that are not in `grid_variables`.
        ndim : int, optional
            The number of dimensions (based on `grid_variables`). If None, it will be inferred 
            from `grid_variables`. Default is 1.
        noutputs : int, optional
            The number of outputs the wrapped function returns. Default is 1.
        default_values : float or iterable of floats, optional
            Default initial values for the parameters. If a single float is given, it will be 
            used for all parameters. Default is 1.0.
        name : str, optional
            The name of the model. Default is 'SimpleModel'.

        Returns
        -------
        object
            An instance of the class, with attributes and parameters set according to the provided arguments.

        Raises
        ------
        ValueError
            If a specified grid variable or parameter is not present in the function arguments.
            If both `params` and `grid_variables` are None.
            If the number of grid variables does not match `ndim`.
            If the total number of function arguments does not match the sum of the number of 
            grid variables and parameters.
        """

        # Extract the parameter names, values, and frozen status from the function
        names, values, frozen = cls._extract_params(func, default_values)
        # Check that we have at least one between params and grid_variables
        if params is None and grid_variables is None:
            _grid = []
            _params = names
            #raise ValueError("At least one between 'params' and 'grid_variables' must be provided.")
        else:
            # Convert grid_variables and params to lists if they are not None
            _grid = list(grid_variables) if grid_variables is not None else None
            _params = list(params) if params is not None else None

            # If grid_variables is given but params is None, infer params
            if _grid is not None and _params is None:
                # Check that all grid variables are in names
                for gv in _grid:
                    if gv not in names:
                        raise ValueError(f'Grid variable {gv} is not present in function call')
                _params = [n for n in names if n not in _grid]

            # If params is given but grid_variables is None, infer grid_variables
            if _params is not None and _grid is None:
                # Check that all params are in names
                for p in _params:
                    if p not in names:
                        raise ValueError(f'Parameter {p} is not present in function call')
                _grid = [n for n in names if n not in _params]

        # Now both _grid and _params are defined
        # Check consistency between ndim and grid_variables
        if ndim is None:
            _n_dims = len(_grid)
        else:
            _n_dims = ndim
            if len(_grid) != _n_dims:
                raise ValueError('Number of dimensions (ndim) does not match the number of grid variables')

        # Check that all given grid variables are in names
        for gv in _grid:
            if gv not in names:
                raise ValueError(f'Grid variable {gv} is not present in function call')

        # Check that all given params are in names
        for p in _params:
            if p not in names:
                raise ValueError(f'Parameter {p} is not present in function call')

        # Check total number of arguments
        if len(names) != len(_grid) + len(_params):
            raise ValueError(
                f"The total number of function arguments ({len(names)}) must equal "
                f"the number of grid variables ({len(_grid)}) plus the number of parameters ({len(_params)})"
            )

        _n_inputs = len(names)   # total number of inputs to evaluate
        _n_outputs = noutputs

        parameters = ParameterHandler()
        # Add parameters (not grid variables) to the ParameterHandler
        for i, n in enumerate(names):
            if n in _params:
                parameters.add_parameter(
                    Parameter(name=n, value=values[i], frozen=frozen[i])
                )
        # adjuct for initial values
        if initial_values is not None:
            if not isinstance(initial_values, dict):
                raise TypeError("Initial values for parames mus be of type dict <str:val>")
            
            for pname, pval in initial_values.items():
                if pname in parameters:
                    parameters[pname].value = pval
                else:
                    warnings.warn(f'parameter name {pname} is not a parameter for the model')
        
        
        # update with param options
        if param_option is not None:
            if not isinstance(param_option, dict):
                raise TypeError('Param Options must be istance of Dict')
            option_keys = ['prior', 'value', 'frozen','description','bounds']
            
            for key in param_option.keys():
                if key in parameters:
                    # loop over possible options
                    for option in option_keys:
                        if option in param_option[key]:
                            parameters[key][option] = param_option[key][option]
            
        parameters._is_inside_model = True
        parameters._update_cache()
        new_cls = cls(func, parameters, _n_dims, _n_inputs, _n_outputs, name)
        new_cls._grid_variables = _grid
        new_cls._tmp_dict = OrderedDict()
        for call_kwarg in names:
            new_cls._tmp_dict[call_kwarg] = 0
        # Ord dict per zippare gli args nelle giuste kwargs
        #{call_kwarg:1 for call_kwarg in names}
        return new_cls



    

    def __str__(self):
        """
        Restituisce una rappresentazione testuale del modello.

        Returns:
            str: Una stringa che rappresenta il modello.
        """
        # Informazioni generali sul modello
        model_info = (
            f"MODEL NAME: {self.name}\n"
            f"FREE PARAMS: {self.n_free_parameters}\n"
            f"GRID VARIABLES: {self.grid_variables}\n"
            f"N-DIM: {self.n_dim}\n"
            + "-" * 100 + "\n"
        )

        # Creazione dei dati per la tabella
        field_names = ["INDEX", "NAME", "VALUE", "FROZEN", "PRIOR", "DESCR"]
        table_data = []

        for i, param in enumerate(self._parameters):
            value_str = f"{param.value:.2f}" if param.value is not None else "None"
            #bounds_str = (
            #    f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            #    if param.bounds is not None
            #    else "None"
            #)
            prior_str = param.prior._get_str()
            frz = "Yes" if param.frozen else "No"

            # Aggiungiamo i dati del parametro come riga
            table_data.append([i, param.name, value_str, frz, prior_str, param.description])

        # Creazione della tabella
        table = tabulate(table_data, headers=field_names, tablefmt="plain")

        # Combina le informazioni generali con la tabella
        return model_info + table

    

    def evaluate(self, *args, **kwargs):
        """
        TODO: add support for jax and autodiff
        Chiama la funzione wrappata `_callable` direttamente senza nessun overhead o controllo.
        Questo metodo è pensato per essere utilizzato in situazioni in cui non si ha bisogno
        di logiche aggiuntive su parametri, frozen.
        """
        return self._callable(*args, **kwargs)
    
    def validate_args(self, args, kwargs):
        """
        Prepara i parametri finali per la chiamata `_callable`.

        Logica:
        - `args` riempie i parametri liberi nell'ordine in cui sono definiti.
        - I parametri congelati sono aggiunti con i loro valori correnti.
        - `kwargs` può sovrascrivere qualsiasi parametro.
        - Il risultato è un unico dizionario `final_args` che viene passato come **kwargs a `_callable`.
        """
        if len(args) != self.n_free_parameters:
            raise ValueError(
                #f"To much args given!. "
                f"expected {self.n_free_parameters}, got {len(args)}."
                "number of args must be equal to number of free parameters"
            )
            
        '''final_kwargs = {**self.parameters_values_dict}
        
        j = 0
        for key in self.parameters_keys:
            if not self[key].frozen:
                final_kwargs[key] = args[j]
        
        final_kwargs.update(**kwargs)'''
            
        # Costruiamo final_args vuoto e lo riempiamo in maniera incrementale
        final_kwargs = {}

        # Parametri liberi (free parameters)
        free_params = self.free_parameters
        for i, val in enumerate(args):
            final_kwargs[free_params[i].name] = val

        # Parametri congelati (frozen parameters)
        # Qui non creiamo nuovi oggetti, iteriamo direttamente sui parametri congelati
        # e assegnamo i valori al dizionario.
        for p in self.frozen_parameters:
            # Se il param. era già stato impostato via args (potenzialmente non avviene mai),
            # viene sovrascritto qui con il valore congelato.
            final_kwargs[p.name] = p.value

        # Sovrascrivi con kwargs (eventuali parametri liberi o congelati)
        # Qui applichiamo direttamente gli aggiornamenti su final_kwargs.
        for k, v in kwargs.items():
            final_kwargs[k] = v

        return final_kwargs


    def call(self, grid, *args, **kwargs):
        """
        Chiama la funzione `_callable` in un contesto di ottimizzazione:
        - `grid`: variabili di griglia passate come argomenti posizionali.
        - `args`: vettore dei parametri liberi nell'ordine in cui sono definiti.
        - `kwargs`: sovrascrive qualunque parametro, libero o congelato.

        La validazione di `args` e la preparazione degli argomenti finali sono metodi distinti.
        """
        final_args = self.validate_args(args, kwargs)
        return self._callable(*grid, **final_args)
        
    '''
    def call(self, grid, *args, **kwargs):
        """
        Chiama la funzione wrappata con gli argomenti forniti.
        
        Args:
            *args: Valori per i parametri non congelati, forniti in ordine.
        
        Returns:
            Il risultato della funzione originale con i parametri congelati.
        """
        if len(args) > self.n_free_parameters+len(self.grid_variables):
            raise ValueError(f"Troppi argomenti forniti. Aspettati al massimo {self.n_free_parameters}.")
        
        # BUG: e se uno definisce le grid variables dopo gli args? --> devo usare kwargs        
        #grid = args[:len(self.grid_variables)]
        # Mappa gli args sui parametri non congelati
        #provided_args = dict(zip(self.unfrozen_params, args))
        provided_args = {p.name:arg for p,arg in zip(self.free_parameters,args)}
        frozen_params = {p.name:p.value for p in self.frozen_parameters}
        # Combina i parametri forniti con quelli congelati
        final_args = {**frozen_params, **provided_args, **kwargs}
        
        # Ordina gli argomenti secondo l'ordine della funzione originale
        #ordered_args = [final_args[param] for param in self.all_params]
        #print('chiamata con,', final_args)
        
        # Chiama la funzione originale
        return self._callable(*grid ,**final_args)'''
        
        
    def __call__(self, *args, **kwargs):
        '''
        Si aspetta che la griglia sia fornita come i primi args, altrimenti 
        deve essere data rta i kwargs
        '''
        
        for i, grid_name in enumerate(self.grid_variables):
            if grid_name not in kwargs:
                self._tmp_dict[grid_name] = args[i]
                
        self._tmp_dict.update(**self.parameters_values_dict)
        self._tmp_dict.update(**kwargs)

        return self._callable(**self._tmp_dict)

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
    
    # PRIOR LOGIC, should not be used for simple models
    
    




class CompositeModel(Model):
    LINEAR_OPERATIONS = ["+", "-", "*", "/", "**"]
    COMPOSITE_OPERATION = "|"

    IS_COMPOSITE = True
    _parameters = OrderedDict()
    _name = "CompositeModel"
    _callable = None

    def __init_subclass__(cls) -> None:
        ## deprecated
        return super().__init_subclass__()

    def __init__(self, left=None, right=None, op="+") -> None:
        self._left = left
        self._right = right
        
        ## check models
        if not isinstance(left, Model):
            raise ValueError(f"CompositeModel: argument {left} is not a Model")
        if not isinstance(right, Model):
            raise ValueError(f"CompositeModel: argument {right} is not a Model")
        
        self.op_str = op
        self._operator = self.map_operator(op)

        self._n_dim, self._n_inputs, self._n_outputs = self._update_n_dim()
        self._parameters = self._init_parameters()
        self.submodels = self._collect_submodels()
        #self._tmp_dict = OrderedDict()
        #for grid_kwarg in self.left.grid_variables:
        #    self._tmp_dict[grid_kwarg] = 0.0
        #self._tmp_dict.update(**self.parameters_values_dict)
        #self._cache = {}

    @property
    def left(self) -> Model|None:
        return self._left

    @property
    def right(self) -> Model|None:
        return self._right

    @property
    def parameters(self) -> ParameterHandler:
        return self._parameters

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def grid_variables(self) -> List[str]:
        # always propagate the left branch
        return self.left.grid_variables

    def map_operator(self, op) -> Callable|None:
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
            val = operator.pow
        else:
            val = None
        return val

    def _update_n_dim(self) -> Tuple[int, int, int]:
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

    def _collect_submodels(self) -> List[Model]:
        #param_map = OrderedDict()
        submodels = []

        def dfs(node):
            nonlocal submodels  # ,param_map
            if not node:
                return

            if not node.left and not node.right:
                submodels.append(node.name)

                return

            dfs(node.left)
            dfs(node.right)

        dfs(self)

        return submodels

    def composite_structure(self) -> str:
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

    def _init_parameters(self) -> ParameterHandler:
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
        parameters._is_inside_model = True
        return parameters


    def __str__(self):
        """
        Restituisce una stringa che rappresenta il modello composito e i suoi parametri.

        Returns:
            str: Una stringa che rappresenta il modello composito, i modelli contenuti e i parametri liberi.
        """
        # Informazioni generali del modello composito
        model_info = (
            f"COMPOSITE MODEL NAME: {self.name}\n"
            f"CONTAINED MODELS: {', '.join(self.submodels)}\n"
            f"GRID VARIABLES: {self.grid_variables}\n"
            f"LOGIC: {self.composite_structure()}\n"
            f"FREE PARAMS: {self.n_free_parameters}\n" + "-" * 60 + "\n"
        )

        # Creazione dei dati per la tabella
        field_names = ["INDEX", "NAME", "VALUE", "FROZEN", "PRIOR", "DESCR"]
        table_data = []

        for i, (param_name, param) in enumerate(self.parameters.items()):
            value_str = f"{param.value:.2f}" if param.value is not None else "None"
            #bounds_str = (
            #    f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            #    if param.bounds is not None
            #    else "None"
            #)
            prior_str = param.prior._get_str()
            frz = "Yes" if param.frozen else "No"

            # Aggiungiamo i dati del parametro come riga
            table_data.append(
                [i, param_name, value_str, frz, prior_str, param.description]
            )

        # Creazione della tabella
        table = tabulate(table_data, headers=field_names, tablefmt="plain")

        # Combina le informazioni generali con la tabella
        return model_info + table
    
    
    
    
    def evaluate(self, *args, **kwargs):
        """
        Valuta il modello composito utilizzando i valori forniti come input.

        Questo metodo si limita a comporre il risultato dell'evaluate del modello `left` e `right`
        utilizzando l'operatore specificato. La composizione è trasparente, ossia i parametri del
        modello composito vengono suddivisi tra `left` e `right`, e poi i due risultati vengono
        combinati. Nel caso di un'operazione composita (COMPOSITE_OPERATION), l'output di `left`
        viene passato come input a `right`.

        Args:
            *args: Valori posizionali per le variabili della griglia. Il numero di variabili di griglia
                è tipicamente `len(self.grid_variables)`.
            **kwargs: Coppie chiave=valore per impostare i parametri del modello. Eventuali parametri
                    forniti qui sovrascrivono quelli già presenti in `parameters_values_dict`.

        Returns:
            Il risultato dell'evaluazione del modello composito, che può essere uno scalare o un array
            a seconda della funzione `_callable` di `left` e `right`.
        """
        # Costruzione della griglia e del dizionario dei parametri
        grid = args[: len(self.grid_variables)]
        #tmp = {**self.parameters_values_dict, **kwargs}
        tmp = {key:val for key,val in zip(self.parameters_keys, args[len(grid):])}
        tmp.update(**kwargs)

        # Prepara gli iteratori per dividere i parametri tra left e right
        tmp_values = iter(tmp.values())
        left_keys, right_keys = self.left.parameters_keys, self.right.parameters_keys

        # Dividi i parametri tra left e right
        left_vals = dict(zip(left_keys, islice(tmp_values, self.left.n_inputs)))
        right_vals = dict(zip(right_keys, tmp_values))

        # Se l'operatore è lineare
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, **left_vals),
                self.right.evaluate(*grid, **right_vals),
            )

        # Se l'operatore è composito
        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = self.left.evaluate(*grid, **left_vals)
            if isinstance(left_res, tuple):
                return self.right.evaluate(*left_res, **right_vals)
            else:
                return self.right.evaluate(left_res, **right_vals)

        # Operatore sconosciuto
        raise ValueError(f"Unknown operation: {self.op_str}")


    def call(self, grid, *args, **kwargs):
        """
        Chiama il modello in un contesto, ad esempio, di ottimizzazione,
        dove `args` rappresentano i valori per i parametri liberi nell'ordine in cui sono definiti.

        Logica:
        - `grid`: variabili di griglia passate come primo argomento.
        - `args`: valori per i parametri liberi (non congelati), nell'ordine in cui appaiono.
        - `kwargs`: può sovrascrivere i parametri (sia liberi che congelati) fornendo coppie chiave=valore.

        I parametri complessivi del modello vengono ricavati da `parameters_values_dict`,
        aggiornati con `kwargs`, e i parametri liberi vengono sostituiti con quelli passati in `args`.
        Infine, il modello `left` e `right` vengono chiamati ricorsivamente e combinati secondo
        l'operatore `op_str`.

        Args:
            grid: variabili di griglia, in genere un insieme di coordinate o input indipendenti.
            *args: valori per i parametri liberi nell'ordine definito dai parametri stessi.
            **kwargs: coppie chiave=valore per sovrascrivere qualsiasi parametro del modello.

        Returns:
            Il risultato della chiamata al modello composito, combinando `left` e `right` attraverso
            l'operatore lineare o composito.
        """
        if len(args) != self.n_free_parameters:
            raise ValueError(
                #f"Troppi argomenti forniti per i parametri liberi. "
                f"expected {self.n_free_parameters} args, got {len(args)}."
            )


        # Prepara i parametri finali
        tmp = list({**self.parameters_values_dict, **kwargs}.values())

        # Identifica gli indici dei parametri liberi
        indices = [i for i in range(len(self._binary_melt_map)) if self._binary_melt_map[i]]
        for idx, value in zip(indices, args):
            tmp[idx] = value

        # Suddivisione dei parametri tra left e right
        left = {
            key: val
            for key, val in zip(self.left.parameters_keys, tmp[: self.left.n_parameters])
        }
        right = {
            key: val
            for key, val in zip(self.right.parameters_keys, tmp[self.left.n_parameters:])
        }
        #print(left)
        #print(tmp, tmp[self.left.n_parameters :])
        #print(right)
        # Chiamata ricorsiva ai modelli figli
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, **left),
                self.right.evaluate(*grid, **right),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = self.left.evaluate(*grid, **left)
            if isinstance(left_res, tuple):            
                return self.right.evaluate(*left_res, **right)
            else:
                return self.right.evaluate(left_res, **right)

        raise ValueError(f"Unknown operation: {self.op_str}")

    
    
    def __call__(self, *args, **kwargs):
        """
        Chiama il modello composito come fosse una funzione.

        Questo metodo è pensato per un utilizzo più "diretto" e user-friendly. Accetta:
        - `args`: I primi `len(self.grid_variables)` argomenti vengono interpretati come variabili di griglia.
        - `kwargs`: Può contenere valori per parametri che non sono congelati. Se un parametro è congelato,
        verrà emesso un warning e il valore fornito sarà ignorato.

        Logica:
        1. Estrae le grid variables dagli args.
        2. Utilizza `parameters_values_dict` come base per i parametri.
        3. Aggiorna i parametri non congelati con eventuali `kwargs`.
        4. Suddivide i parametri aggiornati tra `left` e `right`.
        5. Chiama `left.evaluate` e `right.evaluate` componendo i risultati con l'operatore.

        Args:
            *args: Valori posizionali, i primi `len(self.grid_variables)` sono le variabili di griglia.
            **kwargs: Eventuali coppie chiave=valore per aggiornare i parametri non congelati.

        Returns:
            Il risultato della funzione combinata `left` e `right` secondo l'operatore `op_str`.
        """
        grid = []
        for i,grid_name in enumerate(self.grid_variables):
            if grid_name not in kwargs:
                grid.append(args[i])
            else:
                grid.append(kwargs.pop(grid_name))
        #grid = args[: len(self.grid_variables)]
        
        tmp = self.parameters_values_dict

        if kwargs:
            for key in kwargs:
                if self[key].frozen:
                    warnings.warn(f"Parameter {key} is frozen, new value will be ignored")
                else:
                    tmp[key] = kwargs[key]
        
        # Pre-calcola i valori di tmp una volta sola
        tmp_values = list(tmp.values())
        
        # Suddividi i parametri tra left e right
        
            
        left_vals = {key: val for key, val in zip(self.left.parameters_keys, tmp_values)}
        right_vals = {
            key: val
            for key, val in zip(
                self.right.parameters_keys, tmp_values[len(self.left.parameters_keys):]
            )
        }
        
        
        if self.op_str in self.LINEAR_OPERATIONS:
            
            return self._operator(
                self.left.evaluate(*grid, **left_vals),
                self.right.evaluate(*grid, **right_vals),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = self.left.evaluate(*grid, **left_vals)
            
            if isinstance(left_res, tuple):                
                return self.right.evaluate(*left_res, **right_vals)
            else:
                return self.right.evaluate(left_res, **right_vals)

        raise ValueError(f"Unknown operation: {self.op_str}")
    
    def print_tree(self, prefix: str = "", is_last: bool = True) -> None:
        """
        Stampa la struttura del CompositeModel in stile 'tree'.
        `prefix` è la stringa di indentazione.
        `is_last` indica se il nodo corrente è l'ultimo figlio del padre.
        """

        # Determina il simbolo di "ramo"
        if prefix == "":
            # Siamo alla radice: niente '|--' o '`--'
            branch_symbol = ""
        else:
            branch_symbol = "`-- " if is_last else "|-- "

        # Definisci l'etichetta del nodo. Se preferisci, usa anche self.name
        node_label = f"Composite(op='{self.op_str}')"

        # Stampa il nodo
        print(f"{prefix}{branch_symbol}{node_label}")

        # Calcola il prefisso per i figli
        new_prefix = prefix + ("    " if is_last else "|   ")

        # Costruisci la lista dei figli effettivi (possono essere 0, 1 o 2)
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)

        # Se non ci sono figli, interrompi (capita se left/right == None)
        if not children:
            return  # modello composito "monco" che non ha figli

        # Altrimenti, itera sui figli
        for i, child in enumerate(children):
            # Verifica se è l'ultimo figlio
            child_is_last = i == len(children) - 1

            # Se il figlio è un CompositeModel, ricorsione
            if isinstance(child, CompositeModel):
                child.print_tree(prefix=new_prefix, is_last=child_is_last)
            else:
                # Se è un Model foglia, stampiamo semplicemente il suo nome
                leaf_symbol = "`-- " if child_is_last else "|-- "
                print(f"{new_prefix}{leaf_symbol}{child.name}")
