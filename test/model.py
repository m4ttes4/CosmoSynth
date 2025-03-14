from functools import partial
import inspect
from copy import deepcopy
import warnings
from parameter import Constrain, Parameter, ParameterHandler
from typing import Any, Callable, Dict, List,  Tuple
#from io import StringIO
from collections  import OrderedDict
#from itertools import islice
import operator
from tabulate import tabulate

# TODO implements custom models error to better use try: except:
# TODO add constrain support to __call__ in order to call log_prob_func with **kwargs
# NOTE constrain possono anche essere implementati come il pipe operator
# ex: se faccio funzione che prende tutti input del modello, modifica solo n-esimo secondo il constrain
#       e poi faccio pipe con modello.
#       questo vuol dire modificare la logica di pipe per fare in modo che sia corretta secondo questa nuova logica

# TODO: modificare la nuova interfaccia dei constrain per non esporla al utente
'''
TEST: aggiungere un layer che contiene i constrain.
esempio: classe Layer che applica i constrain
model.add constrain
mantenere differenza tra Tied e Functional in modo da considerare correttamente parametri liberi e non
i constrain si aggiungono al modello e non al parametro
l'API deve intercettare il parametro (dentro l'handler) e freezarlo in accordo col constrain
LIMITI: se metto il constrain su modello composito, il sottomodello adesso ha un parametro freezato anche
se preso da solo.

come funziona:

model.add_constrain(Constrain)
dove Constrain è classe che wrappa funzione 
mappa gli args della funzione ai parametri del handler e se ha un match allora model[match].has_coinstrain = True
'''

class ConstrainLayer:
    # per ora è lista  dentro modello
    pass


__all__ = ['Model',
           'CompositeModel']

def componemodels(op, **kwargs) -> Callable[..., 'CompositeModel']:
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
        self._partial = partial(self._callable, **self.parameters_values_dict)
        
        # legacy da unificare
        self.constrains = []
        self.constrains_names = set()

    def _update_cache(self, key=None, value=None) -> None:        
        self._parameters._update_cache(key, value)

        # CompositeModel has callable = None
        if self._callable is not None:
            self._partial = partial(self._callable, **self.parameters_values_dict)

    # POINTER PROPRIETA
    @property
    def parameters(self) -> ParameterHandler:
        return self._parameters

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value) -> None:
        if not isinstance(value, str):
            raise TypeError("New name must be of type string")
        self._name = value

    @property
    def n_dim(self) -> int:
        return self._n_dims

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def grid_variables(self) -> List[str]:
        return self._grid_variables

    @property
    def parameters_names(self) -> List[str]:  
        return self.parameters.parameters_names

    @property
    def parameters_keys(self) -> List[str]:
        return self.parameters.parameters_keys

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameters_values(self) -> List[float]:  
        return self.parameters.parameters_values

    @property
    def parameters_bounds(self) -> List[Tuple[float, float]]:
        return self.parameters.parameters_bounds

    @property
    def free_parameters(self) -> List[Parameter]:
        return self.parameters.free_parameters

    @property
    def parameters_values_dict(self) -> Dict[str, float]:
        return self.parameters.parameters_values_dict

    @property
    def not_free_parameters(self) -> List[Parameter]:
        return self.parameters.not_free_parameters

    @property
    def n_free_parameters(self) -> int:
        return self.parameters.n_free_params

    @property
    def _binary_freeze_map(self) -> List[bool]:
        return self._parameters._binary_freeze_map
        

    @property
    def _binary_melt_map(self) -> List[bool]:
        return self.parameters._binary_melt_map
    
    @property
    def has_constrains(self) -> bool:
        return len(self.constrains) > 0#any(p.is_constrained for p in self)


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
        

    @property
    def frozen_indeces(self) -> List[int]:
        return self.parameters.frozen_indeces
    
    

    def _map_name_to_index(self, name) -> int:
        '''Return the index of the parameter -name- inside the ordered dict'''
        return self.parameters._map_name_to_index(name)

    def set_parameters_values(self, args=None, **kwargs) -> None:
        """
        TODO: usare il __setitems__ direttamente
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
        self._update_cache()
        # AGGIORNO CACHE
        

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
        self._update_cache()
        # AGGIORNO CACHE
        

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
        TODO: change: ndim = number of output of the model?
        Wrap a given function and create a model class instance with specified parameters and grid variables.
        
        TODO: se grid variables e params sono entrambi None, la griglia è automaticamente presa come i primi ndim args

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
        
        # NOTE currently is not pocssible to do in a different way
        for i in range(len(_grid)):
            if names[i] != _grid[i]:
                raise ValueError(f'Grid elements like {_grid[i]} must be defined before the parameters')
        
        new_cls._call_kwargs = names
        #new_cls._tmp_dict = OrderedDict()
        #for call_kwarg in names:
        #    new_cls._tmp_dict[call_kwarg] = 0
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
        field_names = ["INDEX", "NAME", "VALUE", "IS-FREE", "PRIOR", "DESCR"]
        table_data = []

        for i, param in enumerate(self._parameters):
            value_str = f"{param.value:.2f}" if param.value is not None else "None"
            #bounds_str = (
            #    f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            #    if param.bounds is not None
            #    else "None"
            #)
            prior_str = param.prior._get_str()
            frz = "Yes" if param.is_free else "No"

            # Aggiungiamo i dati del parametro come riga
            table_data.append([i, param.name, value_str, frz, prior_str, param.description])

        # Creazione della tabella
        table = tabulate(table_data, headers=field_names, tablefmt="plain")

        # Combina le informazioni generali con la tabella
        return model_info + table

    

    def evaluate(self, *args, **kwargs):
        """
        Chiama la funzione wrappata `_callable` direttamente senza nessun overhead o controllo.
        Questo metodo è pensato per essere utilizzato in situazioni in cui non si ha bisogno
        di logiche aggiuntive su parametri, frozen.
        """
        return self._callable(*args, **kwargs)
    
    def add_constrain(self, constrain:Constrain):
        if not isinstance(constrain, Constrain):
            raise TypeError('New constrain must be of type Constrain')
        
        if constrain.reduce_varys:
            reduced_key = constrain.tied_param
            self[reduced_key]._is_tied = True
        
        if constrain.name in self.constrains_names:
            raise ValueError('This constrain is already present inside the model')
        
        self.constrains.append(constrain)
        self.constrains_names.add(constrain.name)
    
    def remove_constrain(self, constrain:Constrain=None):
        if constrain is None:
            self.constrains = []
            self.constrains_names = set()
        else:
            # ugly loop
            for i in range(len(self.constrains)):
                if self.constrains[i].name == constrain.name:
                    # see if it was reducing vary
                    if self.constrains[i].reduce_varys:
                        self[self.constrains[i].tied_param]._is_tied = False
                    self.constrains.pop(i)
                    self.constrains_names.remove(constrain.name)
                    
            
        
    def _apply_constrains(self, kwargs):
        if self.has_constrains:
            for layer in self.constrains:
                #print('before',kwargs)
                kwargs = layer(kwargs)
                #print('after', kwargs)
        return kwargs
    
    def validate_args(self, args):
        """
        Prepara i parametri finali per la chiamata `_callable`.

        Logica:
        - `args` riempie i parametri liberi nell'ordine in cui sono definiti.
        - I parametri congelati sono aggiunti con i loro valori correnti.
        
        - Il risultato è un unico dizionario `final_args` che viene passato come **kwargs a `_callable`.
        """
        if len(args) != self.n_free_parameters:
            raise ValueError(
                #f"To much args given!. "
                f"expected {self.n_free_parameters}, got {len(args)}."
                "number of args must be equal to number of free parameters"
            )
                        
        # Costruiamo final_args vuoto e lo riempiamo in maniera incrementale
        final_kwargs = {}

        # Parametri liberi (free parameters)
        #free_params = self.free_parameters
        for i, val in enumerate(args):
            final_kwargs[self.free_parameters[i].name] = val

        for p in self.not_free_parameters:
            final_kwargs[p.name] = p.value
        
        # resolve constrains
        # NOTE: from bechmarks this make it go from 0.76sec to 0.8 sec for 100_000 evaluations
        final_kwargs = self._apply_constrains(final_kwargs)
        
                
        return final_kwargs


    def call(self, grid, *args):
        """
        TODO: modificare la function in modo da suportare i parametri tied
        Chiama la funzione `_callable` in un contesto di ottimizzazione:
        - `grid`: variabili di griglia passate come argomenti posizionali.
        - `args`: vettore dei parametri liberi nell'ordine in cui sono definiti.
        - `kwargs`: sovrascrive qualunque parametro, libero o congelato.

        La validazione di `args` e la preparazione degli argomenti finali sono metodi distinti.
        """
        final_args = self.validate_args(args)
        return self._callable(*grid, **final_args)
    
    def __call__(self,*args,**kwargs) -> Any:
        return self._partial(*args, **kwargs)
    
    

    def __getitem__(self, name: str) -> Parameter:
        return self.parameters[name]

    def __setitem__(self, key, value: Parameter) -> None:
        self.parameters.__setitem__(key, value)
        

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
    
    '''
    TODO: add possib. to change single submodel inside the tree
    '''
    
    LINEAR_OPERATIONS = ["+", "-", "*", "/", "**"]
    COMPOSITE_OPERATION = "|"

    IS_COMPOSITE = True
    _parameters = OrderedDict()
    _name = "CompositeModel"
    _callable = None
    constrains = []
    constrains_names = set()
    

    

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
        self._parameters, self._left_kwarg_map, self._right_kwarg_map = (
            self._init_parameters()
        )
        self.submodels = self._collect_submodels()
        
        #legacy to unify
        self.constrains = []
        self.constrains_names = set()

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
            n_inputs = self.left.n_inputs   #wrong?
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
        kwargs_map = OrderedDict()
        n = 0

        def dfs(node):
            nonlocal n, parameters
            if node is None:
                return

            if not node.left and not node.right:
                for (param,key) in zip(node, node.parameters_keys):
                    name = param.name + f"_{n}"

                    parameters.add_parameter(param, name=name)
                    kwargs_map[name] = key
                    param._handler.append(parameters)
                n += 1

            dfs(node.left)
            dfs(node.right)

        dfs(self)
        parameters._is_inside_model = True
        
        
        # Slice per dividere i parametri tra left e right
        
        left_kwargs = OrderedDict()
        right_kwargs = OrderedDict()
        
        for key,val in zip(list(kwargs_map.keys()), self.left.parameters_keys):
            left_kwargs[key] = val
        for key, val in zip(list(kwargs_map.keys())[self.left.n_parameters:], self.right.parameters_keys):
            right_kwargs[key] = val
        
        return parameters, left_kwargs, right_kwargs


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
        field_names = ["INDEX", "NAME", "VALUE", "IS-FREE", "PRIOR", "DESCR"]
        table_data = []

        for i, (param_name, param) in enumerate(self.parameters.items()):
            value_str = f"{param.value:.2f}" if param.value is not None else "None"
            #bounds_str = (
            #    f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            #    if param.bounds is not None
            #    else "None"
            #)
            prior_str = param.prior._get_str()
            frz = "Yes" if param.is_free else "No"

            # Aggiungiamo i dati del parametro come riga
            table_data.append(
                [i, param_name, value_str, frz, prior_str, param.description]
            )

        # Creazione della tabella
        table = tabulate(table_data, headers=field_names, tablefmt="plain")

        # Combina le informazioni generali con la tabella
        return model_info + table
    
    
    def _map_kwargs(self, kwargs):
        """
        Mappa i kwargs rinominati del modello composito a quelli originali
        dei sotto-modelli utilizzando le mappe univoche `left_kwargs_map` e `right_kwargs_map`.

        Args:
            kwargs (dict): Dizionario dei kwargs forniti al modello composito.

        Returns:
            tuple: Due dizionari, uno per `left` e uno per `right`.
        """
        # Mappa i kwargs per `left` utilizzando `left_kwargs_map`
        left_kwargs = {
            original_key: kwargs[composite_key]
            for composite_key, original_key in self._left_kwarg_map.items()
            if composite_key in kwargs
        }

        # Mappa i kwargs per `right` utilizzando `right_kwargs_map`
        right_kwargs = {
            original_key: kwargs[composite_key]
            for composite_key, original_key in self._right_kwarg_map.items()
            if composite_key in kwargs
        }

        return left_kwargs, right_kwargs
    
    def _map_args(self, args):
        """
        Divide gli args tra la funzione `left` e `right`, ignorando gli args della griglia.

        Args:
            args (tuple): Argomenti posizionali forniti al modello composito.

        Returns:
            tuple: Due tuple contenenti gli args per `left` e `right`.
        """
        # Se non ci sono abbastanza args per superare la griglia, restituisci tuple vuote
        if len(args) <= len(self.grid_variables):
            return (), ()

        # Calcola gli args per `left`, tenendo conto dell'offset della griglia
        left_start = len(self.grid_variables)
        left_end = left_start + self.left.n_parameters
        left_args = args[left_start:left_end]

        # Se non ci sono abbastanza args per `right`, restituisci solo `left_args`
        if left_end >= len(args):
            return left_args, ()

        # Calcola gli args per `right`
        right_args = args[left_end:]
        
        return left_args, right_args
    
    def _map_args_to_free_params(self, args):
        #if len(args) <= len(self.grid_variables) or self.n_free_parameters == 0:
        #    return (), ()

        # Calcola gli args per `left`, tenendo conto dell'offset della griglia
        left_end = self.left.n_free_parameters
        left_args = args[:left_end]

        # Se non ci sono abbastanza args per `right`, restituisci solo `left_args`
        if left_end >= len(args):
            return left_args, ()

        # Calcola gli args per `right`
        right_args = args[left_end:]
        return left_args, right_args
    
    
    def evaluate(self, *args, **kwargs):
        
        grid = []
        k = 0
        for i, name in enumerate(self.grid_variables):
            if name in kwargs:
                grid.append(kwargs.pop(name))
            else:
                k += 1
                grid.append(args[i])
        
        left_args, right_args = self._map_args(args)
        left_kwargs, right_kwargs = self._map_kwargs(kwargs)
        
        
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, *left_args, **left_kwargs),
                self.right.evaluate(*grid,*right_args, **right_kwargs),
            )        
        # Se l'operatore è composito
        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = self.left.evaluate(*grid, *left_args,**left_kwargs)
            if isinstance(left_res, tuple):
                return self.right.evaluate(*left_res, *left_args, **right_kwargs)
            else:
                return self.right.evaluate(left_res,*right_args, **right_kwargs)
            
    def call(self, grid, *args):
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
            
        tmp = {}    # do not populate the dict twice but once

        i = 0
        for key in self.parameters_keys:
            if self[key].is_free:
                tmp[key] = args[i]
                i += 1
            else:
                tmp[key] = self.parameters_values_dict[key]

        self._apply_constrains(tmp)
        
        # now solve the constrains
        # for key, param in zip(self.parameters_keys, self):
        #     if param.has_constrain:
        #         # print(
        #         #     f"param {key} with val {tmp[key]} is map to param {param.constrain.param} with val {tmp[param.constrain.param]} to a val of {param.constrain(tmp[param.constrain.param])}"
        #         # )
        #         # print(f'test value 100: {param.constrain(100)}')
        #         tmp[key] = param.constrain(tmp[param.constrain.param])
                
        left, right = self._map_kwargs(tmp)
        
        # Prepara i parametri finali
        #tmp = list({**self.parameters_values_dict}.values())
        
        # Identifica gli indici dei parametri liberi
        #indices = [i for i in range(len(self._binary_melt_map)) if self._binary_melt_map[i]]
        #for idx, value in zip(indices, args):
        #    tmp[idx] = value
            
        
        # Suddivisione dei parametri tra left e right
        # left = {
        #     key: val
        #     for key, val in zip(self.left.parameters_keys, tmp[: self.left.n_parameters])
        # }
        # right = {
        #     key: val
        #     for key, val in zip(self.right.parameters_keys, tmp[self.left.n_parameters:])
        # }
        
        #print(tmp)
        #print(left)
        #print(right)
        
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
            
    # def call(self, grid, *args):
    #     if len(args) != self.n_free_parameters:
    #         raise ValueError(
    #             # f"Troppi argomenti forniti per i parametri liberi. "
    #             f"expected {self.n_free_parameters} args, got {len(args)}."
    #         )
    #     left_args, right_args = self._map_args_to_free_params(args)
        
    #     if self.op_str in self.LINEAR_OPERATIONS:
    #         return self._operator(
    #             self.left.call(grid, *left_args),
    #             self.right.call(grid, *right_args),
    #         )

    #     elif self.op_str == self.COMPOSITE_OPERATION:
    #         left_res = self.left.call(grid, *left_args)
    #         if isinstance(left_res, tuple):
    #             return self.right.call(left_res, *right_args)
    #         else:
    #             return self.right.call([left_res], *right_args)
    
    def __call__(self, *args, **kwargs):
        grid = []
        for i, grid_name in enumerate(self.grid_variables):
            if grid_name not in kwargs:
                grid.append(args[i])
            else:
                grid.append(kwargs.pop(grid_name))
                
        left_vals, right_vals = self._map_kwargs(kwargs)
        #print(left_vals)
        #print(right_vals)
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left(*grid, **left_vals),
                self.right(*grid, **right_vals),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = self.left(*grid, **left_vals)

            if isinstance(left_res, tuple):
                return self.right(*left_res, **right_vals)
            else:
                return self.right(left_res, **right_vals)
        
    
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
                
    
                
    
