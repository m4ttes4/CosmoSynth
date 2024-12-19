import inspect
from copy import deepcopy
import warnings
from parameter import Parameter, ParameterHandler
from typing import List,  Tuple
from io import StringIO
from collections  import OrderedDict
from itertools import islice
import operator



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
        """
        Chiama la funzione wrappata `_callable` direttamente senza nessun overhead o controllo.
        Questo metodo è pensato per essere utilizzato in situazioni in cui non si ha bisogno
        di logiche aggiuntive su parametri, frozen, ecc.
        """
        return self._callable(*args, **kwargs)
    
    def validate_args(self, args, kwargs):
        """
        Prepara i parametri finali per la chiamata `_callable`.
        Si assume che la validazione su 'args' sia già stata fatta.

        Logica:
        - `args` riempie i parametri liberi nell'ordine in cui sono definiti.
        - I parametri congelati sono aggiunti con i loro valori correnti.
        - `kwargs` può sovrascrivere qualsiasi parametro.
        - Il risultato è un unico dizionario `final_args` che viene passato come **kwargs a `_callable`.
        """
        if len(args) > self.n_free_parameters:
            raise ValueError(
                f"Troppi argomenti forniti. "
                f"Attesi al massimo {self.n_free_parameters}, ricevuti {len(args)}."
            )
        
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
        
        tmp = self.parameters_values_dict
        if kwargs:
            for key in kwargs:
                if self[key].frozen:
                    warnings.warn(
                        f"Parameter {key} is frozen, new value will be ignored"
                    )
                else:
                    tmp[key] = kwargs[key]

        return self._callable(*args, **tmp)

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
            val = operator.pow
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
        tmp = {**self.parameters_values_dict, **kwargs}

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
            left_res = [self.left.evaluate(*grid, **left_vals)]
            return self.right.evaluate(*grid, *left_res)

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
        if len(args) > self.n_free_parameters:
            raise ValueError(
                f"Troppi argomenti forniti per i parametri liberi. "
                f"Attesi al massimo {self.n_free_parameters}, ricevuti {len(args)}."
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
            for key, val in zip(self.right.parameters_keys, tmp[self.left.n_parameters :])
        }

        # Chiamata ricorsiva ai modelli figli
        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.call(grid, **left),
                self.right.call(grid, **right),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = [self.left.call(grid, **left)]
            return self.right.call(*left_res)

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
        grid = args[: len(self.grid_variables)]
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
                self.right.parameters_keys, tmp_values[self.left.n_inputs :]
            )
        }

        if self.op_str in self.LINEAR_OPERATIONS:
            return self._operator(
                self.left.evaluate(*grid, **left_vals),
                self.right.evaluate(*grid, **right_vals),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_res = [self.left.evaluate(*grid, **left_vals)]
            return self.right.evaluate(*left_res)

        raise ValueError(f"Unknown operation: {self.op_str}")
