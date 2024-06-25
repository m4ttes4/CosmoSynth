from .parameters import Parameter, ParameterHandler
import inspect
import re
from copy import deepcopy
from typing import Callable, Tuple, Dict, List, Iterable
import operator
from itertools import compress


def discrimina_iterabili(iterabile):
    '''funzione comoda per discriminare il tipo di iterabile'''
    if isinstance(iterabile, dict):
        return 'dizionario'
    elif isinstance(iterabile, list):
        return 'lista'
    elif hasattr(iterabile, '__iter__') and not isinstance(iterabile, (str, dict)):
        return 'iterabile'
    else:
        return 'altro tipo'
    
def componemodels(op, **kwargs):
    return lambda left, right: CompositeModel(left, right, op, **kwargs)


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


# NOTE appunti di cose da migliorare
#   un modello che ha un parametro da definire dentro __init__, allora il parametro non compare dentro alla lista
#   l'evaluation di modelli senza output non è contemplata, come per esempio delle griglie su cui posso definire i modelli
#
#   -- rendere più elastica la __call__ perchè attualmente gestisce solo parametri come liste o dict (idea... creare un wrapper alla funzione)
#   
#   -- cachare i risultati fissi come numero di parametri e simili per non doversi ricalcolare
#
#   -- aggiungere constrains in forma funzinale ai parametri   
#
#   -- MORE FASSSST 
#
#   -- operare in-place quando si crea un modello composito (no creare copie)
#
#   -- BUG possibile: la discriminazione dei tipi di iterabili per ora prevede liste/dict/np.array
    # ma non prevede iterabili generici come tuple,deque,sets. array.array, quindi si deve trovare metodo più generico
#
#---------------------------------------------------------------------------------------------------------
#   #NOTE gestione migliore del tree:
#    1- definizione stretta di foglie e nodi, FittableModel.isleaf = True, CompositeModel.isleaf=False
#       utile per ripercorrere il tree e ricostruire meglio la ligica
#   
#   #NOTE usare un collections.userdict come base class per MParameterHandler per impedire la 
#       rimozione di parametri dal dizionario da parte del utente 
#
#
#----------------------------------------------------------------------------------------------------------
class ModelMeta(type):
    def __new__(cls, name, bases, dct, **kwargs):
        """
        Crea una nuova classe modello con attributi specifici.
        ATTUALMENTE si occupa solo di creare il dizionario di parametri
        in futuro potrebbe essere inutile

        Parameters:
        -----------
        name : str
            Nome della classe.
        bases : tuple
            Basi della classe.
        dct : dict
            Dizionario degli attributi della classe.
        kwargs : dict
            Argomenti aggiuntivi.

        Returns:
        --------
        type
            Nuova classe tipo.
        """
        new_cls = super().__new__(cls, name, bases, dct, **kwargs)
        _param_names = []

        if name == "ModelMeta":
            return new_cls

        _param_dict = ParameterHandler()

        _n_dim: int = getattr(new_cls, "N_DIMENSIONS", 0)
        _n_inputs: int = getattr(new_cls, "N_INPUTS", 0)
        _n_outputs: int = getattr(
            new_cls, "N_OUTPUTS", 1
        )  # default è avere 1 outputs ovviamente
        _is_composite: bool = getattr(new_cls, "IS_COMPOSITE", False)

        if "evaluate" in dct and not _is_composite:
            _param_names, _param_default, _is_constant = cls._extract_params(
                dct["evaluate"]
            )

            # controllo se tra i primi 3 ci sono 'x', 'y', 'z' in questa sucessione
            # se e solo se _n_inpputs, _n_outputs e _n_dim sono tutti 0
            if _n_dim == 0 and _n_inputs == 0 and _n_outputs == 1:
                if len(_param_names) > 0:
                    _n_dim = calcola_dimensioni(_param_names)
                    _n_inputs = calcola_dimensioni(_param_names)

            for _name, _val, _const in zip(
                _param_names[_n_inputs:],
                _param_default[_n_inputs:],
                _is_constant[_n_inputs:],
            ):
                _param_dict._add_parameter(Parameter(_name, _val, frozen=_const))

        new_cls._param_dict = _param_dict
        new_cls._n_inputs = _n_inputs
        new_cls._n_outputs = _n_outputs
        new_cls._name = name
        new_cls._is_composite = _is_composite
        new_cls._n_dim = _n_dim
        new_cls._grid = _param_names[:_n_inputs] if _param_names else []

        return new_cls

    def __init__(cls, name, bases, dct, **kwargs):
        """
        Inizializza una nuova classe modello.

        Parameters:
        -----------
        name : str
            Nome della classe.
        bases : tuple
            Basi della classe.
        dct : dict
            Dizionario degli attributi della classe.
        kwargs : dict
            Argomenti aggiuntivi.
        """
        super().__init__(name, bases, dct, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Crea una nuova istanza del modello.

        Parameters:
        -----------
        args : tuple
            Argomenti posizionali.
        kwargs : dict
            Argomenti keyword.

        Returns:
        --------
        instance
            Nuova istanza del modello.
        """
        instance = super().__call__(*args, **kwargs)

        IGNORE_KWARGS = ["name", "parameters"]

        name = kwargs.get("name", instance._name)

        # trovare un modo per differenziare kwargs da nome del metodo initi

        # if "parameters" in instance.__dict__ and not instance._is_composite:
        #    param_dict = instance.__dict__["parameters"]
        # else:
        #    param_dict = deepcopy(instance._param_dict)
        param_dict = deepcopy(instance._param_dict)

        if "parameters" in kwargs and not instance._is_composite:
            self._validate_parameters(
                kwargs["parameters"], param_dict, instance._param_dict
            )

        # possibile errore; che succede se do lo stesso parametro come 'parameters' e kwargs?
        # per ora vincono le kwargs
        for p_name in kwargs:
            if p_name in IGNORE_KWARGS:
                continue

            if p_name in param_dict:
                param_dict[p_name].value = kwargs[p_name]
            else:
                raise ValueError(f"key {p_name} is not a parameter for the model!")

        if not instance._is_composite:
            instance._parameters = param_dict
            instance._parameters_names = list(param_dict)

            instance.n_dim = instance._n_dim
            instance.n_inputs = instance._n_inputs
            instance.n_outputs = instance._n_outputs
            # instance.wrapped_call = self.wrapped_call(instance.evaluate, param_dict)

        instance.name = name
        instance._parameters._is_inside_model = True
        instance._grid_variables = instance._grid
        # print(type(param_dict)) ok è ParameterHandler
        if hasattr(instance, "__post_init__"):
            # Piccolo rant personale: se il me del futuro dovesse modificare qualcosa
            # e questa call non dovesse essere più giusta allora posso anche uccidermi (LOL)
            # __init__ della metaclasse non finisce nel __init__ del istanza e quindi devo
            # baipassarlo per cachare la call iniziale per permettere al modello di essere creato
            # e chiamato subito con dei parametri freezati
            
            # rant dal me del futuro, non c'è più bisogno di cachare un wrapper della __call__ perchè 
            # è la call stessa ad eseguire i controlli sui parametri (dovrò migliorare)
            instance.__post_init__()
            # btw il __post_init__ può essere usato da CompositeModel anche
        return instance

    @staticmethod
    def _extract_params(method):
        """
        Estrae i nomi e i valori di default dei parametri dal metodo evaluate.

        Parameters:
        -----------
        method : function
            Metodo evaluate della classe.

        Returns:
        --------
        tuple
            Lista dei nomi dei parametri e lista dei valori di default.
        """
        signature = inspect.signature(method)
        params = {}
        is_constant = []
        for param_name, param in signature.parameters.items():
            if param_name != "self":
                if param.default is inspect.Parameter.empty:
                    params[param_name] = 1

                    is_constant.append(False)
                else:
                    params[param_name] = param.default
                    is_constant.append(True)
        return list(params.keys()), list(params.values()), is_constant

    @staticmethod
    def wrapped_call(func: Callable, _parameters: ParameterHandler):
        """Genera la call dei parametri congelati."""
        freeze_params = {p.name: p.value for p in _parameters if p.frozen is True}
        frozen_param_names = freeze_params.keys()
        free_param_names = [p.name for p in _parameters if not p.frozen]

        def wrapped_evaluate(*args, **kwargs):
            # Controlla se kwargs contiene parametri congelati
            for key in kwargs:
                if key in frozen_param_names:
                    raise ValueError(
                        f"Il parametro '{key}' è congelato e non può essere fornito nelle kwargs."
                    )

            # Mappa args e kwargs ai parametri non congelati
            if len(args) > len(free_param_names):
                raise ValueError(
                    "Il numero di args forniti è maggiore del numero di parametri liberi."
                )

            free_params = dict(zip(free_param_names, args))
            free_params.update(kwargs)

            # Unisci con i parametri congelati
            all_params = {**freeze_params, **free_params}

            # Prepara gli argomenti posizionali per la funzione evaluate
            positional_args = [
                all_params[name] for name in _parameters.parameters_names
            ]

            return func(*positional_args)

        return wrapped_evaluate

    def _validate_parameters(
        self,
        parameters,
        param_dict: ParameterHandler,
        class_param_dict: ParameterHandler,
    ):
        """
        Valida e aggiorna i parametri forniti.

        Parameters:
        -----------
        parameters : list
            Lista di parametri.
        param_dict : dict
            Dizionario dei parametri dell'istanza.
        class_param_dict : dict
            Dizionario dei parametri della classe.

        Raises:
        -------
        TypeError
            Se un parametro non è un'istanza di Parameter.
        ValueError
            Se un parametro non può essere aggiunto durante la creazione della classe.
        """
        if not isinstance(param_dict, ParameterHandler) and not isinstance(
            class_param_dict, ParameterHandler
        ):
            raise TypeError("Questo errore non dovrebbe accadere")
        for param in parameters:
            if not isinstance(param, Parameter):
                raise TypeError(
                    "Initial Parameters for the model must be instances of Parameter"
                )
            if param.name in class_param_dict:
                param_dict[param.name].name = param.name
                param_dict[param.name].value = param.value
                param_dict[param.name].frozen = param.frozen
                param_dict[param.name].bounds = param.bounds
                # param_dict[param.name].description = param.description
                # param_dict[param.name] = Parameter(
                #    param.name,
                #    param.value,
                #    param.frozen,
                #    param.bounds,
                #    param.description,
                # param.share,
                # )
            else:
                raise ValueError(
                    f"Parameter {param.name} cannot be added during class creation as it is not included in the evaluation method!"
                )


# ---------------------------------------------------------------------------------------------------------
#
#
#
#
# ----------------------------------------------------------------------------------------------------------

class FittableModel(metaclass=ModelMeta):
    """
    Classe base per modelli adattabili, utilizzando ModelMeta come metaclasse.

    Attributes:
        parameters_names (List[str]): Nomi dei parametri del modello.
        n_parameters (int): Numero totale di parametri del modello.
        parameters_values (List[float]): Valori dei parametri del modello.
        parameters_bounds (List[Tuple[float, float]]): Limiti dei parametri del modello.
        free_parameters (List[Parameter]): Parametri non congelati del modello.
        frozen_parameters (List[Parameter]): Parametri congelati del modello.
        n_free_parameters (int): Numero di parametri non congelati del modello.
        _binary_freeze_map (List[bool]): Mappa binaria dei parametri congelati.
        _binary_melt_map (List[bool]): Mappa binaria dei parametri non congelati.
        parameters (dict): Dizionario dei parametri del modello.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Inizializza la sottoclasse dopo l'inizializzazione da parte di ModelMeta.

        Args:
            **kwargs: Argomenti passati alla sottoclasse.
        """
        super().__init_subclass__(**kwargs)

    def __init__(self):
        """
        Inizializza la classe base FittableModel.
        """
        super().__init__()
        
        # override della call evaluate nel istanza per poterla chimare con valori di default
        self.__evaluate__ = self.evaluate
        self.evaluate = self.base_evaluate
        
    def __post_init__(self):
        """
        Metodo eseguito dopo l'inizializzazione per ulteriori configurazioni.
        """
        pass

    def _invalidate_cache(self):
        """
        Invalida la cache del modello e dei parametri.
        """
        if hasattr(self, "_cache"):
            del self._cache
        self._parameters._invalidate_cache()

    @property
    def parameters_names(self) -> List[str]:
        """
        Ritorna i nomi dei parametri del modello.

        Returns:
            List[str]: Lista dei nomi dei parametri.
        """
        return self._parameters.parameters_names

    @property
    def n_parameters(self) -> int:
        """
        Ritorna il numero totale di parametri del modello.

        Returns:
            int: Numero totale di parametri.
        """
        return len(self._parameters)

    @property
    def parameters_values(self) -> List[float]:
        """
        Ritorna i valori dei parametri del modello.

        Returns:
            List[float]: Lista dei valori dei parametri.
        """
        return self._parameters.parameters_values

    @property
    def parameters_bounds(self) -> List[Tuple[float, float]]:
        """
        Ritorna i limiti dei parametri del modello.

        Returns:
            List[Tuple[float, float]]: Lista dei limiti dei parametri.
        """
        return self._parameters.parameters_bounds

    @property
    def free_parameters(self) -> List[Parameter]:
        """
        Ritorna i parametri non congelati del modello.

        Returns:
            List[Parameter]: Lista dei parametri non congelati.
        """
        return self._parameters.free_parameters

    @property
    def frozen_parameters(self) -> List[Parameter]:
        """
        Ritorna i parametri congelati del modello.

        Returns:
            List[Parameter]: Lista dei parametri congelati.
        """
        return self._parameters.frozen_parameters

    @property
    def n_free_parameters(self) -> int:
        """
        Ritorna il numero di parametri non congelati del modello.

        Returns:
            int: Numero di parametri non congelati.
        """
        return self._parameters.n_free_params

    @property
    def _binary_freeze_map(self) -> List[bool]:
        """
        Ritorna una mappa binaria dei parametri congelati.

        Returns:
            List[bool]: Mappa binaria dei parametri congelati.
        """
        return [p.frozen for p in self]

    @property
    def _binary_melt_map(self) -> List[bool]:
        """
        Ritorna una mappa binaria dei parametri non congelati.

        Returns:
            List[bool]: Mappa binaria dei parametri non congelati.
        """
        return [not p.frozen for p in self]

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """
        Ritorna il dizionario dei parametri del modello.

        Returns:
            dict: Dizionario dei parametri del modello.
        """
        return self._parameters._parameters

    def set_parameters_values(self, args=None, **kwargs) -> None:
        """
        Imposta i valori dei parametri utilizzando argomenti posizionali o parole chiave.

        Args:
            args (list, opzionale): Una lista di valori per i parametri.
            kwargs (dict, opzionale): Un dizionario con nomi di parametri come chiavi e valori corrispondenti.

        Raises:
            ValueError: Se vengono forniti sia args che kwargs.

        Esempio:
            >>> obj.set_parameters_values([1, 2, 3])
            >>> obj.set_parameters_values(param1=1, param2=2)
        """
        if args and not kwargs:
            self._parameters.set_values(args)
        elif kwargs and not args:
            self._parameters.set_values(kwargs)
        else:
            raise ValueError("Cannot give both args and kwargs!")

    def set_parameters_bounds(self, args=None, **kwargs) -> None:
        """
        Imposta i limiti dei parametri utilizzando argomenti posizionali o parole chiave.

        Args:
            args (list, opzionale): Una lista di limiti per i parametri.
            kwargs (dict, opzionale): Un dizionario con nomi di parametri come chiavi e limiti corrispondenti.

        Raises:
            ValueError: Se vengono forniti sia args che kwargs.

        Esempio:
            >>> obj.set_parameter_bounds([0, 10])
            >>> obj.set_parameter_bounds(param1=(0, 10), param2=(0, 5))
        """
        if args and not kwargs:
            self._parameters.set_bounds(args)
        elif kwargs and not args:
            self._parameters.set_bounds(kwargs)
        else:
            raise ValueError("Cannot give both args and kwargs!")

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
            vals = self.parameters_names
        else:
            vals = args
        for element in vals:
            name = element
            if isinstance(element, int):
                name = self._parameters._map_indices_to_names(element)
            self[name].frozen = state

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

    @staticmethod
    def check_keys_in_dict(keys, dictionary) -> bool:
        """
        Controlla che tutte le chiavi siano presenti nel dizionario.

        Args:
            keys (list): Una lista di chiavi da controllare.
            dictionary (dict): Il dizionario in cui controllare le chiavi.

        Returns:
            bool: True se tutte le chiavi sono presenti, False altrimenti.

        Esempio:
            >>> obj.check_keys_in_dict(['key1', 'key2'], {'key1': 1, 'key2': 2})
        """
        return all(key in dictionary for key in keys)

    @staticmethod
    def extract_values_for_keys(keys, dictionary) -> dict:
        """
        Estrae un sotto-dizionario dato un elenco di chiavi.

        Args:
            keys (list): Una lista di chiavi da estrarre.
            dictionary (dict): Il dizionario da cui estrarre le chiavi.

        Returns:
            dict: Un sotto-dizionario con le chiavi specificate.

        Raises:
            KeyError: Se una o più chiavi non sono presenti nel dizionario.

        Esempio:
            >>> obj.extract_values_for_keys(['key1', 'key2'], {'key1': 1, 'key2': 2, 'key3': 3})
        """
        missing_keys = [key for key in keys if key not in dictionary]
        if missing_keys:
            raise KeyError(f"Le seguenti chiavi mancano nel dizionario: {missing_keys}")
        return {key: dictionary[key] for key in keys}

    def map_kwarg_to_args(self, kwargs):
        """
        Mappa le kwargs alle corrispondenti args.

        Args:
            kwargs (dict): Un dizionario con nomi di parametri come chiavi e valori corrispondenti.

        Returns:
            list: Una lista di valori mappati.

        Esempio:
            >>> obj.map_kwarg_to_args({'param1': 1, 'param2': 2})
        """
        return [
            kwargs.get(name, val)
            for name, val in zip(self.parameters_names, self.parameters_values)
        ]

    def get_grid_from_args_or_kwargs(self, args, kwargs):
        """
        Ottiene la griglia dagli args o kwargs.

        Args:
            args (tuple): Argomenti posizionali.
            kwargs (dict): Parole chiave.

        Returns:
            list: Una lista di valori di griglia.

        Esempio:
            >>> obj.get_grid_from_args_or_kwargs([10, 20], {'grid1': 10, 'grid2': 20})
        """
        if args:
            return args[: self.n_inputs]
        return list(self.extract_values_for_keys(self._grid_variables, kwargs).values())

    def get_vals_from_args_or_kwargs(self, args, kwargs):
        """
        Ottiene i valori dagli args o kwargs.

        Args:
            args (tuple): Argomenti posizionali.
            kwargs (dict): Parole chiave.

        Returns:
            list: Una lista di valori.

        Esempio:
            >>> obj.get_vals_from_args_or_kwargs([30, 40], {'param1': 30, 'param2': 40})
        """
        if not args:  # or len(args) == self.n_inputs:
            return self.map_kwarg_to_args(kwargs)
        return args[self.n_inputs :]

    @staticmethod
    def map_args_to_values(a, maschera, b):
        """
        Mappa i valori di `a` e `b` secondo una maschera fornita.

        Questa funzione sostituisce gli elementi in `a` con gli elementi in `b` secondo una maschera.
        Se un elemento nella maschera è 0, l'elemento corrispondente in `a` viene sostituito con un elemento da `b`.
        Se è 1, l'elemento in `a` viene mantenuto.

        Args:
            a (list): La lista originale di elementi.
            maschera (list): La lista di maschera contenente 0 e 1 per indicare quali elementi sostituire.
            b (list): La lista di elementi da usare per la sostituzione.

        Returns:
            list: Una nuova lista con gli elementi sostituiti secondo la maschera.

        Raises:
            ValueError: Se la lunghezza di `maschera` non corrisponde alla lunghezza di `a` o `b` non contiene abbastanza elementi per la sostituzione.

        Esempio:
            >>> a = [1, 2, 3, 4, 5, 6]
            >>> maschera = [1, 1, 1, 0, 0, 0]
            >>> b = [9, 9, 9]
            >>> map_args_to_values(a, maschera, b)
            [1, 2, 3, 9, 9, 9]
        """
        iter_b = iter(b)
        return [next(iter_b) if not maschera[i] else a[i] for i in range(len(a))]

    @staticmethod
    def map_kwargs_to_values(a, maschera, b) -> dict:
        """
        Mappa i valori di `a` e `b` secondo una maschera fornita.

        Questa funzione sostituisce i valori nel dizionario `a` con i valori nel dizionario `b` secondo una maschera.
        Se un valore nella maschera è 0, il valore corrispondente in `a` viene sostituito con un valore da `b`.
        Se è 1, il valore in `a` viene mantenuto.

        Args:
            a (dict): Il dizionario originale di elementi.
            maschera (list): La lista di maschera contenente 0 e 1 per indicare quali valori sostituire.
            b (dict): Il dizionario di elementi da usare per la sostituzione.

        Returns:
            dict: Un nuovo dizionario con i valori sostituiti secondo la maschera.

        Raises:
            ValueError: Se la lunghezza di `maschera` non corrisponde alla lunghezza di `a` o `b` non contiene abbastanza valori per la sostituzione.

        Esempio:
            >>> a = {'x': 1, 'y': 2, 'z': 3, 'w': 4, 'v': 5, 'u': 6}
            >>> maschera = [1, 1, 1, 0, 0, 0]
            >>> b = {'x': 9, 'y': 9, 'z': 9}
            >>> map_kwargs_to_values(a, maschera, b)
            {'x': 1, 'y': 2, 'z': 3, 'w': 9, 'v': 9, 'u': 9}
        """
        if len(maschera) != len(a):
            raise ValueError(
                "La lunghezza della maschera deve corrispondere alla lunghezza del dizionario 'a'"
            )
        result = {name: param.value for name,param in a.items()}
        
        for key,value in b.items():
            if key not in a:
                raise ValueError(f'{key} is not a parameter')
            
            if not a[key].frozen:
                result[key] = value
        
        return result
        
        #iter_b = iter(b.values())
        
        #return {
        #    k: next(iter_b) if not mask else v.value
        #    for (k, v), mask in zip(a.items(), maschera)
        #}

    def __call__(self, grid: list, params: list|dict = None):
        """
        Calcola i valori dei parametri basati sulla griglia e sui parametri forniti.

        Args:
            grid (list): La lista di valori di griglia.
            params (list or dict, optional): I parametri da usare per il calcolo. Può essere una lista o un dizionario.

        Returns:
            Risultato della funzione `evaluate` basato sulla griglia e sui valori dei parametri calcolati.

        Raises:
            ValueError: Se il numero di parametri non corrisponde al numero di parametri liberi.
        """
        
        
        if params is None:  # Caso 1: nessun parametro fornito
            
            params = [p.value for p in self.free_parameters]
            #print((isinstance(params, Iterable)) and (not isinstance(params, dict)))
            
        
        if isinstance(params, dict):
            
            for name in params:
                if name not in self:
                    raise ValueError(f'Param {name} is not a parameter')
                elif self[name].frozen:
                    raise ValueError(f'Param {name} is Frozen!')
                
            vals = list(
                self.map_kwargs_to_values(
                    self.parameters, self._binary_freeze_map, params
                ).values()
            )
            
        elif isinstance(params, Iterable) and (not isinstance(params, dict)):
            if (
                len(params) != self.n_free_parameters
            ):  # Caso 2: numero sbagliato di parametri
                raise ValueError(
                    f"Number of params {len(params)} does not match number of free parameters {(self.n_free_parameters)}"
                )
            vals = self.map_args_to_values(
                self.parameters_values, self._binary_freeze_map, params
            )
        else:
            
            raise ValueError("Invalid type for params! Must be list or dict.")

        return self.evaluate(*grid, *vals)

    def __getitem__(self, name: str) -> Parameter:
        return self._parameters.__getitem__(name)
    
    def __setitem__(self, key, value:Parameter) -> None:
        return self._parameters.__setitem__(key, value)

    def __contains__(self, key: str) -> bool:
        return self._parameters.__contains__(key)

    def __iter__(self):
        return self._parameters.__iter__()

    def __len__(self) -> int:
        return self._parameters.__len__()

    def __str__(self):
        """
        Restituisce una rappresentazione testuale del modello.

        Returns:
            str: Una stringa che rappresenta il modello.
        """
        total_string = f"MODEL NAME: {self.name} \n"
        total_string += f"FREE PARAMS: {self._parameters.n_free_params}\n"
        total_string += f"GRID VARIABLES: {self._grid_variables}\n"
        total_string += "_" * 70 + "\n"
        total_string += (
            f"{'':<4} {'NOME':<15} {'VALORE':<10} {'FREEZE':<10} {'BOUNDS':<20}\n"
        )
        total_string += "_" * 70 + "\n"
        for i, param in enumerate(self._parameters):
            value_str = f"{param.value:.2f}"
            bounds_str = f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"
            # share_str = share_str = ", ".join(
            #    [param.name for param in param.share if param is not self]
            # )
            total_string += f"{i:<4} {param.name:<15} {value_str:<10} {param.frozen:<10} {bounds_str:<20}\n"
        return total_string

    def copy(self):
        return deepcopy(self)
    
    def base_evaluate(self, *args, **kwargs):
        # poco elegante, da migliorare
        grid = args[: self.n_inputs]

        if len(args) == self.n_inputs:  # se non ho dato args al di fuori dalla griglia
            vals = self.map_kwarg_to_args(kwargs)
        else:
            vals = args[self.n_inputs :]
        return self.__evaluate__(*grid, *vals)

    __add__ = componemodels("+")
    __mul__ = componemodels("*")
    __or__ = componemodels("|")
    __truediv__ = componemodels("/")
    __sub__ = componemodels("-")


# ---------------------------------------------------------------------------------------------------------
#
#
#
#
# ----------------------------------------------------------------------------------------------------------

class CompositeModel(FittableModel):
    LINEAR_OPERATIONS = ["+", "-", "*", "/", "**"]
    COMPOSITE_OPERATION = "|"

    IS_COMPOSITE = True

    def __init_subclass__(self, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self, left: FittableModel, right: FittableModel, op, **kwargs) -> None:
        """
        Inizializza una nuova istanza di CompositeModel.

        ATTENZIONE: Composite Model crea una copia di left e right in modo da agire
        su un tree condiviso dai sottomodelli.

        Args:
            left (FittableModel): Il modello di sinistra.
            right (FittableModel): Il modello di destra.
            op (str): Operatore per combinare i modelli.
            kwargs: Altri argomenti passati al costruttore del super.
        """
        super().__init__(**kwargs)

        self.op_str = op  # stringa dell'operatore
        self._op = self.map_operator(op)

        self._left = left.copy()
        self._right = right.copy()

        self._update_model_properties()

    @property
    def left(self):
        """
        Restituisce il modello di sinistra.

        Returns:
            FittableModel: Il modello di sinistra.
        """
        return self._left

    @left.setter
    def left(self, new_left):
        """
        Imposta un nuovo modello di sinistra e aggiorna le proprietà correlate.

        Args:
            new_left (FittableModel): Il nuovo modello di sinistra.

        Raises:
            TypeError: Se new_left non è un'istanza di FittableModel.
        """
        if not isinstance(new_left, FittableModel):
            raise TypeError("New left must be instance of FittableModel")
        self._left = new_left.copy()
        self._update_model_properties()

    @property
    def right(self):
        """
        Restituisce il modello di destra.

        Returns:
            FittableModel: Il modello di destra.
        """
        return self._right

    @right.setter
    def right(self, new_right):
        """
        Imposta un nuovo modello di destra e aggiorna le proprietà correlate.

        Args:
            new_right (FittableModel): Il nuovo modello di destra.

        Raises:
            TypeError: Se new_right non è un'istanza di FittableModel.
        """
        if not isinstance(new_right, FittableModel):
            raise TypeError("New right must be instance of FittableModel")
        self._right = new_right.copy()
        self._update_model_properties()

    def _update_model_properties(self):
        """
        Aggiorna le proprietà del modello in base ai modelli di sinistra e destra attuali.
        """
        self._update_n_dim()
        self.sub_models = self._collect_submodels()
        self._parameters, self.kwarg_map = self._combine_parameters()
        self._invalidate_cache()
        self._grid_variables = list(self.sub_models.values())[
            0
        ]._grid_variables  # da aggiustare

    def _invalidate_cache(self):
        """
        Invalida la cache del modello e dei sottomodelli.
        """
        super()._invalidate_cache()
        self.left._invalidate_cache()
        self.right._invalidate_cache()

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
            if (self.left.n_inputs != self.right.n_inputs) or (
                self.left.n_outputs != self.right.n_outputs
            ):
                raise ValueError("Number of inputs/output do not match!")

            if self.left.n_dim != self.right.n_dim:
                raise ValueError("Number of dimensions do not match!")

            self.n_dim = self.left.n_dim
            self.n_inputs = self.left.n_inputs
            self.n_outputs = self.left.n_outputs

        elif self.op_str == self.COMPOSITE_OPERATION:
            if self.left.n_outputs != self.right.n_inputs:
                raise ValueError(
                    "Number of output for left must be = n_inputs of right!"
                )
            if self.left.n_dim != self.right.n_dim:
                raise ValueError("Number of dimensions do not match!")

            self.n_dim = self.left.n_dim
            self.n_inputs = self.left.n_inputs
            self.n_outputs = self.right.n_outputs
            
    @staticmethod
    def check_last_three_chars(s: str) -> Tuple[bool, str]:
        """
        Verifica se gli ultimi tre caratteri di una stringa sono nel formato '_{i}'
        dove i è un intero compreso tra 0 e 99.

        Args:
            s (str): La stringa da verificare.

        Returns:
            tuple:
                bool: True se la stringa termina con '_{i}' dove i è compreso tra 0 e 99, altrimenti False.
                str: Il nome senza il suffisso '_{i}' se presente, altrimenti la stringa originale.
        """
        pattern = r"_\d{1,2}$"
        match = re.search(pattern, s)
        if match:
            num = int(match.group()[1:])  # Estrarre il numero dal match (ignorare '_')
            if 0 <= num <= 99:
                return True, s[: match.start()]
        return False, s

    def _combine_parameters(self) -> Tuple[ParameterHandler, dict]:
        """
        Rimappa i nomi dei parametri in accordo con l'id dei sottomodelli.

        Returns:
            tuple: Handler dei parametri combinati e mappa delle kwargs.

        """
        params = ParameterHandler()
        kwarg_map = {}
        for i, (mod_id, mod) in enumerate(self.sub_models.items()): #itero sui sottomodelli
            for param in mod:
                #name = param.name
                _, name = self.check_last_three_chars(param.name)   #rimuovo l'id e sostituisco con quello nuovo
                                
                #if f"_{i}" not in name[-3:]:
                param.name = name + f"_{i}"
                params._add_parameter(param)

                kwarg_map[f"{param.name}"] = (mod_id, name)

        return params, kwarg_map

    def _collect_submodels(self):
        """
        Raccoglie le foglie del tree in modo ricorsivo per vedere i sottomodelli
        che compongono il modello composito.

        Returns:
            dict: Dizionario dei sottomodelli.

        """
        submodels = {}
        counter = 0

        def helper(m, prefix):
            nonlocal counter
            if isinstance(m, CompositeModel):
                if hasattr(m, "left") and m.left is not None:
                    helper(m.left, f"{prefix}")
                if hasattr(m, "right") and m.right is not None:
                    helper(m.right, f"{prefix}")
            else:
                model_id = f"{prefix}_{counter}"
                submodels[model_id] = m
                counter += 1

        helper(self, "model")
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

    def __str__(self):
        """
        Restituisce una stringa che rappresenta il modello composito e i suoi parametri.

        Returns:
            str: Una stringa che rappresenta il modello composito, i modelli contenuti e i parametri liberi.
        """
        total_string = f"COMPOSITE MODEL NAME: {self.name} \n"
        total_string += (
            f"CONTAINED MODELS: {[model.name for model in self.sub_models.values()]}\n"
        )
        total_string += (
            f"GRID VARIABLES: {list(self.sub_models.values())[0]._grid_variables}\n"
        )
        total_string += f"LOGIC: {self.composite_structure()}\n"
        total_string += f"FREE PARAMS: {self.n_free_parameters}\n"
        total_string += "_" * 70 + "\n"
        total_string += (
            f"{'':<4} {'NOME':<15} {'VALORE':<10} {'FREEZE':<10} {'BOUNDS':<20} \n"
        )
        total_string += "_" * 70 + "\n"
        for i, (param_name, param) in enumerate(self.parameters.items()):
            value_str = f"{param.value:.2f}"
            bounds_str = f"({param.bounds[0]:.2f}, {param.bounds[1]:.2f})"

            total_string += f"{i:<4} {param_name:<15} {value_str:<10} {param.frozen:<10} {bounds_str:<20}\n"
        return total_string

    def map_args(self, val):
        """
        Mappa gli argomenti ai valori dei parametri liberi

        Args:
            val (list): Lista dei valori dei parametri.

        Returns:
            tuple: Valori mappati per i modelli left e right.
        """
        if val is None or len(val) == 0:
            val = self.parameters_values

        len_left_params = len(self.left)

        val_left = compress(val[:len_left_params], self.left._binary_melt_map)
        val_right = compress(val[len_left_params:], self.right._binary_melt_map)
        return list(val_left), list(val_right)

    def map_args_full(self, val):
        """
        Mappa gli argomenti ai valori dei parametri.

        Args:
            val (list): Lista dei valori dei parametri.

        Returns:
            tuple: Valori mappati per i modelli left e right.
        """
        if val is None or len(val) == 0:
            val = self.parameters_values

        len_left_params = len(self.left.parameters_values)

        val_left = val[:len_left_params]
        val_right = val[len_left_params:]

        return val_left, val_right

    def __call__(self, grid, params: List[float] = None):
        """
        Esegue il modello composito con i parametri forniti.
        forzando il numero di dimensioni ad essere uguale al
        numero di parametri liberi

        Args:
            grid (list): Lista dei valori della griglia.
            params (list, opzionale): Lista dei valori dei parametri.

        Returns:
            list: Risultati del modello composito.

        Raises:
            ValueError: Se il numero di parametri non corrisponde al numero di parametri liberi.
        """
        if params is None:  # Caso 1: nessun parametro fornito
            params = [p.value for p in self.free_parameters]
            # print((isinstance(params, Iterable)) and (not isinstance(params, dict)))

        if isinstance(params, dict):
            for name in params:
                if name not in self:
                    raise ValueError(f"Param {name} is not a parameter")
                elif self[name].frozen:
                    raise ValueError(f"Param {name} is Frozen!")

            vals = list(
                self.map_kwargs_to_values(
                    self.parameters, self._binary_freeze_map, params
                ).values()
            )

        elif isinstance(params, Iterable) and (not isinstance(params, dict)):
            if (
                len(params) != self.n_free_parameters
            ):  # Caso 2: numero sbagliato di parametri
                raise ValueError(
                    f"Number of params {len(params)} does not match number of free parameters {(self.n_free_parameters)}"
                )
            vals = self.map_args_to_values(
                self.parameters_values, self._binary_freeze_map, params
            )
        else:
            raise ValueError("Invalid type for params! Must be list or dict.")

        val_left, val_right = self.map_args(vals)
        #print(val_left,val_right)

        if self.op_str in self.LINEAR_OPERATIONS:
            left_result = self.left(grid, val_left)
            right_result = self.right(grid, val_right)
            return self._op(left_result, right_result)

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_result = self.left(grid, val_left)
            #right_result = self.right(left_result, val_right)

            return self.right(grid=[left_result],params=val_right)

        #return self._op(left_result, right_result)

    def evaluate(self, *args, **kwargs):
        """
        Esegue il modello composito con gli argomenti forniti.

        Args:
            args (tuple): Argomenti posizionali.
            kwargs (dict): Parole chiave.

        Returns:
            list: Risultati del modello composito.

        Raises:
            ValueError: Se vengono forniti sia args che kwargs.
        """
        grid = args[: self.n_inputs]  # --- primi n-elementi sono la griglia

        if not kwargs:  # se non do kwargs
            val = args[self.n_inputs :]

        if kwargs:  # if invece di elif per fare override
            val = self.map_kwarg_to_args(kwargs)

        val_left, val_right = self.map_args_full(val)

        if self.op_str in self.LINEAR_OPERATIONS:
            return self._op(
                self.left.evaluate(*grid, *val_left),
                self.right.evaluate(*grid, *val_right),
            )

        elif self.op_str == self.COMPOSITE_OPERATION:
            left_results = self.left.evaluate(*grid, *val_left)
            return self.right.evaluate(left_results, *val_right)
        