from typing import List, Literal, Tuple, Union, Dict, Iterator
import numpy as np
from collections import OrderedDict
import warnings
from typing import Iterable
from io import StringIO
from copy import deepcopy
from tabulate import tabulate



__all__ = ['Parameter',
           'ParameterHandler',
           'Constrain',
           'FunctionConstrain',
           'TieConstrain']

#if TYPE_CHECKING:
from priors import Prior, UniformPrior

'''
Constrain Idea: 
Tied = constrain che obbliga un parametro ad assumere un valore sulla base di un altro
[riduce i gradi di libertà]
Functional = constrain che modifica il valore di un parametro secondo una funzione
[non riduce i gradi di libertà]

Come gestire i constrain?
se un parametro è tied non è free ma deve comparire nella function call?
risposta: credo di no LOL

TODO: [] fare i dovuti check che i parametri free e frozen risolvano correttamente i constrains
'''

class Parameter:
    """
    Classe che rappresenta un singolo parametro di un modello.

    Attributes:
        name (str): Nome del parametro.
        value (float): Valore del parametro.
        frozen (bool): Stato di congelamento del parametro.
        bounds (Tuple[float, float]): Limiti del parametro.
        description (str): Descrizione del parametro.
    
    TODO: supporto a call esterne e alle unità (che diventa la nuova description),
        supporto a parametri con valori diversi da float?
        VectorParameter(Parameter)
        DictParameter(Parameter)
        FloatParameter = VectorParameter(dim=1) ?
        Implementazione dei Prior
        Supporto ai constrain
        
        questo può essere fatto usando @singledispatch di functools
        
    TODO: [ok] modificare self.name una volta dentro al handler modifica parameters_keys
    
    TODO: supporto a constrain multipli
    regole per i constrain: sono trasformazioni che prendono un valore e lo modificano.
    function(value) return new_value
    
    NOTE: Bounds legacy rimangono per linear opt. con scipy siccome i prior non sono supportati
    """

    def __init__(
        self,
        name: str,
        value: float,
        frozen: bool = False,
        bounds: Tuple[float, float] = (-float("inf"), float("inf")),
        description: str = "",
        prior:Prior = None,
        #handler: 'ParameterHandler' = None
        #constrain:'Constrain' = None, # to be updated for fitting purposes
    ) -> None:
        """
        Inizializza un nuovo parametro.

        Args:
            name (str): Nome del parametro.
            value (float): Valore del parametro.
            frozen (bool, opzionale): Stato di congelamento del parametro. Default è False.
            bounds (Tuple[float, float], opzionale): Limiti del parametro. Default è (-inf, inf).
            description (str, opzionale): Descrizione del parametro. Default è "".

        Raises:
            TypeError: Se i tipi degli argomenti non sono corretti.
            ValueError: Se i valori degli argomenti non sono validi.
        """
        ParameterValidator.validate_name(name)
        ParameterValidator.validate_bounds(bounds)
        ParameterValidator.validate_value_in_bounds(value, bounds)

        self._name = name
        self._value = value
        self._frozen = frozen
        self._bounds = bounds
        self._description = description
        self._handler = []
        self._chached_properties = ['value','bounds','frozen']
        
        # legacy, devono essere modificate
        self._constrain = False
        self._is_tied = False
        
        if prior is None:
            self._prior = UniformPrior(-float('inf'), float('inf'))
        else:
            self._prior = prior
    
    def _update_handler_cache(self) -> None:
        if self._handler:
            for handler in self._handler:
                handler._update_cache()
    
    @property
    def is_free(self) -> bool:
        '''helper function to see if a param is frozen or constrained'''
        return not (self.frozen or self.is_tied)
        
    @property
    def is_tied(self) -> bool:
        #if self.constrain is not None and self.constrain.reduce_varys:
        #    return True
        return self._is_tied
    
    
    
    
    @property
    def has_constrain(self) -> bool:
        return self._constrain
    
    @has_constrain.setter
    def has_constrains(self, value):
        self._constrain = value
        self._update_handler_cache()
    
    @property
    def handler(self) -> List['ParameterHandler']:
        return self._handler
    
    @property
    def prior(self) -> Prior:
        return self._prior
    
    @prior.setter
    def prior(self, value) -> None:
        ParameterValidator.validate_prior(value)
        self._prior = value
        
    
    @property
    def name(self) -> str:
        """
        Ritorna il nome del parametro.

        Returns:
            str: Nome del parametro.
        """
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """
        Imposta un nuovo nome per il parametro.

        Args:
            new_name (str): Nuovo nome del parametro.

        Raises:
            TypeError: Se il nuovo nome non è una stringa.
        """
        ParameterValidator.validate_name(new_name)
        self._name = new_name
        self._update_handler_cache()

    @property
    def value(self) -> float:
        """
        Ritorna il valore del parametro.

        Returns:
            float: Valore del parametro.
        """
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        """
        Imposta un nuovo valore per il parametro.

        Args:
            new_value (float): Nuovo valore del parametro.

        Raises:
            ValueError: Se il parametro è congelato o il nuovo valore è fuori dai limiti.
        """
        if self.frozen is True:
            warnings.warn(
                f"Parameter {self.name} is frozen, new value will be ignored!"
            )
            return
        ParameterValidator.validate_value_in_bounds(new_value, self._bounds)
        self._value = new_value
        self._update_handler_cache()
        

    @property
    def bounds(self) -> Tuple[float, float]:
        """
        Ritorna i limiti del parametro.

        Returns:
            Tuple[float, float]: Limiti del parametro.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds: Tuple[float, float]) -> None:
        """
        Imposta nuovi limiti per il parametro.

        Args:
            new_bounds (Tuple[float, float]): Nuovi limiti del parametro.

        Raises:
            TypeError: Se i nuovi limiti non sono una tupla di due elementi.
            ValueError: Se i nuovi limiti non sono validi.
        """
        if self.frozen is True:
            warnings.warn(
                f"Parameter {self.name} is frozen, new bounds will be ignored!"
            )
            return
        ParameterValidator.validate_bounds(new_bounds)
        ParameterValidator.validate_value_in_bounds(self._value, new_bounds)
        self._bounds = new_bounds
        self._update_handler_cache()

    @property
    def frozen(self) -> bool:
        """
        Ritorna lo stato di congelamento del parametro.

        Returns:
            bool: True se il parametro è congelato, False altrimenti.
        """
        return self._frozen

    @frozen.setter
    def frozen(self, is_true: bool) -> None:
        """
        Imposta lo stato di congelamento del parametro.

        Args:
            is_true (bool): Stato di congelamento del parametro.

        Raises:
            TypeError: Se il valore non è un booleano.
        """
        ParameterValidator.validate_frozen(is_true)
        self._frozen = is_true
        self._update_handler_cache()

    @property
    def description(self) -> str:
        """
        Ritorna la descrizione del parametro.

        Returns:
            str: Descrizione del parametro.
        """
        return self._description

    @description.setter
    def description(self, str: str) -> None:
        """
        Imposta una nuova descrizione per il parametro.

        Args:
            str (str): Nuova descrizione del parametro.

        Raises:
            TypeError: Se la descrizione non è una stringa.
        """
        ParameterValidator.validate_description(str)
        self._description = str

    def copy(self) -> "Parameter":
        """
        Ritorna una copia del parametro.

        Returns:
            Parameter: Copia del parametro.
        """
        return deepcopy(self)

    def __len__(self) -> int:
        """
        Ritorna la lunghezza del parametro (sempre 1).

        Returns:
            int: Lunghezza del parametro.
        """
        return 1

    def __iter__(self):
        """
        Ritorna un iteratore per il parametro.
        inutile(?)

        Returns:
            Iterator: Iteratore per il parametro.
        """
        return iter(
            [self._name, self._value, self._frozen, self._bounds, self._description]
        )

    def __getitem__(self, key):
        """
        Ritorna l'attributo specificato del parametro.

        Args:
            key (str): Nome dell'attributo.

        Returns:
            Any: Valore dell'attributo.
        """
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        """
        Imposta l'attributo specificato del parametro.

        Args:
            key (str): Nome dell'attributo.
            value (Any): Valore da assegnare all'attributo.
        """
        setattr(self, key, value)

    def __str__(self) -> str:
        """
        Ritorna una rappresentazione testuale del parametro.

        Returns:
            str: Rappresentazione testuale del parametro.
        """
        
        buffer = StringIO()

        buffer.write(f"PARAM NAME: {self.name}\n")

        # Definizione delle intestazioni
        field_names = ["NAME", "VALUE", "IS-FREE", "PRIOR", "DESCR:"]

        # Preparazione dei dati
        value_str = f"{self._value:.5g}"
        froz_str = "Yes" if self.is_free else "No"
        prior_str = self.prior._get_str()

        # Creazione della tabella con una lista di righe
        table_data = [[self.name, value_str, froz_str, prior_str, self.description]]

        # Creazione della tabella con tabulate
        table = tabulate(
            table_data,
            headers=field_names,
            tablefmt="plain",
            showindex="always",
            colalign=("left",),
            #floatfmt=".3f",
        )

        # Aggiungiamo la tabella al buffer
        buffer.write(table + "\n")

        return buffer.getvalue()
    
    def __call__(self, value):
        return self.prior(value)


class ParameterValidator:
    """
    Classe per la gestione della validazione di parametri singoli.
    """
    
    @staticmethod
    def validate_prior(prior: Prior) -> None:
        if not isinstance(prior, Prior):
            raise TypeError('Pior Must be istance of class Prior')

    @staticmethod
    def validate_name(name: str) -> None:
        """
        Valida il nome del parametro.

        Args:
            name (str): Nome del parametro.

        Raises:
            TypeError: Se il nome non è una stringa.
        """
        if not isinstance(name, str):
            raise TypeError("Il nome del parametro deve essere una stringa!")

    @staticmethod
    def validate_bounds(bounds: Tuple[float, float]) -> None:
        """
        Valida i limiti del parametro.

        Args:
            bounds (Tuple[float, float]): Limiti del parametro.

        Raises:
            TypeError: Se i limiti non sono una tupla di due elementi.
            ValueError: Se i limiti non sono validi.
        """
        if not isinstance(bounds, (list, np.ndarray, tuple)):
            raise TypeError(
                f"New bounds must be in form of iterable, you gave {type(bounds)}"
            )
        if len(bounds) != 2:
            raise ValueError("I limiti devono essere una tupla con due elementi.")
        if not bounds[0] <= bounds[1]:
            raise ValueError(
                "Il limite inferiore deve essere minore o uguale al limite superiore."
            )

    @staticmethod
    def validate_value_in_bounds(value: float, bounds: Tuple[float, float]) -> None:
        """
        Valida che il valore del parametro sia entro i limiti specificati.

        Args:
            value (float): Valore del parametro.
            bounds (Tuple[float, float]): Limiti del parametro.

        Raises:
            TypeError: Se il valore non è un numero.
            ValueError: Se il valore è fuori dai limiti.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be of type Number, not ", type(value))
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Il valore {value} è fuori dai limiti {bounds}")

    @staticmethod
    def validate_frozen(is_true: bool) -> None:
        """
        Valida lo stato di congelamento del parametro.

        Args:
            is_true (bool): Stato di congelamento del parametro.

        Raises:
            TypeError: Se il valore non è un booleano.
        """
        if not isinstance(is_true, bool):
            raise TypeError(
                f'Il valore di "frozen" può essere solo True o False, hai fornito {is_true, type(is_true)}'
            )

    @staticmethod
    def validate_description(strg: str) -> None:
        """
        Valida la descrizione del parametro.

        Args:
            strg (str): Descrizione del parametro.

        Raises:
            TypeError: Se la descrizione non è una stringa.
        """
        if not isinstance(strg, str):
            raise TypeError("Description must be a string!")
    
    @staticmethod
    def validate_constrain(value: 'Constrain') -> None:
        if not isinstance(value, Constrain):
            raise TypeError('Parameter constrain must be of type Constrain')



class ParameterHandler:
    """
    Classe che gestisce un insieme di parametri per un modello.

    Attributes:
        _parameters (OrderedDict[str, Parameter]): Dizionario dei parametri.
        _is_inside_model (bool): Indica se il gestore è stato aggiunto a un modello.
        _cache (Dict[str, Any]): Cache per le proprietà dei parametri.
    """

    def __init__(
        self, parameters: Union["Parameter", List["Parameter"]] = None
    ) -> None:
        """
        Inizializza il gestore dei parametri.

        Args:
            parameters (Union[Parameter, List[Parameter]], opzionale): Un singolo parametro o una lista di parametri.
        """
        self._parameters = OrderedDict()
        self._is_inside_model = (
            False  # Una volta dentro modello non posso aggiungere parametri
        )
        self._cache = {}
        
        self._cached_propreties = [
            "parameters_names",
            "parameters_values",
            "parameters_bounds",
            "parameters_keys",
            "parameters_values_dict",
            "binary_freeze_map",
            "binary_melt_map",
            "frozen_indeces",
            "not_frozen_indeces",
            "free_parameters",
            "not_free_parameters",
        ]  # list of attributes names that are cached
        
        self._cache_builders = {
            "parameters_names": self._build_parameters_names,
            "parameters_values": self._build_parameters_values,
            "parameters_bounds": self._build_parameters_bounds,
            "parameters_keys": self._build_parameters_keys,
            "parameters_values_dict": self._build_parameters_values_dict,
            "binary_freeze_map": self._build_binary_freeze_map,
            "binary_melt_map": self._build_binary_melt_map,
            "frozen_indeces": self._build_frozen_indeces,
            "not_frozen_indeces": self._build_not_frozen_indeces,
            "free_parameters": self._build_free_parameters,
            "not_free_parameters":self._build_not_free_parameters
        }
        
        assert len(self._cached_propreties) == len(self._cache_builders)
        #self._update_cache()
        

        if isinstance(parameters, Parameter):
            self.add_parameter(parameters)
        elif isinstance(parameters, Iterable):
            for param in parameters:
                self.add_parameter(param)
        elif parameters is None:
            pass
        else:
            raise TypeError(
                "Parameters must be of type Parameter or List[Parameter]",
                type(parameters),
            )

    def _invalidate_cache(self) -> None:
        """
        Invalida la cache dei parametri.
        """
        #del self._cache
        self._cache.clear()
    
    @property
    def cached_propreties(self):
        return self._cached_propreties

                
    # ......................
    #     UPDATE CACHE
    # ......................
    def _update_cache(self, key=None, value=None) -> None:
        """
        Aggiorna la cache dei parametri.
        Se key e value sono None, ricostruisce tutto.
        Altrimenti, aggiorna solo la chiave indicata.
        """
        if key is None and value is None:
            # Aggiornamento completo: usa i builder per tutte le chiavi
            for k in self._cached_propreties:
                builder = self._cache_builders.get(k)
                if builder is not None:
                    self._cache[k] = builder()
                else:
                    raise ValueError(f"Nessun builder definito per la chiave: {k}")
        else:
            # Aggiornamento parziale
            if key not in self._cached_propreties:
                raise ValueError(
                    f"La chiave '{key}' non è presente in cached_propreties."
                )

            # Se value non è None, vuol dire che l'utente specifica manualmente
            # il valore. Altrimenti, se value è None, lo ricostruiamo noi (builder).
            if value is not None:
                self._cache[key] = value
            else:
                builder = self._cache_builders.get(key)
                if builder is not None:
                    self._cache[key] = builder()
                else:
                    raise ValueError(f"Nessun builder definito per la chiave: {key}")
            
    # ......................
    #    METODI BUILDERS PER CACHE
    # ......................
    def _build_parameters_names(self) -> List[str]:
        return [p.name for p in self]

    def _build_parameters_values(self) -> List[float]:
        return [p.value for p in self]

    def _build_parameters_bounds(self) -> List[Tuple[float]]:
        return [p.bounds for p in self]

    def _build_parameters_keys(self):# -> list:
        return list(self._parameters.keys())

    def _build_parameters_values_dict(self) -> Dict[str, float]:
        return {
            key: val for key, val in zip(self.parameters_keys, self.parameters_values)
        }

    def _build_binary_freeze_map(self) -> List[bool]:
        return [p.frozen for p in self]

    def _build_binary_melt_map(self) -> List[bool]:
        return [not p.frozen for p in self]

    def _build_frozen_indeces(self) -> List[int]:
        return [
            i
            for i in range(len(self._binary_freeze_map))
            if self._binary_freeze_map[i] is True
        ]

    def _build_not_frozen_indeces(self) -> List[int]:
        return [
            i
            for i in range(len(self._binary_freeze_map))
            if self._binary_freeze_map[i] is False
        ]
    def _build_free_parameters(self) -> List[Parameter]:
        '''
        NOTE: this do not return only not-frozen params, but also unconstreined params
        '''
        return [p for p in self if p.is_free]
    
    def _build_not_free_parameters(self) -> List[Parameter]:
        return [p for p in self if p.is_free is False]

    @property
    def is_inside_model(self) -> bool:
        """
        Indica se il gestore è stato aggiunto a un modello.

        Returns:
            bool: True se è stato aggiunto a un modello, False altrimenti.
        """
        return self._is_inside_model

    def lock(self) -> None:
        """
        Blocca il gestore per prevenire l'aggiunta di nuovi parametri.
        """
        self._is_inside_model = True

    def unlock(self) -> None:
        """
        Sblocca il gestore per permettere l'aggiunta di nuovi parametri.
        """
        self._is_inside_model = False

    @property
    def parameters_values(self) -> List[float]:
        """
        Ritorna i valori dei parametri.

        Returns:
            List[float]: Lista dei valori dei parametri.
        """
        return self._cache.get("parameters_values", self._build_parameters_values())#[p.value for p in self])
        

    @property
    def parameters_names(self) -> List[str]:
        """
        Ritorna i nomi dei parametri.

        Returns:
            List[str]: Lista dei nomi dei parametri.
        """
        return self._cache.get("parameters_names", self._build_parameters_names())#[p.name for p in self])
       
    @property
    def parameters_keys(self) -> List[str]:
        """
        Ritorna i nomi dei parametri, dentro il dizionario.

        Returns:
            List[str]: Lista dei nomi dei parametri.
        """
        return self._cache.get("parameters_keys", self._build_parameters_keys())#[p for p in self._parameters.keys()])
        
    @property
    def parameters_values_dict(self):
        """
        Ritorna il dizionario key-value dei parametri, dove key
        non è necessariamente il nome del parametro
        __low level, meglio non giocarci.

        Returns:
            List[str]: Lista dei nomi dei parametri.
        """
        return self._cache.get("parameters_values_dict", self._build_parameters_values_dict())
            
    @property
    def parameters_bounds(self) -> List[Tuple[float, float]]:
        """
        Ritorna i limiti dei parametri.

        Returns:
            List[Tuple[float, float]]: Lista dei limiti dei parametri.
        """
        return self._cache.get("parameters_bounds", self._build_parameters_bounds())#[p.bounds for p in self])

    @property
    def n_free_params(self) -> int:
        """
        Ritorna il numero di parametri liberi.

        Returns:
            int: Numero di parametri liberi.
        """
        return len(self.free_parameters)

    @property
    def free_parameters(self) -> List["Parameter"]:
        """
        Ritorna solo i parametri liberi.

        Returns:
            List[Parameter]: Lista dei parametri liberi.
        """
        return self._cache.get("free_parameters", self._build_free_parameters())
        

    @property
    def not_free_parameters(self) -> List["Parameter"]:
        """
        Ritorna solo i parametri congelati.

        Returns:
            List[Parameter]: Lista dei parametri congelati.
        """
        return self._cache.get("not_free_parameters", self._build_not_free_parameters())
        
    # deprecated
    @property
    def _binary_freeze_map(self) -> List[bool]:
        # possibile da cachare
        return self._cache.get("binary_freeze_map", self._build_binary_freeze_map())
        
    # deprecated
    @property
    def _binary_melt_map(self) -> List[bool]:
        # possibile da cachare
        return self._cache.get("binary_melt_map", self._build_binary_melt_map())
    
    
    @property
    def not_frozen_indeces(self):
        return self._cache.get("not_frozen_indeces", self._build_not_frozen_indeces())
        

    @property
    def frozen_indeces(self):
        return self._cache.get("frozen_indeces", self._build_frozen_indeces())
       
       
    def _map_name_to_index(self, key: str) -> int:
        """
        Mappa il nome di un parametro al corrispondente indice.

        Args:
            key (str): Nome del parametro.

        Returns:
            int: Indice del parametro.

        Raises:
            KeyError: Se il nome del parametro non è trovato.
        """
        # try:   stop ask forgiveness, just go fast pls
        return self.parameters_keys.index(key)
        

    def _map_indices_to_names(self, index: int) -> str:
        """
        Mappa l'indice di un parametro al corrispondente nome.

        Args:
            index (int): Indice del parametro.

        Returns:
            str: Nome del parametro.

        Raises:
            IndexError: Se l'indice è fuori dai limiti.
        """

        if index < 0 or index >= len(self.parameters_keys):
            raise IndexError(f"Index '{index}' is out of bounds for the dictionary")
        return self.parameters_keys[index]

    def set_values(
        self, values: Union[List, Dict], include_frozen: bool = False
    ) -> None:
        """
        Imposta i valori dei parametri.

        Args:
            values (Union[List, Dict]): Valori da assegnare (lista o dizionario).
            include_frozen (bool, opzionale): Se includere i parametri congelati. Default è False.
        """
        
        self._assign_attribute(values, "value", include_frozen=include_frozen)
        self._update_cache()
    


    def set_bounds(
        self, bounds: Union[List, Dict], include_frozen: bool = False
    ) -> None:
        """
        Imposta i limiti dei parametri.

        Args:
            bounds (Union[List, Dict]): Limiti da assegnare (lista o dizionario).
            include_frozen (bool, opzionale): Se includere i parametri congelati. Default è False.
        """
        self._assign_attribute(bounds, "bounds", include_frozen=include_frozen)
        self._update_cache()
        #self._update_cache("parameters_bounds", [p.bounds for p in self])

    def set_frozen(
        self, is_frozen: Union[List, Dict], include_frozen: bool = False
    ) -> None:
        """
        Imposta lo stato di congelamento dei parametri.

        Args:
            is_frozen (Union[List, Dict]): Stato di congelamento da assegnare (lista o dizionario).
            include_frozen (bool, opzionale): Se includere i parametri già congelati. Default è False.
        """
        self._assign_attribute(is_frozen, "frozen", include_frozen=include_frozen)
        self._update_cache()
        # self._update_cache("parameters_values", [p.value for p in self])

    def _assign_attribute(
        self, items: Union[List, Dict], attribute: str, include_frozen: bool = False
    ) -> None:
        """
        Assegna un valore a un attributo dei parametri.

        Args:
            items (Union[List, Dict]): Valori da assegnare (lista o dizionario).
            attribute (str): Nome dell'attributo da assegnare.
            include_frozen (bool, opzionale): Se includere i parametri congelati. Default è False.

        Raises:
            ValueError: Se il numero di elementi non corrisponde al numero di parametri.
            TypeError: Se items non è né una lista né un dizionario.
        """
        if isinstance(items, (list, np.ndarray)):
            params = list(self) if include_frozen else self.free_parameters
            if len(items) != len(params):
                raise ValueError(
                    f"Number of items {len(items)} must match number of {'all' if include_frozen else 'free'} parameters ({len(params)})"
                )
            for param, val in zip(params, items):
                setattr(param, attribute, val)
        elif isinstance(items, dict):
            for name, val in items.items():
                param = self[name]
                if not include_frozen and param.frozen:
                    continue
                setattr(param, attribute, val)
        else:
            raise TypeError("Items must be a list or dictionary")

    def add_parameter(self, parameter: "Parameter", name=None) -> None:
        """
        Aggiunge un parametro al gestore.

        Args:
            parameter (Parameter): Il parametro da aggiungere.

        Raises:
            ValueError: Se il parametro esiste già o se si tenta di aggiungere un parametro dopo la creazione del modello.
        """
        
        #if self._is_inside_model:
        #    raise ValueError(f"Cannot add parameter {name} to model after it is locked")
        
        if not isinstance(parameter, Parameter):
            raise TypeError("Added parameter must be istance of Parameter class")
        
        if name is None:
            name = parameter.name
        
        if name not in self._parameters and self._is_inside_model:
            raise ValueError(
                f"Parameter {name} does not exists in function call. Write a new function call or build a composite model"
            )


        self._parameters[name] = parameter
        parameter._handler.append(self)
        # self.parameter_map[parameter.name] = parameter.name

        # self._invalidate_cache()

    def __getitem__(self, name: str) -> "Parameter":
        """
        Ritorna un parametro usando il suo nome.

        Args:
            name (str): Nome del parametro.

        Returns:
            Parameter: Il parametro richiesto.

        Raises:
            KeyError: Se il parametro non è trovato.
        """
        #try:
        return self._parameters[name]
        

    def __setitem__(self, key: str, value: "Parameter") -> None:
        """
        Imposta un parametro usando il suo nome.

        Args:
            key (str): Nome del parametro.
            value (Parameter): Il parametro da impostare.

        Raises:
            ValueError: Se si tenta di impostare un parametro dopo la creazione del modello.
            TypeError: Se value non è un'istanza di Parameter.
            ValueError: Se il parametro esiste già.
        """
        if key not in self and self._is_inside_model:
            raise ValueError(
                f"Parameter {key} does not exists in function call please Write a new function call or build a composite model"
            )
        self.add_parameter(value)
        self._update_cache()

    def __contains__(self, key: str) -> bool:
        """
        Verifica se un parametro è presente usando il suo nome.

        Args:
            key (str): Nome del parametro.

        Returns:
            bool: True se il parametro è presente, False altrimenti.
        """
        return key in self._parameters

    def __iter__(self) -> Iterator["Parameter"]:
        """
        Itera sui parametri.

        Returns:
            Iterator[Parameter]: Iteratore sui parametri.
        """
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """
        Ritorna il numero di parametri.

        Returns:
            int: Numero di parametri.
        """
        return len(self._parameters)
    
    def __str__(self) -> str:
        """
        Ritorna una rappresentazione in formato tabella dei parametri gestiti.

        Returns:
            str: Tabella che mostra nome, valore, bounds e stato frozen dei parametri.
        """
        
        # Definizione delle intestazioni
        field_names = ["NAME", "VALUE", "PRIOR", "IS-FREE", "DESCR"]

        # Preparazione dei dati
        table_data = []
        for i, param in enumerate(self):
            name = self.parameters_names[i]
            value = str(param.value) if param.value is not None else "None"
            #bounds = str(param.bounds) if param.bounds is not None else "None"
            prior_str = param.prior._get_str()
            frozen = "Yes" if param.is_free else "No"

            # Aggiungiamo la riga dei dati
            table_data.append([name, value, prior_str, frozen, param.description])

        # Creazione della tabella con tabulate
        table = tabulate(
            table_data,
            headers=field_names,
            tablefmt="plain",
            showindex="always",
            colalign=("left",),
        )

        # Ritorniamo la tabella
        return table

    

    def items(self):
        """
        Ritorna gli elementi del gestore come coppie chiave-valore.

        Returns:
            ItemsView: Vista degli elementi del gestore.
        """
        return self._parameters.items()

    def keys(self):
        """
        Ritorna le chiavi del dizionario base.

        Returns:
            KeysView: Vista delle chiavi del gestore.
        """
        return self._parameters.keys()

    def values(self):
        """
        Ritorna i parametri.

        Returns:
            ValuesView: Vista dei valori del gestore.
        """
        return self._parameters.values()


class Constrain:
    '''
    A constrain must implement 2 very important methods:
    1) __call__ to allow a callable to talk with a parameter
    2) reduce_varys to know if the affected parameter should be fitted or not
    eg: a parameter that is forced to share a value with another is no longer free 
    '''
    def __init__(self, name,*,reduce_varys=False):
        self.name = name
        self._reduce_varys = reduce_varys
    
    @property
    def reduce_varys(self) -> Literal[False]:
        """if this kind of constrains reduces the number of free params"""
        return self._reduce_varys
    
    def __call__(self, *args, **kwargs):
        '''logica della __call__ il primo args è il valore corrente del parametro'''
        raise NotImplementedError('Single Constrains should implemente their method')
    
class FunctionConstrain(Constrain):
    def __init__(self,func):
        if not callable(func):
            raise TypeError("Functional constrain must wrap a function (callable).")
    

        super().__init__(name = func.__name__, reduce_varys=False)
        self._func = func
        
    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        if not callable(value):
            raise TypeError("Functional constrain must wrap a funtion")

        self._func = value

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    
    
   
        
    
    
    
'''
NOTE: come funziona:
Tied legge args della funzione e li trasforma in key per il modello a cui è applicato
quali params sono tied? il primo di default
ora, dopo essere stato aggiunto al layer del modello, il primo args della funzione (la key del modello quindi)
viene freezata
'''
class TieConstrain(Constrain):
    def __init__(self, func, tied_param:str):
        # Controllo che `func` sia callable
        if not callable(func):
            raise TypeError("Functional constrain must wrap a function (callable).")

        super().__init__(name=func.__name__, reduce_varys=True)
        self._func = func
        self.tied_param = tied_param
            
        
    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        if not callable(value):
            raise TypeError("Functional constrain must wrap a funtion")
        
        self._func = value
                               
    def __call__(self, *args ,**kwargs):        
        return self.func(*args, **kwargs)
        
