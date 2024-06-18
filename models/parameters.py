from typing import List, Tuple, Union, Callable, Dict
import numpy as np

# import warnings
from functools import wraps
from copy import deepcopy

def cached_property(func):
    """Cacha la proprietà così da rendere il calcolo più veloce."""

    @property
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, "_cache"):
            self._cache = {}  # Inizializza il cache se non esiste
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self)  # Calcola e memorizza il risultato
        return self._cache[func.__name__]  # Ritorna il valore memorizzato

    return wrapper

class Constrain:
    def __init__(self, func: Callable, *args) -> None:
        pass


class Parameter:
    """
    Classe che rappresenta un singolo parametro di un modello.

    Attributes:
        name (str): Nome del parametro.
        value (float): Valore del parametro.
        frozen (bool): Stato di congelamento del parametro.
        bounds (Tuple[float, float]): Limiti del parametro.
        description (str): Descrizione del parametro.
    """

    def __init__(
        self,
        name: str,
        value: float,
        frozen: bool = False,
        bounds: Tuple[float, float] = (-float('inf'), float('inf')),
        description: str = "",
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
        if self._frozen:
            raise ValueError(
                f"Parametro {self._name} è congelato! Il nuovo valore non può essere impostato."
            )
        ParameterValidator.validate_value_in_bounds(new_value, self._bounds)
        self._value = new_value

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
        ParameterValidator.validate_bounds(new_bounds)
        ParameterValidator.validate_value_in_bounds(self._value, new_bounds)
        self._bounds = new_bounds

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
        total_string = f"PARAM NAME: {self.name}\n"
        total_string += "_" * 100 + "\n"
        total_string += f"{'NOME':<15} {'VALORE':<10} {'FREEZE':<10} {'BOUNDS':<20} {'DESCRIZIONE':<20} \n"
        total_string += "_" * 100 + "\n"

        value_str = f"{self._value:.5g}"
        bounds_str = f"({self._bounds[0]:.5g}, {self._bounds[1]:.5g})"
        total_string += f"{self.name:<15} {value_str:<10} {self._frozen:<10} {bounds_str:<20} {self.description:<20} \n"
        return total_string


class ParameterValidator:
    """
    Classe per la gestione della validazione di parametri singoli.
    """

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


class Constant(Parameter):
    def __init__(
        self,
        name: str,
        value: float,
        frozen: bool = True,
        bounds: Tuple[float, float] = (-float('inf'), float('inf')),
        description: str = "",
    ) -> None:
        super().__init__(name, value, frozen, bounds, description)

    @Parameter.value.setter
    def value(self, new_value: float) -> None:
        raise AttributeError("Cannot modify the value of a Constant.")

    @Parameter.bounds.setter
    def bounds(self, new_bounds: Tuple[float, float]) -> None:
        raise AttributeError("Cannot modify the bounds of a Constant.")

    @Parameter.frozen.setter
    def frozen(self, is_true: bool) -> None:
        raise AttributeError("Cannot modify the frozen state of a Constant.")
    
    
    
    
    
    




class ParameterHandler:
    """
    Classe che gestisce un insieme di parametri per un modello.

    Attributes:
        _parameters (Dict[str, Parameter]): Dizionario dei parametri.
        _is_inside_model (bool): Indica se il gestore è stato aggiunto a un modello.
    """

    def __init__(self, parameters: Union[Parameter, List[Parameter]] = None) -> None:
        """
        Inizializza il gestore dei parametri.

        Args:
            parameters (Union[Parameter, List[Parameter]], opzionale): Un singolo parametro o una lista di parametri.
        """
        self._parameters = {}
        self._is_inside_model = (
            False  # Una volta dentro modello non posso aggiungere parametri
        )
        if isinstance(parameters, Parameter):
            self._add_parameter(parameters)
        elif isinstance(parameters, list):
            for param in parameters:
                self._add_parameter(param)
        elif parameters is None:
            pass
        else:
            raise TypeError(
                "Parameters must be of type Parameter or List[Parameter]",
                type(parameters),
            )

    def _assign_attribute(self, items: Union[List, Dict], attribute: str) -> None:
        """
        Assegna un valore a un attributo dei parametri non congelati.

        Args:
            items (Union[List, Dict]): Valori da assegnare (lista o dizionario).
            attribute (str): Nome dell'attributo da assegnare.

        Raises:
            ValueError: Se il numero di elementi nella lista non corrisponde al numero di parametri liberi.
            TypeError: Se items non è né una lista né un dizionario.
        """
        if isinstance(items, (list, np.ndarray)):
            if len(items) != self.n_free_params:
                raise ValueError(
                    f"Number of items {len(items)} must match number of free parameters! {self.n_free_params}"
                )
            for param, val in zip(self.free_parameters, items):
                setattr(param, attribute, val)
        elif isinstance(items, dict):
            for name, val in items.items():
                setattr(self[name], attribute, val)
        else:
            raise TypeError("Items must be a list or dictionary")

    def _map_names_to_indices(self, key: str) -> int:
        """
        Mappa il nome di un parametro al corrispondente indice.

        Args:
            key (str): Nome del parametro.

        Returns:
            int: Indice del parametro.

        Raises:
            KeyError: Se il nome del parametro non è trovato.
        """
        try:
            return list(self._parameters.keys()).index(key)
        except ValueError:
            raise KeyError(f"Key '{key}' not found in the dictionary")

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
        keys = list(self._parameters.keys())
        if index < 0 or index >= len(keys):
            raise IndexError(f"Index '{index}' is out of bounds for the dictionary")
        return keys[index]

    def _invalidate_cache(self) -> None:
        """
        Invalida la cache dei parametri.
        """
        if hasattr(self, "_cache"):
            del self._cache

    def _is_frozen(self, parameter: Parameter) -> bool:
        """
        Verifica se un parametro è congelato.

        Args:
            parameter (Parameter): Il parametro da verificare.

        Returns:
            bool: True se il parametro è congelato, False altrimenti.
        """
        return parameter.frozen

    def _is_not_frozen(self, parameter: Parameter) -> bool:
        """
        Verifica se un parametro non è congelato.

        Args:
            parameter (Parameter): Il parametro da verificare.

        Returns:
            bool: True se il parametro non è congelato, False altrimenti.
        """
        return not parameter.frozen

    #@cached_property
    @property
    def parameters_values(self) -> List[float]:
        """
        Ritorna i valori dei parametri.

        Returns:
            List[float]: Lista dei valori dei parametri.
        """
        return [p.value for p in self]

    #@cached_property
    @property
    def parameters_names(self) -> List[str]:
        """
        Ritorna i nomi dei parametri.

        Returns:
            List[str]: Lista dei nomi dei parametri.
        """
        return [p.name for p in self]

    #@cached_property
    @property
    def parameters_bounds(self) -> List[Tuple[float, float]]:
        """
        Ritorna i limiti dei parametri.

        Returns:
            List[Tuple[float, float]]: Lista dei limiti dei parametri.
        """
        return [p.bounds for p in self]

    #@cached_property
    @property
    def n_free_params(self) -> int:
        """
        Ritorna il numero di parametri liberi.

        Returns:
            int: Numero di parametri liberi.
        """
        return len(self.free_parameters)

    #@cached_property
    @property
    def free_parameters(self) -> List[Parameter]:
        """
        Ritorna solo i parametri liberi.

        Returns:
            List[Parameter]: Lista dei parametri liberi.
        """
        return [p for p in self if p.frozen is False]

    #@cached_property
    @property
    def frozen_parameters(self) -> List[Parameter]:
        """
        Ritorna solo i parametri congelati.

        Returns:
            List[Parameter]: Lista dei parametri congelati.
        """
        return [p for p in self if p.frozen is True]

    def set_values(self, values: Union[List, Dict]) -> None:
        """
        Imposta i valori dei parametri non congelati.

        Args:
            values (Union[List, Dict]): Valori da assegnare (lista o dizionario).
        """
        self._assign_attribute(values, "value")

    def set_bounds(self, bounds: Union[List, Dict]) -> None:
        """
        Imposta i limiti dei parametri non congelati.

        Args:
            bounds (Union[List, Dict]): Limiti da assegnare (lista o dizionario).
        """
        self._assign_attribute(bounds, "bounds")

    def set_frozen(self, is_frozen: Union[List, Dict]) -> None:
        """
        Imposta lo stato di congelamento dei parametri.

        Args:
            is_frozen (Union[List, Dict]): Stato di congelamento da assegnare (lista o dizionario).
        """
        self._assign_attribute(is_frozen, "frozen")

    def _add_parameter(self, parameter: Parameter) -> None:
        """
        Aggiunge un parametro al gestore.

        Args:
            parameter (Parameter): Il parametro da aggiungere.

        Raises:
            ValueError: Se il parametro esiste già o se si tenta di aggiungere un parametro dopo la creazione del modello.
        """
        if self._is_inside_model:
            raise ValueError(
                f"Cannot add parameter {parameter.name} to model after the creation"
            )
        if parameter.name in self._parameters:
            raise ValueError(f"Parameter {parameter.name} already exists.")
        self._parameters[parameter.name] = parameter
        self._invalidate_cache()

    def _get_parameter(self, name: str) -> Parameter:
        """
        Ritorna un parametro dal gestore.

        Args:
            name (str): Nome del parametro.

        Returns:
            Parameter: Il parametro richiesto.

        Raises:
            ValueError: Se il parametro non è trovato.
        """
        if name not in self._parameters:
            raise ValueError(f"Parameter {name} not found.")
        return self._parameters[name]

    def __getitem__(self, name: str) -> Parameter:
        """
        Ritorna un parametro usando l'operatore di accesso a dizionario.

        Args:
            name (str): Nome del parametro.

        Returns:
            Parameter: Il parametro richiesto.
        """
        return self._get_parameter(name)
    
    def __setitem__(self, key, value) -> None:
        # tutto un test 
        if not isinstance(value, Parameter):
            raise TypeError(f'new param must be instance of Parameter, not {type(value)}')
        self._parameters[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Verifica se un parametro è presente usando il suo nome.

        Args:
            key (str): Nome del parametro.

        Returns:
            bool: True se il parametro è presente, False altrimenti.
        """
        return key in self._parameters

    def __iter__(self):
        """
        Itera sui parametri.

        Returns:
            Iterator: Iteratore sui parametri.
        """
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """
        Ritorna il numero di parametri.

        Returns:
            int: Numero di parametri.
        """
        return len(self._parameters)

    def items(self):
        """
        Ritorna gli elementi del gestore come coppie chiave-valore.

        Returns:
            ItemsView: Vista degli elementi del gestore.
        """
        return self._parameters.items()
    
    def keys(self):
        """
        Ritorna le key del dizionario base.

        Returns:
            KeysView: Vista delle keys del gestore.
        """
        return self._parameters.keys()

