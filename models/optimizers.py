import numpy as np
from base import FittableModel
from typing import Union
import tqdm

class OptimizerError:
    def __init__(self, msg) -> None:
        self.msg = msg

# ----------------------------------------------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------------------------------------------


class BaseOptimizer:
    def __init__(self, model:FittableModel, **kwargs) -> None:

        self.model = model
        

    def pre_optim(self,grid, x0,data) -> None:
        # 1 --- check del numero di dimensioni del modello e dei dati
        if self.model.n_free_parameters != len(x0):
            raise ValueError(
                f"Dimension of x0 {len(x0)} do not match dimension of parameter space {self.model.n_free_parameters}"
            )

        if self.model.n_dim != len(np.shape(data[0])):
            raise ValueError(
                f"data number of dimension {np.shape(data[0])[0]} number of model dimensions {self.model.n_dim} "
            )
        
        # 2 --- controllo se x0 è dict e lo converto in lista
        if isinstance(x0, dict):
            # controllo dentro o fuori bounds, controllo frozen
            
            for key, value in x0.items():
                if self.model[key].frozen is True:
                    raise ValueError(f'Parameter {key} is frozen and can not be fitted as regual parameter')
                
                if not self.model[key].bounds[0] < value < self.model[key].bounds[1]:
                    raise ValueError(
                        f"Initial guess for parameter {key} is outside it s bounds {self.model[key].bounds}"
                   )
            
            x0 = list(x0.values())

        elif isinstance(x0, Union[list,np.ndarray]):
            # già controllato numero di dimensioni
            for param , guess in zip(self.model.free_parameters, x0):
                if not param.bounds[0]<guess<param.bounds[1]:
                    raise ValueError(
                        f"Initial guess for parameter {param.name} is outside it s bounds {param.bounds}"
                    )
        else:
            raise TypeError('Initial guess is not a supported type!')
        
        if not isinstance(grid,list):
            raise TypeError('Grid must be provided in form of a list')
        
        if np.shape(grid)[0] != self.model.n_dim:
            raise ValueError(
                f"Grid dimension {len(np.shape(grid))} is not the same as model dimension {self.model.n_dim}!"
            )
        return grid,x0, data
    
    def minimize(self):
        raise NotImplementedError('Minimize function must be implemented by single optimizer')

#----------------------------------------------------------------------------------------------------------------
#
#
#
#-----------------------------------------------------------------------------------------------------------------
class LSTQFitter(BaseOptimizer):
    '''classe che implementa le principali funzioni di minimizazzione basate su chiquadro'''
    def __init__(self, model: FittableModel, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def loss(self, grid, x0, data):
        chi2 = np.nansum((self.model(grid = grid, params=x0)-data)**2)
        return chi2
    


# ----------------------------------------------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------------------------------------------


class  NelderMead(LSTQFitter):
    def __init__(
        self,
        model: FittableModel,
        treshold=1e-4,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta=0.05,
        **kwargs,
    ) -> None:
        
        super().__init__(model, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.treshold = treshold
        self.delta = delta  # dispersione iniziale dei vertici del simplesso

    def _generate_initial_simplex(self, n_dim, x0, bounds, delta):
        """Genera il simplesso iniziale, assicurandosi che i vertici siano compresi nei bounds.
        x0 --> punto iniziale
        n_dim --> numero di dimensioni del hyperspazio (len(x0))
        """
        simplex = np.zeros((n_dim + 1, n_dim))
        simplex[0] = x0

        # Creazione vettorizzata dei punti aggiuntivi del simplesso
        perturbations = delta * np.eye(n_dim)
        simplex[1:] = x0 + perturbations

        # Applicazione dei bounds
        simplex = np.clip(simplex, [b[0] for b in bounds], [b[1] for b in bounds])

        return simplex

    def reflect(self, bounds):
        """calcola punto di riflessione"""
        new_point = self.centroid + self.alpha * (self.centroid - self.worst)
        new_point = np.clip(new_point, [b[0] for b in bounds], [b[1] for b in bounds])
        return new_point

    def expand(self, reflect_point, bounds):
        """calcola punto di espansione dato punto di riflessione"""
        new_point = self.centroid + self.gamma * (reflect_point - self.centroid)
        new_point = np.clip(new_point, [b[0] for b in bounds], [b[1] for b in bounds])
        return new_point

    def contract(self, point, bounds):
        """calcola punto di contrazione dato punto di riflessione e/o contrazione"""
        new_point = self.centroid + self.beta * (point - self.centroid)
        new_point = np.clip(new_point, [b[0] for b in bounds], [b[1] for b in bounds])
        return new_point

    def shrink(self, bounds, x0, grid,data):
    
        """Riduce il simplesso verso il miglior vertice."""
        self.simplex[1:] = self.best + self.delta * (self.simplex[1:] - self.best)
        self.simplex[1:] = np.clip(
            self.simplex[1:], [b[0] for b in bounds], [b[1] for b in bounds]
        )
        self.loss_history[1:] = np.apply_along_axis(self.loss(grid,x0,data), 1, self.simplex[1:])

    def get_vertices(self):
        """
        aggiorna i tre vertici: migliore, peggiore e secondo peggiore e centroide.

        Returns:
        best (np.array): Il miglior vertice (minore valore di loss).
        worst (np.array): Il peggior vertice (maggiore valore di loss).
        second_worst (np.array): Il secondo peggior vertice (secondo maggiore valore di loss).
        indices (tuple): Gli indici dei vertici (best_index, worst_index, second_worst_index).
        """

        # Ordina gli indici dei valori della loss in ordine crescente
        sorted_indices = np.argsort(self.loss_history)

        self.best_index = sorted_indices[0]
        self.worst_index = sorted_indices[-1]
        self.second_worst_index = sorted_indices[-2]

        self.best = self.simplex[self.best_index]
        self.worst = self.simplex[self.worst_index]
        self.second_worst = self.simplex[self.second_worst_index]

        # Calcola il centroide dei migliori vertici (escludendo il peggiore)
        self.centroid = np.mean(
            np.delete(self.simplex, self.worst_index, axis=0), axis=0
        )

    def minimize(self,grid, x0, data, maxiter=None, progress = False):

        grid, x0, data = self.pre_optim(grid, x0, data)

        # --- inizializazzione della funzione
        #x0 = np.asfarray(x0).flatten()  # converte in float e lo flatterizza
        N = len(x0)
        rank = len(np.shape(x0))
        constrains = [p.bounds for p in self.model.free_parameters]

        if N > 2000:
            self.alpha = 1
            self.beta = 1 + (2 / N)
            self.gamma = 0.75 - (1 / 2 * N)
            self.delta = 1 - (1 / N)

        if not -1 < rank < 2:
            raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
        if maxiter is None:
            maxiter = N * 300
        
        # --- genero simplesso iniziale
        self.simplex = self._generate_initial_simplex(N, x0, constrains, self.delta)

        # --- genero array dei valori di loss
        #print(type(data), type(x0), type(grid))
        #print([self.loss(grid=grid, x0=point, data=data) for point in self.simplex])

        self.loss_history = np.array([self.loss(grid=grid, x0=point, data=data) for point in self.simplex])

        if progress is True:
            nmax = tqdm.tqdm(np.arange(maxiter))
        else:
            nmax = np.arange(maxiter)
        
        for iter in nmax:
            self.get_vertices()  # aggiorno i valori del simplesso

            reflection = self.reflect(bounds=constrains)  # calcolo reflection

            loss_best = self.loss_history[self.best_index]
            loss_second_worst = self.loss_history[self.second_worst_index]
            loss_reflection = self.loss(grid, reflection, data)

            if loss_best < loss_reflection < loss_second_worst:
                self.simplex[self.worst_index] = reflection
                self.loss_history[self.worst_index] = loss_reflection

            elif loss_reflection < loss_best:
                expansion = self.expand(reflection, bounds=constrains)
                loss_expansion = self.loss(grid, expansion, data)

                if loss_expansion < loss_reflection:
                    self.simplex[self.worst_index] = expansion
                    self.loss_history[self.worst_index] = loss_expansion
                else:
                    self.simplex[self.worst_index] = reflection
                    self.loss_history[self.worst_index] = loss_reflection

            elif loss_reflection >= loss_second_worst:
                loss_worst = self.loss_history[self.worst_index]
                best_point = self.worst if loss_worst < loss_reflection else reflection

                contraction = self.contract(best_point, bounds=constrains)
                loss_contraction = self.loss(grid, contraction, data)

                if loss_contraction < loss_reflection:
                    self.simplex[self.worst_index] = contraction
                    self.loss_history[self.worst_index] = loss_contraction
                else:
                    self.shrink(
                        bounds=constrains, grid=grid, x0=x0, data=data)  # Chiama la funzione di shrink

            if np.max(self.loss_history) - np.min(self.loss_history) < self.treshold:
                print("Treshold raggiunta")

                break

        return self.simplex[self.best_index]
