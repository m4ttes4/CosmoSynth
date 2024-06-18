import numpy as np
import tqdm

class NelderMead:
    def __init__(
        self,
        treshold=1e-4,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta=0.05,
    ):
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

    def shrink(self, bounds, args):
        def wrapped_loss(x):
            return self.func(x, *args)

        """Riduce il simplesso verso il miglior vertice."""
        self.simplex[1:] = self.best + self.delta * (self.simplex[1:] - self.best)
        self.simplex[1:] = np.clip(
            self.simplex[1:], [b[0] for b in bounds], [b[1] for b in bounds]
        )
        self.loss_history[1:] = np.apply_along_axis(
            wrapped_loss, 1, self.simplex[1:]
        )

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

    def optimize(
        self,
        func,
        x0,
        args,
        bounds=None,
        maxiter=None,
        progress = False
        
    ):
        """
        maxfun --> numero max di call di funzione
        """
        # --- inizializazzione della funzione
        x0 = np.asfarray(x0).flatten()  # converte in float e lo flatterizza
        N = len(x0)
        self.func = func
        rank = len(np.shape(x0))
        
        # --- aggiornamento (Fuchang Gao · Lixing Han 2010)
        # --- TITLE: Implementing the Nelder-Mead simplex algorithm with adaptive parameters
        if N >2000:
            self.alpha = 1
            self.beta = 1 + (2/N)
            self.gamma = 0.75 - (1/2*N)
            self.delta = 1 - (1/N)

        constrains = [(-np.inf, np.inf) for _ in range(N)] if bounds is None else bounds
        # --- aggiungere un controllo sui bounds?

        # --- funzione di loss wrappata per gestire solo i punti da minimizzare
        def wrapped_loss(x):
            return func(x, *args)

        if not -1 < rank < 2:
            raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
        if maxiter is None:
            maxiter = N * 300

        # --- genero simplesso iniziale
        self.simplex = self._generate_initial_simplex(N, x0, constrains, self.delta)
        
        # --- genero array dei valori di loss
        self.loss_history = np.array([wrapped_loss(point) for point in self.simplex])
        
        # ------------------------------------------------------------
        # --- inizio del loop
        if progress is True:
            nmax = tqdm.tqdm(np.arange(maxiter))
        else:
            nmax = np.arange(maxiter)
        for iter in nmax:
            
            self.get_vertices()  # aggiorno i valori del simplesso

            reflection = self.reflect(bounds=constrains)  # calcolo reflection

            loss_best = self.loss_history[self.best_index]
            loss_second_worst = self.loss_history[self.second_worst_index]
            loss_reflection = wrapped_loss(reflection)

            if loss_best < loss_reflection < loss_second_worst:
                self.simplex[self.worst_index] = reflection
                self.loss_history[self.worst_index] = loss_reflection

            elif loss_reflection < loss_best:
                expansion = self.expand(reflection, bounds=constrains)
                loss_expansion = wrapped_loss(expansion)

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
                loss_contraction = wrapped_loss(contraction)

                if loss_contraction < loss_reflection:
                    self.simplex[self.worst_index] = contraction
                    self.loss_history[self.worst_index] = loss_contraction
                else:
                    self.shrink(bounds=constrains, args=args)  # Chiama la funzione di shrink

           
            if np.max(self.loss_history) - np.min(self.loss_history) < self.treshold:
                print("Treshold raggiunta")

                break
        """
        if calculate_errors is True:
            errors = self._compute_errors(self.simplex, self.loss_history, wrapped_loss)
        else:
            errors = None
        """

        return self.simplex[self.best_index]