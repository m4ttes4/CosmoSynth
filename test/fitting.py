import corner.corner
import numpy as np
#import pandas as pd
from typing import Union, Dict, Optional
import emcee
import corner

from model import Model

class MCMCResult:
    #TODO riscrivere questa classe che fa proprio merda
    """
    Classe che raccoglie e gestisce i risultati di un fitting MCMC.
    L'utente deve passare un 'sampler' (es. emcee.EnsembleSampler),
    oltre a dati, modello, var_names, ecc.
    La catena viene recuperata internamente (senza dover passare chain o flat_samples).

    Changing 'discard' o 'thin' ricalcola automaticamente le statistiche.
    """

    def __init__(
        self,
        sampler,
        model,
        grid,
        data,
        var_names,
        discard=0,
        thin=1,
        # eventuali parametri aggiuntivi
        message="",
        success=True,
        **kwargs,
    ):
        """
        Parametri
        ---------
        sampler : oggetto di tipo emcee.EnsembleSampler (o simile)
            Da cui recuperare i campioni MCMC.
        model : oggetto/funzione
            Utilizzato per calcolare il modello teorico (ad es. model.call).
        grid : array-like
            Griglia (o x) su cui valutare il modello.
        data : array-like
            Dati sperimentali/osservati.
        var_names : list
            Nomi dei parametri variabili nell'ottimizzazione.
        discard : int
            Numero di step iniziali da scartare (burn-in).
        thin : int
            Thinning factor.
        message : str
            Messaggio opzionale sul risultato.
        success : bool
            Flag di successo o meno.
        kwargs : dict
            Altri attributi opzionali da aggiungere all'oggetto.
        """
        self.sampler = sampler
        self.model = model
        self.grid = grid
        self.data = data
        self.var_names = var_names

        # Parametri per gestire burn-in e thinning
        self._discard = discard
        self._thin = thin

        # Altri attributi (es. stato, message, etc.)
        self.success = success
        self.message = message

        # Inizializziamo alcuni attributi che vogliamo calcolare
        self.best_fit = None
        self.residual = None
        self.chisqr = None
        self.redchi = None
        self.aic = None
        self.bic = None
        self.parameter_summary = None

        # Se vogliamo, possiamo salvare ndata e simili:
        self.ndata = len(data) if hasattr(data, "__len__") else 1
        self.nwalkers = (
            self.sampler.nwalkers if hasattr(self.sampler, "nwalkers") else None
        )
        self.nvarys = len(var_names)

        # Qualsiasi altro attributo presente in kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Calcola subito le statistiche iniziali
        self.calculate_statistics()

    @property
    def discard(self):
        """Numero di step di burn-in scartati."""
        return self._discard

    @discard.setter
    def discard(self, value):
        self._discard = value
        # Ricalcolo le statistiche ogni volta che discard cambia
        self.calculate_statistics()

    @property
    def thin(self):
        """Fattore di thinning."""
        return self._thin

    @thin.setter
    def thin(self, value):
        self._thin = value
        # Ricalcolo le statistiche ogni volta che thin cambia
        self.calculate_statistics()

    @property
    def chain(self):
        """
        Catena MCMC completa (senza flatten), recuperata dal sampler.
        Dimensioni tipiche: (nsteps, nwalkers, n_params)
        Nota: emcee < 3 aveva dimensioni invertite
        """
        # flat=False per ottenere la catena completa
        return self.sampler.get_chain(
            discard=self._discard, thin=self._thin, flat=False
        )

    @property
    def flatchain(self):
        """
        Catena MCMC "flattened" (2D), recuperata dal sampler.
        Dimensioni tipiche: (nsteps * nwalkers, n_params)
        """
        # flat=True per ottenere la catena appiattita
        return self.sampler.get_chain(discard=self._discard, thin=self._thin, flat=True)

    def calculate_statistics(self):
        """
        Ricalcola best-fit (mediana), residui, chisq, redchi, AIC, BIC,
        e percentili (16, 84) usando i campioni in flatchain.
        """
        # Se non ci sono campioni, esci
        flat = self.flatchain
        if flat is None or len(flat) == 0:
            self.best_fit = None
            self.residual = None
            self.chisqr = 0.0
            self.redchi = np.inf
            self.aic = np.inf
            self.bic = np.inf
            self.parameter_summary = {}
            return

        # Calcolo best fit come mediana
        self.best_fit = np.median(flat, axis=0)  # array di dimensione (n_params,)
        self.best_fit_dict = {
            name: val for name, val in zip(self.var_names, self.best_fit)
        }
        # Calcolo residui con i parametri best-fit (usando self.model)
        # Assumiamo che la model.call(grid, *params) sia la sintassi
        model_output = self.model.call(self.grid, *self.best_fit)
        # Ravel per sicurezza
        self.residual = (self.data - model_output).ravel()

        # Calcolo chisqr
        self.chisqr = float((self.residual**2).sum())

        # Gradi di libertà e chi-quadro ridotto
        # (eventualmente potremmo salvare un attributo self.nfree = self.ndata - self.nvarys)
        nfree = self.ndata - self.nvarys
        if nfree > 0:
            self.redchi = self.chisqr / nfree
        else:
            self.redchi = np.inf

        # Calcolo AIC e BIC usando chisqr come -2 ln L
        eps = 1e-250
        chisqr_valid = max(self.chisqr, eps * self.ndata)
        _neg2_log_likel = self.ndata * np.log(chisqr_valid / self.ndata)
        self.aic = _neg2_log_likel + 2 * self.nvarys
        if self.ndata > 0:
            self.bic = _neg2_log_likel + np.log(self.ndata) * self.nvarys
        else:
            self.bic = np.inf

        # Calcolo dei percentili 16, 84 e mediana per ciascun parametro
        self.parameter_summary = {}
        for i, pname in enumerate(self.var_names):
            param_samples = flat[:, i]
            p16 = np.percentile(param_samples, 16)
            p84 = np.percentile(param_samples, 84)
            med = np.median(param_samples)
            self.parameter_summary[pname] = {"median": med, "p16": p16, "p84": p84}

    def plot_corner(self):
        """
        Esempio di funzione per generare il corner plot.
        """
        flat = self.flatchain
        if flat is None or len(flat) == 0:
            print("Nessun campione per il corner plot.")
            return
        corner.corner(flat, labels=self.var_names)

    def __repr__(self):
        return (
            f"<MCMCResult success={self.success} chisqr={self.chisqr:.4g} "
            f"redchi={self.redchi:.4g} ndata={self.ndata} nvarys={self.nvarys} "
            f"discard={self.discard} thin={self.thin}>"
        )

    def __str__(self):
        lines = [
            "=== MCMCResult ===",
            f"    success   = {self.success}",
            f"    message   = {self.message}",
            f"    discard   = {self.discard}",
            f"    thin      = {self.thin}",
            f"    ndata     = {self.ndata}",
            f"    nvarys    = {self.nvarys}",
            f"    chisqr    = {self.chisqr:.4g}",
            f"    redchi    = {self.redchi:.4g}",
            f"    aic       = {self.aic:.4g}",
            f"    bic       = {self.bic:.4g}",
        ]
        # Se abbiamo un best_fit
        if self.best_fit is not None:
            lines.append("    best_fit:")
            for pname, val in zip(self.var_names, self.best_fit):
                lines.append(f"  {pname} = {val:.4g}  {self.model[pname].description}")

        # Se abbiamo summary percentile
        if self.parameter_summary:
            lines.append("\n    Parameter summary (median [p16, p84]):")
            for pname, stats in self.parameter_summary.items():
                lines.append(
                    f"        {pname:15s}: {stats['median']:.4g} "
                    f"[{stats['p16']:.4g}, {stats['p84']:.4g}]"
                )

        return "\n".join(lines)















class MCMC:
    """
    Classe wrapper di base per l'utilizzo dell'algoritmo MCMC (Markov Chain Monte Carlo)
    tramite la libreria `emcee`.

    Parameters
    ----------
    model : object
        Istanza di un oggetto modello che deve fornire le seguenti proprietà/metodi:
        - `free_parameters`: lista dei parametri liberi, ciascuno con `.value` e `.bounds` (tuple (min, max)).
        - `call(grid, *theta)`: metodo che, dati un grid e i valori dei parametri, restituisce l'output del modello.
        - `parameters_names`: lista dei nomi di tutti i parametri (sia liberi che non).
        - `parameters_keys`: chiavi identificative di tutti i parametri (sia liberi che non).
        - `parameters_values_dict`: dizionario dei parametri, con nome come chiave e valore come valore del parametro.
        - `n_dim`: dimensione del grid (es. 1D, 2D, ...).
        - `__getitem__(pname)`: per accedere a un parametro a partire dal suo nome (restituisce un oggetto con `.frozen`).

    **kwargs
        Ulteriori argomenti passati internamente a `emcee.EnsembleSampler`. Ad esempio:
        - `moves`
        - `backend`
        - etc.
        
    TODO: add support for different statistcs
    TODO: modify call to account for number of dimensions of the model
    """

    def __init__(self, model, **kwargs) -> None:
        self._model = model
        self.emcee_kwargs = kwargs
    
        # NOTE currently to discuss
        if self.model.n_outputs > 1:
            raise NotImplementedError('Multiple outputs are not currentrly supported')
        
        #if self.model.has_constrains:
        self.loglike = self.unconstrained_loglike
        #else:
        #self.loglike = self.constrained_loglike

    @property
    def model(self):
        """
        Riferimento al modello associato a questa classe MCMC.

        Returns
        -------
        object
            Il modello utilizzato per la stima MCMC.
        """
        return self._model
    
    @model.setter
    def model(self, value):
        if not isinstance(value, Model):
            raise TypeError('Model to be optimize must be istance of class Model')

    def logprior(self, theta: np.ndarray) -> float:
        """
        Calcola il log-prior per i parametri `theta`.

        Il prior è zero se tutti i parametri rientrano nei rispettivi bounds,
        altrimenti ritorna `-np.inf`.

        Parameters
        ----------
        theta : np.ndarray
            Valori correnti dei parametri del modello.

        Returns
        -------
        float
            Il valore della log-prior.
        """
        return sum(param(val) for param, val in zip(self.model.free_parameters, theta))
        
    def constrained_loglike(self, theta, xdata, ydata, yerr) -> float:
        '''loglike function but for constrained parameters'''
        # map constrains to relative args
        raise NotImplementedError('Please boss, I m tired')
    
    def unconstrained_loglike(
        self,
        theta: np.ndarray,
        xdata: Union[list, np.ndarray],
        ydata: Union[list, np.ndarray],
        yerr: Union[list, np.ndarray],
    ) -> float:
        """
        Calcola la log-likelihood dati i dati e il modello,
        assumendo errori gaussiani indipendenti.

        Parameters
        ----------
        theta : np.ndarray
            Valori correnti dei parametri del modello.
        xdata : array-like
            Dati indipendenti (es. valori di ascissa).
        ydata : array-like
            Dati osservati (es. valori di ordinata).
        yerr : array-like
            Incertezze (errore standard) associate ai dati osservati.

        Returns
        -------
        float
            Valore della log-likelihood.
        """
        # Calcolo del modello
        #
        
        ymodel = self.model.call(xdata, *theta)
       
        return -0.5 * np.nansum(((ydata - ymodel) / yerr) ** 2)

    def log_probability(
        self,
        theta: np.ndarray,
        xdata: Union[list, np.ndarray],
        ydata: Union[list, np.ndarray],
        yerr: Union[list, np.ndarray],
    ) -> float:
        """
        Calcola la log-probability (somma di log-prior e log-likelihood).

        Parameters
        ----------
        theta : np.ndarray
            Valori correnti dei parametri del modello.
        xdata : array-like
            Dati indipendenti.
        ydata : array-like
            Dati osservati.
        yerr : array-like
            Incertezze (errore standard) associate ai dati osservati.

        Returns
        -------
        float
            Valore della log-probabilità (log-prior + log-likelihood).
        """
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta, xdata, ydata, yerr)

    def _emcee_positions(
        self,
        theta0: np.ndarray,
        nwalkers: int,
        bounds: list,
        dispersion: float,
    ) -> np.ndarray:
        """
        Genera una matrice di posizioni iniziali per emcee con dispersione
        del 10% rispetto ai parametri iniziali. I punti fuori dai bound
        specificati vengono rigenerati.

        Parameters
        ----------
        theta0 : np.ndarray
            Posizione iniziale dei parametri, array di forma (ndim,).
        nwalkers : int
            Numero di walkers per emcee.
        bounds : list of tuple
            Lista di tuple (lower_bound, upper_bound) per ciascun parametro.

        Returns
        -------
        np.ndarray
            Array di forma (nwalkers, ndim) contenente le posizioni iniziali
            dei walkers.
        """
        ndim = len(theta0)
        pos = np.zeros((nwalkers, ndim))

        if dispersion < 0 or dispersion > 1:
            raise ValueError("Initial points dispersion must be > 0 and < 1")

        # Controllo che l'inizializzazione non sia fuori dai bound
        for i, point in enumerate(theta0):
            if not (bounds[i][0] < point < bounds[i][1]):
                raise ValueError(
                    f"Il valore iniziale per il parametro '{self.model.parameters_names[i]}' "
                    f"è fuori dal bound {bounds[i]}!"
                )

        def generate_valid_position() -> np.ndarray:
            while True:
                # Genera una posizione con dispersione casuale (10%)
                candidate = theta0 + dispersion * np.abs(theta0) * (
                    2 * np.random.rand(ndim) - 1
                )
                # Controlla se tutti i parametri rispettano i bound
                if all(
                    np.isfinite(p.prior(p.value)) for p in self.model.free_parameters 
                    #lower <= value <= upper
                    #for value, (lower, upper) in zip(candidate, bounds)
                ):
                    return candidate

        # Popola l'array con posizioni valide per ciascun walker
        for i in range(nwalkers):
            pos[i] = generate_valid_position()

        return pos

    def _check_initial_state(
        self,
        theta0: Optional[Union[list, np.ndarray, Dict[str, float]]],
        grid: Union[list, np.ndarray],
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        """
        Verifica e prepara lo stato iniziale `theta0` in base a come viene fornito (lista, array, dict).
        Inoltre controlla la compatibilità tra dimensione dei dati e output del modello.

        Parameters
        ----------
        theta0 : list or np.ndarray or dict, optional
            Stima iniziale dei parametri. Se `None`, vengono usati i valori
            già presenti nel modello per i parametri liberi.
        grid : array-like
            Grid su cui valutare il modello (può essere monodimensionale o multidimensionale).
        data : array-like
            Valori osservati corrispondenti alla valutazione del modello su `grid`.

        Returns
        -------
        np.ndarray
            Array di valori iniziali (stima dei parametri) in formato numpy (ndim,).

        Raises
        ------
        TypeError
            Se `grid` non è un array-like o se `theta0` non è un tipo supportato (list, dict, np.ndarray).
        ValueError
            Se la dimensione del `grid` non corrisponde a quella attesa dal modello,
            o se la lunghezza di `theta0` non corrisponde al numero di parametri liberi,
            o se la dimensione dei dati non corrisponde alla dimensione dell'output del modello.
        """
        
        
        if not isinstance(grid, (list, np.ndarray)):
            raise TypeError(
                "`grid` deve essere una lista o un numpy array (es. [X], [X, Y, ...])."
            )
        if isinstance(grid, list):
            grid = np.array(grid)

        # Controllo dimensione `grid` e modello
        if np.shape(grid)[0] != self.model.n_dim:
            raise ValueError(
                f"La dimensione del grid ({grid.ndim}) non corrisponde "
                f"alla dimensione del modello ({self.model.n_dim})."
            )

        # Se theta0 è None, uso i valori del modello
        if theta0 is None:
            theta0 = [p.value for p in self.model.free_parameters]

        # Se theta0 è un dict
        if isinstance(theta0, dict):
            # Copio i parametri correnti del modello
            initial = {**self.model.parameters_values_dict}
            for pname, pval in theta0.items():
                param = self.model[pname]
                if not param.is_free:
                    raise ValueError(
                        f"Il parametro '{pname}' è frozen. Fornire solo valori per parametri free."
                    )
                initial[pname] = pval
            # Ricostruisco array in ordine
            theta0 = np.array(
                [
                    initial[name]
                    for name in self.model.parameters_keys
                    if not self.model[name].frozen
                ]
            )
        elif isinstance(theta0, (list, np.ndarray)):
            theta0 = np.array(theta0, dtype=float)
        else:
            raise TypeError(
                "L'ipotesi iniziale (theta0) deve essere una lista, un dict o un numpy array."
            )

        # Verifico che sia veramente un np.ndarray
        assert isinstance(theta0, np.ndarray)

        # Check lunghezza theta0 = numero di parametri liberi
        if len(theta0) != len(self.model.free_parameters):
            raise ValueError(
                f"Il numero di valori iniziali ({len(theta0)}) non corrisponde "
                f"al numero di parametri liberi ({len(self.model.free_parameters)})."
            )

        # Verifica della compatibilità tra dati e output del modello
        model_output = self.model.call(grid, *theta0)
        if np.shape(model_output) != np.shape(data):
            raise ValueError(
                f"La dimensione dei dati {np.shape(data)} non corrisponde "
                f"alla dimensione dell'output del modello {np.shape(model_output)}."
            )

        return theta0, grid

    def _look_invalid_initial_points(self, theta):
        names = [name for name in self.model.parameters_keys if self.model[name].is_free]
        assert len(names) == len(theta)
        
        for name, val in zip(names, theta):
            if not np.isfinite(self.model[name].prior(val)):
                raise ValueError(f'val {val} for param {name} has conflict with prior {self.model[name].prior}')
        
    
    def fit(
        self,
        #grid: Union[list, np.ndarray],
        data: Union[list, np.ndarray],
        theta0: Optional[Union[list, np.ndarray, Dict[str, float]]] = None,
        error: Optional[Union[list, np.ndarray]] = None,
        *,
        nwalkers: int = 32,
        nsteps: int = 5000,
        discard: int = 100,
        dispersion: float | int = 0.1,
        optimize: bool = False,
        progress: bool = True,
        thin: int = 1,
        **kwargs,
    ) -> MCMCResult:
        # TODO modificare dispersion e initial pointin modo da permettere di avere come input direttamente l'array corretto come emcee
        """
        Esegue il fitting MCMC del modello sui dati forniti.

        Parameters
        ----------
        
        data : array-like
            Dati osservati corrispondenti alla valutazione del modello su `grid`.
        theta0 : list or np.ndarray or dict, optional
            Stima iniziale dei parametri. Se `None`, vengono usati i valori
            già presenti nel modello per i parametri liberi.
        error : array-like, optional
            Incertezze (errore standard) sui dati osservati.
            Se `None`, si assume un errore costante (o lo si può gestire diversamente).
        nwalkers : int, optional
            Numero di walker da utilizzare nell'algoritmo MCMC (default=32).
        nsteps : int, optional
            Numero di passi da eseguire per ogni walker (default=5000).
        discard : int, optional
            Numero di step iniziali (burn-in) da scartare (default=100).
        thin : int, optional
            Frequenza di thinning; un valore di 15 significa prendere 1 campione ogni 15 (default=15).
        **kwargs : dict
            model.grid_variables possono essere passate direttammente.\
            Argomenti aggiuntivi passati a `emcee.EnsembleSampler.run_mcmc`.

        Returns
        -------
        flat_samples : np.ndarray
            Array di campioni estratti dalla catena MCMC dopo il burn-in e il thinning.
        fig : matplotlib.figure.Figure
            Figura `corner` che mostra le distribuzioni a posteriori dei parametri.
        """
        # check sulla griglia data come kwargs
        grid = []
        for var in self.model.grid_variables:
            if var in kwargs:
                element = kwargs.pop(var)
                grid.append(element)
            else:
                raise KeyError(f"La variabile di griglia '{var}' non è presente in kwargs.")

        self.emcee_kwargs.update(**kwargs)
        # Controlla lo stato iniziale e i dati
        theta0, grid = self._check_initial_state(theta0=theta0, grid=grid, data=data)
        self._look_invalid_initial_points(theta0)

        if error is None:
            error = np.ones(np.shape(data))
            
        if optimize is True:
            # guess initial position by optimization
            from scipy.optimize import minimize

            initial_point = minimize(
                lambda xtheta, xgrid, xdata, err: -self.log_probability(xtheta, xgrid, xdata, err),
                x0=theta0,
                args=(grid, data, error),
                bounds=[p.bounds for p in self.model if p.is_free],
                
            )
            if initial_point.success:
                print(f"Optimization done, initial position is {initial_point.x}")
                theta0 = initial_point.x
            # raise NotImplementedError('Prior Optimization not implemented yet')

        # Genera posizioni iniziali per i walker
        init_positions = self._emcee_positions(
            dispersion=dispersion,
            theta0=theta0,
            nwalkers=nwalkers,
            bounds=[p.bounds for p in self.model if p.is_free],
        )

        nwalkers, ndim = init_positions.shape
        assert ndim == len(self.model.free_parameters)
        # Inizializza il sampler di emcee
        # check for cnstrained parameters:
        
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.log_probability,
            args=(grid, data, error),
            **self.emcee_kwargs,
        )

        # setattr( sampler, "log_prob_fn",lambda p: self.log_probability(p, grid, data, error))
        # Esecuzione MCMC vera e propria
        sampler.run_mcmc(init_positions, nsteps, progress=progress, **kwargs)

        # Prepara i nomi dei parametri
        labels = [
            key for key in self.model.parameters_keys if self.model[key].is_free
        ]

        
        # Inizializziamo MCMCResult
        # 2) Crea l'oggetto MCMCResult passandogli il sampler e gli altri oggetti necessari
        result = MCMCResult(
            sampler=sampler,
            model=self.model,  # ad esempio un oggetto con metodo .call
            grid=grid,
            data=data,
            var_names=labels,  # lista dei nomi
            discard=discard,  # burn-in
            thin=thin,  # thinning
            success=True,
            message="MCMC sampling completed successfully",
        )
        if optimize:
            result.__setattr__('linear_bf', initial_point.x)

        #

        return result
