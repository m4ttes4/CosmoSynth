# CosmoSynth
 Fast and easy to use low-level Framework for managing complex parametric models


# TODO
 add support for tied parameters
 add supoprt for multiple datasets
 asteamline the validation of args and kwarsg in a common way for __call__, evaluate e .call()


 # IDEA
 per aggiungere il supporto a parametri tied o simili, posso 
 modificare soltanto il fitter in modo che durante la valutazione del modello applichi i dovuti constrains.
 In principio questo mi permette di definire come constrain anche il freeze di un parametro
 I constrain di un parametro possono essere rappresentati a questo punto come una priority-que in cui applico i callable nel corretto ordine