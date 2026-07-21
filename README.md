# Documentazione del Progetto: Classificazione delle Razze Feline

## Introduzione

L'obiettivo del progetto è sviluppare un modello di **Machine Learning supervisionato** in grado di classificare la razza di un gatto a partire da alcune caratteristiche fisiche e descrittive presenti nel dataset.

L'intero progetto è stato sviluppato in **Python**, seguendo una pipeline composta dalle principali fasi di un progetto di Data Science: caricamento dei dati, pulizia, preprocessing, analisi esplorativa (EDA), addestramento del modello, ottimizzazione degli iperparametri, valutazione finale e generazione delle predizioni sul dataset di test.

---

# 1. Tecnologie e Librerie Utilizzate

Per lo sviluppo del progetto è stato utilizzato l'ecosistema scientifico di Python, standard per le applicazioni di Data Science e Machine Learning.

## Pandas

Utilizzato per:

- Caricamento dei file CSV.
- Gestione dei DataFrame.
- Pulizia e trasformazione dei dati.
- Gestione dei valori mancanti.

---

## NumPy

Utilizzato per eseguire operazioni matematiche e numeriche sui dati e per la gestione degli array utilizzati durante l'addestramento del modello.

---

## Matplotlib

Utilizzato per la creazione dei grafici relativi all'analisi esplorativa dei dati (EDA) e alla valutazione del modello.

---

## Seaborn

Libreria costruita sopra Matplotlib che permette di realizzare grafici statistici più leggibili ed esteticamente curati, come:

- Heatmap
- Boxplot
- Scatter Plot
- Grafici a barre

---

## Scikit-Learn

È la libreria principale del progetto e fornisce tutti gli strumenti necessari alla pipeline di Machine Learning.

In particolare vengono utilizzati:

- `LabelEncoder`
- `OrdinalEncoder`
- `train_test_split`
- `GridSearchCV`
- `RandomForestClassifier`
- Metriche di valutazione (`Accuracy`, `F1-Score`, `Confusion Matrix`)

---

# 2. Scelta dell'Algoritmo

Per la classificazione è stato scelto il **Random Forest Classifier**.

Si tratta di un algoritmo di tipo **Ensemble Learning**, composto da numerosi alberi decisionali che collaborano tra loro per produrre una previsione finale.

La scelta è motivata dai seguenti vantaggi:

- Ottime prestazioni sui dataset tabulari.
- Capacità di apprendere relazioni non lineari.
- Ridotta sensibilità agli outlier.
- Maggiore resistenza all'overfitting rispetto a un singolo albero decisionale.
- Non richiede la normalizzazione delle feature numeriche.
- Permette di calcolare automaticamente l'importanza delle caratteristiche utilizzate dal modello, migliorandone l'interpretabilità.

---

# 3. Pipeline di Lavoro

L'intero progetto segue una pipeline ordinata e riproducibile.

## Step 1 – Caricamento dei dati

Il programma verifica la presenza dei file:

- `cats_dataset.csv`
- `test_set.csv`

In caso di assenza di uno dei file l'esecuzione viene interrotta mostrando un messaggio di errore esplicativo.

---

## Step 2 – Pulizia del Target

Prima dell'addestramento viene effettuata una pulizia della variabile target (`razza`).

In particolare vengono eliminate:

- righe con valori mancanti (`NaN`);
- righe contenenti la stringa `"nan"` come valore della razza.

Questa operazione garantisce che il modello venga addestrato solamente su dati validi.

---

## Step 3 – Pulizia del Dataset

Durante l'analisi preliminare sono state individuate diverse anomalie nei dati.

La procedura di pulizia esegue automaticamente le seguenti operazioni:

- conversione delle virgole decimali in punti (`4,5 → 4.5`);
- rimozione di caratteri non numerici e unità di misura dalle colonne numeriche mediante espressioni regolari (Regex);
- conversione delle colonne numeriche nel corretto formato;
- trasformazione dei valori non convertibili in `NaN`;
- eliminazione degli spazi bianchi superflui nelle colonne testuali.

Questa fase rende il dataset uniforme e pronto per il preprocessing.

---

## Step 4 – Suddivisione del Dataset

Il dataset viene suddiviso in:

- **80% Training Set**
- **20% Validation Set**

La suddivisione viene effettuata mediante `train_test_split`.

È stata inoltre utilizzata la stratificazione (`stratify=y`) per mantenere la stessa distribuzione delle razze nei due insiemi di dati.

Per garantire la riproducibilità dei risultati viene utilizzato un valore fisso di:

```python
random_state = 42
```

---

## Step 5 – Gestione dei Valori Mancanti

Dopo la suddivisione del dataset vengono gestiti i valori mancanti.

Le statistiche vengono calcolate esclusivamente sul **Training Set**, evitando fenomeni di **Data Leakage**.

Le strategie adottate sono:

### Variabili numeriche

Sostituzione dei valori mancanti con la media della colonna.

### Variabili categoriche

Sostituzione dei valori mancanti con la moda della colonna.

---

## Step 6 – Codifica delle Variabili Categoriche

Poiché gli algoritmi di Machine Learning lavorano esclusivamente con dati numerici, le variabili categoriche vengono convertite mediante due encoder differenti.

### LabelEncoder

Utilizzato esclusivamente per codificare la variabile target (`razza`).

### OrdinalEncoder

Utilizzato per convertire tutte le feature categoriche.

L'encoder viene addestrato solamente sul Training Set e successivamente applicato al Validation Set e al Test Set.

Inoltre è configurato per gestire automaticamente categorie mai viste durante l'addestramento senza interrompere l'esecuzione del programma.

---

## Step 7 – Ottimizzazione degli Iperparametri

Per migliorare le prestazioni del modello viene utilizzato **GridSearchCV**.

Vengono testate diverse combinazioni dei principali iperparametri della Random Forest:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Ogni configurazione viene valutata mediante **Cross Validation a 3 Fold**.

Al termine viene selezionata automaticamente la configurazione con le migliori prestazioni.

---

# 4. Analisi Esplorativa dei Dati (EDA)

Tutti i grafici vengono salvati automaticamente nella cartella:

```
grafici/
```

## distribuzione_razze.png

Grafico a barre che mostra il numero di gatti appartenenti a ciascuna razza.

Permette di verificare se il dataset è bilanciato oppure presenta classi sottorappresentate.

---

## matrice_correlazione.png

Heatmap che rappresenta la correlazione tra le variabili numeriche del dataset.

---

## relazione_interessante.png

Boxplot che confronta la distribuzione del peso delle diverse razze.

Gli outlier estremamente elevati vengono esclusi esclusivamente dal grafico per migliorarne la leggibilità.

---

## eta_vs_peso.png

Scatter Plot che rappresenta ogni gatto nel piano Età-Peso.

I punti sono colorati in base alla razza.

Permette di osservare eventuali cluster naturali tra le diverse classi.

---

## matrice_confusione.png

Heatmap della matrice di confusione.

Mostra il confronto tra la razza reale e quella predetta dal modello.

Consente di individuare le classi maggiormente confuse tra loro.

---

## importanza_feature.png

Grafico a barre orizzontali che rappresenta l'importanza delle feature calcolata automaticamente dalla Random Forest.

Permette di comprendere quali caratteristiche hanno influenzato maggiormente la classificazione.

---

# 5. Valutazione del Modello

Le prestazioni vengono valutate esclusivamente sul **Validation Set**, mai utilizzato durante l'addestramento.

Le metriche considerate sono:

## Accuracy

Percentuale complessiva di classificazioni corrette.

---

## F1-Score Macro

Media dell'F1-Score calcolato separatamente per ogni razza.

Questa metrica permette di valutare il modello in maniera equilibrata anche in presenza di classi con numerosità differenti.

I valori ottenuti vengono stampati automaticamente durante l'esecuzione del notebook.

---

# 6. File CSV Utilizzati

## cats_dataset.csv

Dataset di addestramento contenente le caratteristiche dei gatti e la relativa razza.

---

## test_set.csv

Dataset di test contenente solamente le caratteristiche dei gatti.

Viene utilizzato per generare le predizioni finali.

---

## predictions.csv

File prodotto automaticamente dal programma.

Contiene due colonne:

- **ID** → identificativo del gatto.
- **razza_prevista** → razza stimata dal modello.

---

# 7. Conclusioni

Il progetto implementa una pipeline completa di Machine Learning comprendente tutte le principali fasi di un processo di classificazione: pulizia dei dati, preprocessing, analisi esplorativa, addestramento del modello, ottimizzazione degli iperparametri, valutazione delle prestazioni e generazione delle predizioni finali.

Le scelte progettuali adottate, come l'utilizzo della **Random Forest**, la gestione accurata dei valori mancanti, la codifica delle variabili categoriche, la stratificazione del dataset e l'ottimizzazione tramite **GridSearchCV**, consentono di ottenere un modello robusto, riproducibile e adatto alla classificazione automatica delle razze feline.
