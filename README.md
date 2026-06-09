# Project 2 — Machine Learning on Single-Cell RNA-seq Data

> Laurea Magistrale in Data Science & AI — AI and Machine Learning
> Tutor: Chiara Napoli (chiara.napoli@uniroma1.it)
> Studentessa: Julieta Burstein

---

## Obiettivo (come richiesto dalla tutora)

> *"Partire dalla expression matrix fornita, utilizzare le label assegnate
> e addestrare un classificatore (es. Random Forest o LightGBM),
> valutandone le performance con metriche come accuracy e confusion matrix
> e identificando i geni più rilevanti."*

Il progetto è un problema di **classificazione multiclasse supervisionata**:
dato il profilo di espressione genica di una cellula, predire il suo
**tipo cellulare** (19 classi, fortemente sbilanciate).

---

## Dataset

Cellule da sangue periferico (PBMC) di **3 pazienti COVID-19** e **3 controlli sani**.
Preprocessing già applicato dal corso: QC filtering, normalizzazione, selezione HVG, batch correction.

**Dataset finale:** 5.645 cellule × 1.976 HVG, label via CellTypist.

| File | Descrizione | Shape |
|------|-------------|-------|
| `data_train/X_train.csv` | Matrice di training | (4516, 1976) |
| `data_test/X_test.csv`  | Matrice di test    | (1129, 1976) |
| `data_train/y_train.csv` | Label di training  | (4516,) |
| `data_test/y_test.csv`  | Label di test      | (1129,) |

> ⚠️ Lo split train/test è **predefinito** e non va rimescolato.
> ⚠️ Il test set non va mai usato per fit o tuning.

---

## Struttura del progetto

```
├── data_train/                        # CSV di training (X_train, y_train)
├── data_test/                         # CSV di test (X_test, y_test)
├── src/                               # Moduli Python riutilizzabili
│   ├── setup.py                       # load_pbmc() + path helpers
│   ├── data_loader.py                 # caricamento + shape + preview
│   ├── gene_groups.py                 # sottogruppi funzionali dei geni top
│   ├── models.py                      # factory: LR, SVM lineare, RF, LightGBM
│   ├── metrics.py                     # compute_metrics, confusion matrix, report
│   └── plots.py                       # plotting helpers (salvano in outputs/figures)
├── notebooks/                         # ENTREGA PRINCIPALE (5 notebook)
│   ├── FullPipelineTrigger.ipynb      # Esegue i notebook in sequenza (nbconvert --execute)
│   ├── data_loader.ipynb              # Anteprima: shape + head
│   ├── EDA.ipynb                      # Struttura, QC, geni discriminanti, gruppi, PCA
│   ├── Machine_learning_models.ipynb  # Dummy, LogReg, SVM, RF, LightGBM + CV + SHAP
│   └── SingleCell_Colab_Setup.ipynb   # Punto di ingresso per Google Colab
├── mio/                               # Materiale di supporto dello studente
├── requirements.txt
└── README.md
```

**Perché `src/` separato dai notebook.**
I moduli in `src/` contengono la logica riutilizzabile (load dati, factory dei modelli, metriche, plot). I notebook sono il "racconto" — importano da `src` e narrano passo per passo. Così evitiamo di duplicare codice tra notebook.

---

## Risultati principali (test set, 1129 cellule)

| Modello | Accuracy | Weighted F1 | Balanced Acc |
|---|---:|---:|---:|
| Logistic Regression            | 0.8884 | 0.8851 | 0.7011 |
| **SVM linear**                 | **0.8973** | **0.8995** | **0.7271** |
| Random Forest                  | 0.8760 | 0.8570 | 0.5979 |
| LightGBM (plain)               | 0.8937 | 0.8895 | 0.6124 |
| LightGBM (GridSearchCV tuned)  | 0.8902 | 0.8860 | 0.6063 |
| LightGBM + class_weight        | 0.8928 | 0.8888 | 0.6098 |
| LightGBM + **SMOTE**           | 0.8919 | 0.8858 | **0.6265** |
| LightGBM + **PCA denoise**     | 0.8981 | 0.8954 | 0.6956 |
| LightGBM + PCA latent (50 d)   | 0.8928 | 0.8892 | 0.6877 |
| LightGBM + MLP-AE latent (32 d) | 0.8432 | 0.8337 | 0.4733 |

> Le righe **Logistic Regression, SVM linear, Random Forest, LightGBM** sono
> riprodotte da `Machine_learning_models.ipynb`. Le righe **GridSearchCV tuned**,
> **SMOTE** e **PCA / MLP-AE** provengono da esperimenti ed estensioni precedenti,
> tenuti qui come riferimento per la discussione orale.

**Cosa leggere:**
- L'**accuracy** è alta (~0.89) un po' per tutti: significa che la classe dominante (CD8 T) si riconosce bene.
- La **balanced accuracy** fa vedere chi gestisce bene le classi rare. **SVM linear** è il migliore qui (0.727). I modelli ad alberi (RF, LightGBM) stanno sotto perché gli alberi fanno split su feature singole e le classi piccole non generano abbastanza split.
- **PCA denoise** migliora leggermente LightGBM (0.898 vs 0.894): pulire le 1976 feature a 50 componenti principali aiuta su rumore e dropout.
- L'**MLP autoencoder** in 32 dim perde segnale: 32 dimensioni sono poche per 19 classi.
- **SMOTE** non migliora l'accuracy ma **alza la balanced accuracy** (0.627 vs 0.610): confermata la sua utilità sulle classi rare.

---

## Geni più rilevanti (risposta esplicita alla tutora)

La tutora chiedeva: *"come sono stati identificati IL7R, PF4, CD3E dal modello?"*

I geni sono identificati in tre modi complementari, tutti presenti nei notebook:

1. **Feature importance nativa di Random Forest** (`Machine_learning_models.ipynb`) — riduzione media di impurità (Gini) quando il modello splitta su ciascun gene. Output salvato in `outputs/figures/ml_top_genes_randomforest.png`.

2. **Feature importance nativa di LightGBM** (`Machine_learning_models.ipynb`) — conteggio degli split che usano ciascun gene nel boosting. Output in `outputs/figures/ml_top_genes_lightgbm.png`.

3. **SHAP** (`Machine_learning_models.ipynb`, sezione interpretabilità) — usa `TreeExplainer` su LightGBM addestrato. SHAP attribuisce a ogni gene il suo contributo *per campione, per classe*: dice non solo che un gene è importante, ma **per quale classe e in che direzione**. Per la domanda della tutora sui geni **IL7R, PF4, CD3E**, l'analisi SHAP mostra che IL7R è rilevante per le classi T-cell, PF4 per Platelet, CD3E trasversale alle T-cell. Figura: `outputs/figures/ml_shap_summary_bar.png`.

La differenza importante: le feature importance native dicono "questo gene è usato molto dal modello"; SHAP dice **per quale classe e in che direzione**. Per la domanda della tutora la risposta corretta viene da SHAP.

---

## Come eseguire il codice

### A) In locale (consigliato)

```bash
# 1. crea e attiva un ambiente virtuale
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell)
# source .venv/bin/activate       # macOS / Linux

# 2. installa le dipendenze
pip install -r requirements.txt

# 3. apri i notebook
jupyter lab notebooks/
```

Dai notebook: apri `FullPipelineTrigger.ipynb`, che esegue in sequenza
`data_loader`, `EDA` e `Machine_learning_models` con `nbconvert --execute --inplace`
(gli output vengono salvati dentro ciascun notebook). In alternativa apri i tre
notebook singolarmente per vedere il dettaglio. In VS Code / Cursor seleziona il
kernel del `.venv`.

### B) Su Google Colab

Il repository è **privato** e i **CSV non sono su GitHub** (`.gitignore`).
Per eseguire tutto in Colab con un solo click, usare il notebook di setup:

**Link Colab diretto:**
https://colab.research.google.com/github/juliet4b/ML_SingleCell_Classification/blob/main/notebooks/SingleCell_Colab_Setup.ipynb

`notebooks/SingleCell_Colab_Setup.ipynb` fa automaticamente:
1. Monta Google Drive
2. Clona il repo privato (serve essere **collaborator** su GitHub)
3. Installa `requirements.txt`
4. Copia i 4 CSV da una cartella Drive in `data_train/` e `data_test/`
5. Esegue `notebooks/FullPipelineTrigger.ipynb` (con `FAST_MODE` attivo per Colab gratuito)

#### Passi per la studentessa (prima di condividere)

1. Push del progetto pulito sul repo privato `juliet4b/ML_SingleCell_Classification`.
2. Invitare la professora come **collaborator** (Settings → Collaborators → Read).
3. I 4 CSV sono nella cartella Drive condivisa [**data**](https://drive.google.com/drive/folders/1k0s5LaNHVm4Bb4TX4rPHkbb1gfz4CacI?usp=sharing)
   (`X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`).
4. Aprire il link Colab sopra → **Run all** (in Colab: aprire il link Drive → *Aggiungi collegamento a Drive* → percorso `/content/drive/MyDrive/data`).
5. Condividere con la professora: link Colab, link repo (o invito GitHub), [link cartella CSV su Drive](https://drive.google.com/drive/folders/1k0s5LaNHVm4Bb4TX4rPHkbb1gfz4CacI?usp=sharing).

#### Passi per la professora

1. Accettare l'invito GitHub (se usa `git clone` dal setup notebook).
2. Aprire la [cartella CSV su Drive](https://drive.google.com/drive/folders/1k0s5LaNHVm4Bb4TX4rPHkbb1gfz4CacI?usp=sharing) e cliccare *Aggiungi collegamento a Drive* (icona cartella con +).
3. Aprire il link Colab → **Run all** → autorizzare il montaggio di Drive quando richiesto.
4. Al termine, aprire dal pannello file:
   - `notebooks/data_loader.ipynb`
   - `notebooks/EDA.ipynb`
   - `notebooks/Machine_learning_models.ipynb`

**Tempi:** con `FAST_MODE = True` (default nel setup Colab) la pipeline richiede circa 15–30 minuti.
Se `Machine_learning_models.ipynb` va in timeout, eseguire solo quel notebook separatamente
oppure usare Colab Pro. In modalità veloce (`SC_FAST_MODE`) vengono ridotti gli alberi RF/LightGBM,
saltati il test di sensibilità RF e la cross-validation, e ridotto il campione SHAP.

**Fallback:** se `git clone` fallisce (autenticazione repo privato), il setup notebook
può copiare il progetto da una cartella Drive (`DRIVE_PROJECT_FOLDER` nella configurazione).

Il bootstrap dei notebook (`_find_root`) risale le cartelle cercando `src/setup.py`,
quindi funziona sia in locale sia in Colab senza modifiche, purché i CSV siano
in `data_train/` e `data_test/`.

---

## Flusso del progetto

```
CSV forniti (X_train, X_test, y_train, y_test)
        │
        ├── data_loader            : shape + anteprima
        ├── EDA :
        │     sanity-check (duplicati, leakage), distribuzione classi (sbilanciamento),
        │     geni discriminanti (1σ), heatmap z-score, gruppi di geni (biologia + varianza),
        │     boxplot marcatori, correlazione (co-espressione), PCA (varianza + scatter 2D)
        │
        └── Machine_learning_models :
              Dummy / LogReg / SVM lin / RandomForest / LightGBM
              evaluation su test (accuracy, Macro F1, balanced acc, CM),
              cross-validation, feature importance (Gini) + SHAP + IL7R/PF4/CD3E
```

Preprocessing (scaler): **solo** per LogReg/SVM (modelli distance-based). RF e LightGBM non hanno scaler (gli split su soglia sono invarianti alla scala).

No data leakage: ogni trasformazione (`StandardScaler`, `PCA`, `AE`, `SMOTE`) viene `.fit()` solo su `X_train`; `X_test` viene soltanto `.transform()`.
