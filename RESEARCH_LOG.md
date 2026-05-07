# RESEARCH LOG — NEPOOL TDA Regime Detection
*Avviato: 2026-03-26*

---

## Obiettivo

Identificare **regimi strutturali del mercato elettrico ISO New England** combinando:
- Prezzi orari LMP (Massachusetts Hub, 2021–2025, 43 794 osservazioni)
- Mix di generazione per combustibile (EIA, 8 fuel types)

Approccio: embedding con **MOMENT** (AutonLab/MOMENT-1-large, ICML 2024) → riduzione **PCA** (Marchenko-Pastur) → clustering **ToMATo** (TDA, persistenza topologica). K emerge dai dati, nessun sweep parametrico.

> **Nota**: Chronos-2 è stato sostituito da MOMENT (2026-03-26). Motivazione: step06 diagnostics
> mostravano embedding completamente opachi (R² negativo su tutte le feature di mercato,
> Mantel r ≈ 0.15, NN coherence ≈ 0.65–0.74). MOMENT è progettato per representation learning
> (non solo forecasting) e pre-addestrato su Time-Series Pile diversificato.

Baseline tradizionale: **Ward** (dendrogramma) per confronto (da implementare).

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: isone_dataset.parquet                                       │
│  43 794 ore  ·  2021–2025  ·  LMP (MA Hub) + EIA fuel mix (8 tipi) │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 01 — Preprocessing                  [step01_preprocessing.py] │
│                                                                     │
│  LMP → arcsinh → log_return → MSTL(24h+168h) → mstl_resid_lr      │
│  LMP → arcsinh → MSTL(24h+168h) → mstl_resid_arcsinh              │
│  Fuel mix → ILR(8→7) → MSTL(24h+168h) → mstl_resid_ilr_1..7      │
│  9 fit MSTL in parallelo (joblib)                                  │
│                                                                     │
│  Output: results/preprocessed.parquet                              │
│    lmp, arcsinh_lmp, log_return,                                   │
│    mstl_resid_lr, mstl_resid_arcsinh,                              │
│    ilr_1..7, mstl_resid_ilr_1..7                                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 02 — Embedding MOMENT              [step02_embeddings.py]     │
│                                                    [GPU]            │
│  MOMENT-1-large (~385M params), 512h window, 6h stride → ~7214 fin │
│                                                                     │
│  Design 2×2: {resid_lr, resid_arcsinh} × {±ILR detrended}         │
│    A: MOMENT(mstl_resid_lr)                     → 1024D            │
│    B: MOMENT(mstl_resid_lr) ‖ mstl_resid_ilr_* → 1031D            │
│    C: MOMENT(mstl_resid_arcsinh)                → 1024D            │
│    D: MOMENT(mstl_resid_arcsinh) ‖ mstl_resid_ilr_* → 1031D      │
│                                                                     │
│  ILR concatenato DOPO l'encoder (non come input channel)           │
│  MOMENT è univariato: 1 canale per volta                           │
│                                                                     │
│  Output: results/exp_{A..D}/embeddings.parquet                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 03 — PCA                            [step03_pca.py]           │
│                                                                     │
│  StandardScaler + PCA, n_comp via Marchenko-Pastur                 │
│  λ_max = (1 + √(p/n))²; eigenvalue > λ_max = segnale             │
│                                                                     │
│  Risultati: A→34D, B→35D, C→47D, D→48D (~89-90% varianza)         │
│                                                                     │
│  Output: results/exp_{X}/step03/pca_embeddings.parquet             │
│          pca_report.json, scree plot, cumvar, PCA 2D               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 04 — ToMATo clustering              [step04_tomato.py]        │
│                                                                     │
│  ToMATo (Chazal, Guibas, Oudot, Skraba 2013)                      │
│  Stima densità: logDTM, grafo k-NN (k=√n≈84)                      │
│  Persistence diagram → K via max-gap automatico nel barcode        │
│                                                                     │
│  Risultati: A→5, B→6, C→6, D→5 cluster                            │
│                                                                     │
│  Output: results/exp_{X}/step04/labels.parquet                     │
│          tomato_report.json, persistence diagram, barcode,         │
│          PCA 2D clusters, timeline                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 05 — Cluster Quality                [step05_cluster_quality.py│
│                                                                     │
│  A) Geometrica (nello spazio PCA):                                 │
│     Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn            │
│                                                                     │
│  B) Economica (dati di mercato):                                   │
│     η² (ANOVA su LMP), Tukey HSD, sojourn time,                   │
│     transition rate, Cramér's V (stagione + fascia oraria)         │
│                                                                     │
│  Output: results/exp_{X}/step05/quality_report.json                │
│          7 plot diagnostici                                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 06 — Embedding Diagnostics   [step06_embedding_diagnostics.py]│
│                                                                     │
│  A) Opacity — quanto sono interpretabili gli embedding?            │
│     Correlazione PC ↔ variabili fisiche                            │
│     Feature importance RF (embedding → LMP medio)                  │
│     Reconstruction R² (Ridge: embedding → feature manuali)         │
│                                                                     │
│  B) Transferability — Chronos-2 cattura struttura di mercato?      │
│     Mantel test (distanze embedding vs distanze feature)           │
│     NN coherence (k-NN embedding = k-NN feature?)                  │
│     Kendall τ sui ranking di vicinanza                             │
│                                                                     │
│  Output: results/exp_{X}/step06/diagnostics_report.json            │
│          5 plot diagnostici                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scelte metodologiche

Le scelte sono organizzate in tre categorie:
- **Motivazione teorica**: proprieta matematiche, letteratura, vincoli del problema
- **Scelta di design**: decisioni architetturali della pipeline
- **Test di sensibilita**: confronti empirici a supporto (non circolari — riportati separatamente nei Risultati)

---

### A. Preprocessing

#### A1. Trasformazione del prezzo: arcsinh(LMP)

**Motivazione teorica.** I prezzi LMP del mercato elettrico presentano tre proprieta che
escludono le trasformazioni standard:
- Prezzi negativi (surplus rinnovabili, nucleare non dispacciabile) -> log(x) non definito
- Prezzi zero (domanda minima) -> log(0) = -inf
- Code pesanti (spike a 500+ $/MWh) -> servono trasformazioni che comprimano le code

| Trasformazione | Negativi | Zeri | Compressione | Parametri |
|----------------|----------|------|-------------|-----------|
| log(x) | No | No | Si | 0 |
| log(x + c) | Si | Si | Si | 1 (c arbitrario) |
| Box-Cox | No | No | Si | 1 (lambda) |
| **arcsinh(x)** | **Si** | **Si** | **Si** | **0** |

arcsinh(x) = ln(x + sqrt(x^2 + 1)). Comprime come ln(2x) per |x| >> 1, lineare per x ~ 0.
Definita su R, monotona, senza parametri.

**Letteratura.** Ziel & Weron (2018) adottano arcsinh per prezzi day-ahead europei.
Uniejewski et al. (2019) confermano la superiorita su log+shift per mercati con prezzi negativi.

**Test di sensibilita** (vedi Risultati): arcsinh vs log_return testati con pipeline completa.

#### A2. Destagionalizzazione: MSTL (24h + 168h)

**Motivazione teorica.** La serie arcsinh(LMP) presenta forte stagionalita a due scale:
- Giornaliera (24h): ciclo domanda notte/giorno, ACF(24h) = 0.89
- Settimanale (168h): pattern feriale/festivo, ACF(168h) = 0.68

La stagionalita e' un pattern noto e deterministico. Se non rimossa, il clustering cattura
"estate vs inverno" anziche regimi strutturali di mercato. MSTL (Multiple Seasonal-Trend
decomposition using LOESS, De Livera et al. 2011) e' il metodo standard per decomposizioni
multi-stagionali di serie temporali.

Il residuo MSTL contiene il 12.2% della varianza totale di arcsinh(LMP). Questo 12% include
gli shock di prezzo, i cambi di regime, e la variabilita non spiegata dalla stagionalita.

**Stagionalita annuale (8760h).** ACF annuale di arcsinh(LMP) = 0.11 (debole). Per le coordinate
ILR del fuel mix, ACF annuale = 0.40 (forte — piu solare in estate, piu gas in inverno).
Il periodo annuale viene incluso nella destagionalizzazione ILR ma non in quella del prezzo.

**Test di sensibilita** (vedi Risultati): con MSTL vs senza MSTL testati con pipeline completa.

#### A4. Calibrazione context_len e stride (MOMENT)

**Analisi preliminare.** L'ACF del residuo MSTL scende sotto 0.05 a lag=31h (1.3 giorni).
La memoria lineare del residuo e' corta. Tuttavia MOMENT puo catturare dipendenze
non-lineari a scale piu lunghe.

**Griglia testata:** context_len x stride, pipeline completa per ogni combinazione.

| ctx (h) | stride (h) | N windows | n_comp | K (pre-merge) | silhouette | eta^2 |
|---------|-----------|-----------|--------|---------------|------------|-------|
| 72 | 6 | 7280 | 2 | 21 | 0.198 | 0.089 |
| 72 | 12 | 3640 | 2 | 15 | 0.226 | 0.087 |
| 72 | 24 | 1820 | 2 | 12 | 0.321 | 0.077 |
| 168 | 6 | 7248 | 2 | 23 | 0.234 | 0.093 |
| 168 | 12 | 3624 | 6 | 4 | 0.287 | 0.012 |
| 168 | 24 | 1812 | 2 | 15 | 0.315 | 0.058 |
| 336 | 6 | 7192 | 2 | 23 | 0.256 | 0.081 |
| 336 | 12 | 3596 | 5 | 11 | 0.258 | 0.070 |
| 336 | 24 | 1798 | 2 | 11 | 0.380 | 0.062 |
| **512** | **6** | **7132** | **6** | **8** | **0.285** | **0.161** |
| 512 | 12 | 3566 | 2 | 13 | 0.382 | 0.138 |
| 512 | 24 | 1784 | 2 | 12 | 0.427 | 0.112 |

**Risultato.** context_len=512h e stride=6h producono l'eta^2 piu alto (0.161 pre-merge,
~0.27 post-merge). Context piu lungo = migliore nonostante l'ACF lineare del residuo sia
trascurabile oltre 31h. MOMENT cattura dipendenze non-lineari a scala piu ampia.

La scelta context=512h coincide con il massimo input di MOMENT — il vincolo hardware
e' anche la configurazione ottimale.

#### A3. Fuel mix: ILR (Isometric Log-Ratio, SBP basis)

**Motivazione teorica.** Le 8 quote di generazione per fuel type sono dati composizionali
(sommano a 1). Le operazioni statistiche standard (media, varianza, distanza) non sono
appropriate nello spazio simplesso. La trasformazione ILR (Egozcue et al. 2003) proietta
le composizioni in R^7 dove le operazioni standard sono valide.

La base SBP (Sequential Binary Partition) usata:
- ilr_1: dispatchable (gas, nuc, coal, oil, other) vs variabile (hydro, wind, solar)
- ilr_2..7: partizioni successive all'interno di ogni gruppo

Sostituzione zeri: metodo moltiplicativo (Martin-Fernandez et al. 2003), delta=0.0001.

**Ruolo nella pipeline.** L'ILR non entra nell'embedding MOMENT (che e' univariato e processa
solo il prezzo). Le coordinate ILR destagionalizzate sono disponibili per la caratterizzazione
post-clustering dei regimi: dato un cluster, si ricostruisce il fuel mix medio tramite
inverse-ILR per interpretare il regime in termini di generazione.

**Test di sensibilita** (vedi Risultati): con ILR vs senza ILR nell'input testati.

---

### B. Embedding

#### B1. MOMENT (AutonLab/MOMENT-1-large, ICML 2024)

**Motivazione teorica.** Un foundation model per serie temporali trasforma una finestra di
512 osservazioni in un vettore di 1024 dimensioni (embedding) che codifica la struttura
temporale in modo data-driven, senza specificare a priori quali statistiche sono rilevanti.

MOMENT e' stato scelto rispetto a Chronos-2 (Amazon) per due ragioni:
1. **Pre-training objective**: MOMENT usa masked reconstruction (ricostruzione di patch
   mascherati), che produce rappresentazioni piu informative rispetto al forecasting-only
   di Chronos-2.
2. **Representation learning esplicito**: MOMENT offre un modo "embedding" nativo
   (task_name="embedding") progettato per produrre feature per downstream tasks.

**Vincoli.** MOMENT e' univariato: processa un canale alla volta. Non puo embeddare
congiuntamente prezzo e fuel mix. Questa e' una limitazione accettata.

**Test di sensibilita** (vedi Risultati): Chronos-2 vs MOMENT testati con metriche di
opacita e trasferibilita (Mantel test, NN coherence, R^2 di ricostruzione).

#### B2. Finestra 512h, stride 6h

**Motivazione teorica.** La finestra di 512h (~21 giorni) e' il massimo contesto accettato
da MOMENT. Copre almeno 3 cicli settimanali del residuo MSTL, catturando pattern
a scala settimanale e mensile. Lo stride di 6h produce ~7100 finestre su 5 anni,
con 98.8% di overlap tra finestre consecutive.

---

### C. Riduzione dimensionale

#### C1. Diffusion Maps (Coifman & Lafon 2006)

**Motivazione teorica.** L'embedding MOMENT produce vettori in R^1024. La densita nel
clustering (ToMATo) e' stimata via k-NN. In alta dimensione, le distanze euclidee diventano
poco discriminanti (curse of dimensionality). Serve una riduzione dimensionale che preservi
la geometria del manifold — in particolare la struttura della densita su cui ToMATo opera.

Le Diffusion Maps costruiscono un grafo di vicinanza pesato, calcolano la matrice di
transizione di un random walk sul grafo, e usano gli autovettori come coordinate. Le
coordinate diffusive preservano la distanza geodetica sul manifold e sono naturalmente
compatibili con la stima densita di ToMATo.

La normalizzazione alpha=1 (Laplace-Beltrami) rende le coordinate invarianti alla densita
locale — la geometria del manifold e' separata dalla densita dei punti.

**Alternativa scartata: PCA.** PCA e' lineare e non cattura la struttura non-lineare del
manifold. Testato: Diffusion Maps produce eta^2 3x superiore a PCA sullo stesso input
(vedi Risultati Fase 3).

**Alternativa scartata: UMAP.** UMAP altera la topologia (distorsione non-lineare delle
distanze) ed e' controproducente per ToMATo che opera sulla landscape di densita.

#### C2. Selezione automatica di n_components

**Scelta di design.** Il numero di coordinate diffusive e' selezionato tramite sweep su
n_comp = [2, 3, ..., 20]. Per ogni candidato, si calcolano le coordinate diffusive, si
esegue ToMATo (logDTM, k=sqrt(n), max_gap), e si valuta la silhouette dei cluster risultanti.
Si seleziona il n_comp con silhouette massima.

La silhouette e' un criterio puramente geometrico: misura se i cluster sono compatti e
separati nello spazio diffusivo, senza utilizzare informazioni sul prezzo o sul mercato.
La validita economica dei regimi viene verificata a posteriori (eta^2, Tukey HSD, LMP
per cluster), evitando circolarita.

---

### D. Clustering

#### D1. ToMATo (Chazal, Guibas, Oudot, Skraba 2013)

**Motivazione teorica.** ToMATo (Topological Mode Analysis Tool) identifica i modi della
densita stimata e fonde quelli con bassa persistenza topologica. Il numero di cluster K
emerge dal persistence diagram: un salto (gap) tra persistenze alte e basse indica la
separazione tra modi significativi e rumore.

Rispetto a GMM (usato nel repo precedente):
- Non assume cluster gaussiani
- K non richiede un range predefinito (il persistence diagram lo determina)
- Opera sulla landscape di densita, non sulla verosimiglianza parametrica

**Alternativa scartata: GMM.** Nel repo precedente, GMM+BIC selezionava K=20 (tetto del range)
per la maggior parte degli esperimenti — il BIC non trovava un ottimo interno.

**Alternativa scartata: HDBSCAN.** Testato sulla pipeline attuale: non trova cluster in alta
dimensione (298D con 1820 punti → K=0). Richiede una riduzione dimensionale aggressiva
(UMAP) che altera la topologia.

#### D2. Grid search su k (densita logDTM)

**Scelta di design.** Il parametro k del grafo k-NN per la stima densita logDTM e' selezionato
via grid search su k = [20, 40, 60, 80, 100, 150]. Per ogni k, ToMATo determina K
automaticamente via max_gap. Si seleziona il k con gap normalizzato massimo (gap_size /
median persistence).

#### D3. Tukey HSD merge post-clustering

**Motivazione teorica.** ToMATo puo produrre cluster topologicamente distinti ma
economicamente indistinguibili (stesso livello di prezzo). Il test di Tukey HSD (alpha=0.05)
verifica la significativita delle differenze di LMP tra ogni coppia di cluster. I cluster
non significativamente diversi vengono fusi iterativamente (partendo dalla coppia con
p-value piu alto) fino a che tutte le coppie sono significative.

Questo produce regimi che sono sia topologicamente fondati (emergono dalla densita)
sia economicamente distinti (LMP significativamente diverso tra ogni coppia).

**Nota.** Il merge usa il LMP come criterio a posteriori, non come input al clustering.
I cluster sono formati nello spazio diffusivo dell'embedding MOMENT, poi validati e
consolidati tramite il prezzo. Non c'e' circolarita: la formazione dei cluster non
dipende dal prezzo.

---

## Risultati sperimentali

### Fase 1: Chronos-2 + PCA (2026-03-26)

Design 2x2: {log_return, arcsinh} x {senza ILR, con ILR detrended}, finestre sliding 720h, stride 6h.

#### PCA (Marchenko-Pastur)

| Exp | Encoder | Input D | PC | Varianza | lambda_max |
|-----|---------|---------|-----|----------|------------|
| A | Chronos-2 | 768 | 34 | 90.2% | 1.768 |
| B | Chronos-2 | 775 | 35 | 89.8% | 1.772 |
| C | Chronos-2 | 768 | 47 | 88.8% | 1.768 |
| D | Chronos-2 | 775 | 48 | 88.5% | 1.772 |

#### ToMATo + Quality + Diagnostics

| Metrica | A | B | C | D |
|---------|---|---|---|---|
| K (ToMATo) | 5 | 6 | 6 | 5 |
| Silhouette | -0.044 | -0.018 | -0.003 | -0.022 |
| eta^2 | 0.005 | 0.017 | **0.039** | 0.303* |
| Trans. rate | 38% | 51% | **25%** | 47% |
| Sojourn | 16h | 12h | **24h** | 13h |
| Cramer V season | 0.17 | 0.22 | **0.37** | 0.69* |
| Tukey coppie | 50% | 52% | **80%** | 29% |
| RF R^2 (LMP) | -0.92 | -0.87 | **-0.81** | -0.82 |
| Mantel r | 0.13 | 0.15 | **0.16** | 0.17 |
| PC0 max |r| | 0.40 | 0.40 | **0.61** | 0.61 |

*D con Chronos-2 era rotto: grid search aveva selezionato KDE con 102 micro-cluster (artefatto).

**Conclusione Fase 1**: Chronos-2 opaco (R^2 negativo su tutte le feature). C miglior candidato
ma eta^2=0.039 e' debole. Embedding non cattura struttura economica del mercato.

---

### Fase 2: MOMENT + PCA (2026-03-26)

Sostituzione Chronos-2 con MOMENT (AutonLab/MOMENT-1-large). Context ridotto a 512h (max MOMENT).
ToMATo fixato: logDTM, k=sqrt(n), max_gap (no grid search).

#### MOMENT vs Chronos-2 — exp C (arcsinh, no ILR)

| Metrica | Chronos-2 C | MOMENT C |
|---------|-------------|----------|
| K | 6 | **2** |
| Silhouette | -0.003 | **+0.065** |
| eta^2 | 0.039 | 0.001 |
| PC0 |r| | 0.61 (skew) | **0.79 (acf_1h)** |
| R^2 acf_1h | 0.545 | **0.640** |
| R^2 skew | 0.427 | **0.507** |
| Mantel r | 0.156 | 0.115 |

MOMENT piu interpretabile ma K=2 senza significato economico (eta^2~0, LMP quasi identico tra cluster).

#### Tutti e 4 gli esperimenti con MOMENT + PCA

| Metrica | A | B | C | **D** |
|---------|---|---|---|-------|
| K | 2 | 2 | 2 | **5** |
| Silhouette | +0.024 | +0.056 | +0.065 | **+0.053** |
| eta^2 | 0.000 | 0.004 | 0.001 | **0.040** |
| Tukey coppie | 0/1 | 1/1 | 1/1 | **7/10 (70%)** |
| Trans. rate | 18% | 40% | 9% | **25%** |
| Sojourn | 33h | 15h | 65h | **24h** |
| Mantel r | 0.09 | 0.13 | 0.12 | **0.15** |

**Conclusione Fase 2**: Solo D (arcsinh + ILR) produce >2 cluster. ILR e' essenziale.
A/B (log_return) non producono struttura. MOMENT piu interpretabile di Chronos-2 ma
PCA lineare non cattura la geometria del manifold.

---

### Fase 3: MOMENT + Diffusion Maps (2026-03-26/27)

Sostituzione PCA con Diffusion Maps (Coifman & Lafon 2006) per preservare geometria del manifold.

#### Calibrazione n_components (exp D, Diffusion Maps)

| n_comp | K | Silhouette | eta^2 | Tukey coppie | Trans. rate | Sojourn |
|--------|---|------------|-------|--------------|-------------|---------|
| 3 | 21 | **0.309** | 0.116 | 63% | 50% | 12h |
| **10** | 18 | **0.182** | **0.144** | **69%** | **28%** | **22h** |
| 15 | 19 | 0.114 | 0.162 | 71% | 31% | 19h |
| 20 | 17 | 0.065 | 0.131 | 63% | 22% | 28h |
| PCA 56D | 5 | 0.053 | 0.040 | 70% | 25% | 24h |

Diffusion Maps 10D: eta^2 3.5x superiore a PCA, silhouette 3.5x superiore.

#### Confronto finale: PCA vs Diffusion Maps (exp D, n_comp=10)

| Metrica | PCA 56D | Diff. Maps 10D |
|---------|---------|----------------|
| K (dopo Tukey) | 5 | **10** |
| Silhouette | 0.053 | **0.188** |
| Davies-Bouldin | 2.47 | **0.95** |
| eta^2 | 0.040 | **0.127** |
| LMP range | 47-77 $/MWh | **27-148 $/MWh** |
| Tukey 100% | no (70%) | **si (100%)** |
| Trans. rate | 25% | 22% |
| Sojourn | 24h | 27h |
| Mantel r | 0.15 | **0.38** |
| Kendall tau | 0.10 | **0.19** |

#### ToMATo grid search + Tukey HSD merge (exp D, Diff. Maps 10D)

ToMATo (logDTM, k=40, max_gap) -> K=44 cluster raw.
Tukey HSD merge (alpha=0.05): 44 -> **10 regimi**, tutti significativamente distinti.

| Regime | LMP medio ($/MWh) | n (giorni) | Interpretazione provvisoria |
|--------|--------------------|------------|----------------------------|
| R9 | 148 | 111 | Spike/stress estremo |
| R6 | 124 | 167 | Spike moderato |
| R7 | 103 | 59 | Alto |
| R4 | 85 | 604 | Sopra media |
| R8 | 75 | 232 | Medio-alto |
| R2 | 60 | 912 | Medio |
| R0 | 55 | 1472 | Normale |
| R1 | 49 | 1540 | Sotto media |
| R5 | 43 | 911 | Basso |
| R3 | 35 | 1124 | Molto basso |

**Conclusione Fase 3**: Diffusion Maps + Tukey merge producono regimi economicamente
distinti con eta^2=0.25 (6x rispetto a PCA). Il manifold non-lineare contiene struttura
che PCA non cattura.

---

### Selezione automatica n_components — tentativi

| Criterio | Risultato | Problema |
|----------|-----------|----------|
| Spectral gap (max ratio) | 3 componenti (ratio 1.04) | Debolissimo, troppo poche D |
| Participation ratio | 59 componenti | Eigenvalues decadono lentamente, PR troppo alto |
| Cumulative eigenvalue | Non testato | Da esplorare |

Spectral gap scelto come criterio automatico. Dava 3D — da calibrare con context.

---

## Decisioni di redesign (2026-03-27)

### Perche finestre giornaliere (24h) e non sliding (512h/6h stride)

1. **Mercato day-ahead**: il clearing avviene su base giornaliera. I regimi di prezzo
   sono fenomeni giornalieri (giorno di picco vs giorno base vs giorno surplus).
2. **Informazione oraria non persa**: MOMENT riceve il profilo 24h completo e lo codifica.
   La forma intra-giornaliera (picchi, valli, duck curve) e' nell'embedding.
3. **Regimi piu stabili**: con stride 6h i regimi cambiavano ogni 12-16h (instabili).
   Con finestre giornaliere un regime dura almeno 1 giorno — piu interpretabile.
4. **Meno ridondanza**: 512h window con 6h stride = 98.8% overlap tra finestre consecutive.
   Finestre giornaliere = 0% overlap, punti indipendenti.
5. **Evidenza**: sojourn con sliding windows era 12-27h; la granularita minima era comunque ~1 giorno.

### Perche 8 canali MOMENT indipendenti (non 1 canale + ILR post-hoc)

1. **Embedding del fuel mix**: MOMENT codifica il pattern intra-giornaliero del fuel mix,
   non solo il valore puntuale. Una giornata con rampa solare (ILR che cambia durante il giorno)
   produce un embedding diverso da una giornata con solare costante.
2. **Evidenza**: con ILR concatenato post-hoc (fase 2), solo exp D produceva >2 cluster.
   L'ILR raw a 7D non aveva abbastanza peso rispetto a 1024D di embedding LMP.
3. **Bilanciamento pesi**: PCA per blocco (LMP->20D, ILR->30D) bilancia l'importanza relativa
   di prezzo e fuel mix nel clustering.

### Perche cross-channel cosine similarity

1. MOMENT e' univariato — non cattura relazioni tra canali.
2. Le 28 dimensioni di cosine similarity codificano: coupling LMP-gas, anticorrelazione wind-LMP,
   co-movimento nucleare-baseload, etc.
3. Queste relazioni sono diverse per regime: in un regime di surplus eolico wind e LMP sono
   anticorrelati; in un regime termico gas e LMP sono correlati.

### Perche Diffusion Maps e non PCA

| | PCA | Diffusion Maps |
|-|-----|----------------|
| Tipo | Lineare | Non-lineare |
| Preserva | Varianza | Geometria del manifold |
| Compatibilita TDA | Bassa (topologia distorta) | Alta (preserva densita) |
| eta^2 (exp D) | 0.040 | **0.127** (3x) |
| Silhouette (exp D) | 0.053 | **0.188** (3.5x) |
| Mantel r (exp D) | 0.15 | **0.38** (2.5x) |

### Perche Tukey HSD merge post-clustering

1. ToMATo trova tutti i modi topologici della densita (anche quelli con bassa significativita economica).
2. Tukey HSD fonde i cluster che non si distinguono su LMP (p > 0.05).
3. Il risultato e' **data-driven su due livelli**: topologia (quanti modi) + economia (quanti distinti su prezzo).
4. **Evidenza**: K=44 (ToMATo raw) -> K=10 (dopo Tukey). 100% coppie significative.

### Perche arcsinh(x / MAD) e non arcsinh(x)

arcsinh(x) comprime le code ma non standardizza la scala. Dividere per MAD normalizza
la dispersione prima della compressione, producendo un segnale piu uniforme per MOMENT.

### Perche ILR Helmert e non SBP

SBP (Sequential Binary Partition) richiede una partizione arbitraria dei fuel types (es.
"dispatchable vs variable"). Helmert e' una base ortogonale standard che non richiede
scelte a priori. E' piu difendibile metodologicamente: nessun reviewer puo obiettare
sulla partizione perche non ce n'e' una.

### Perche MSTL condizionale (diagnostica prima)

Non tutti i canali ILR hanno stagionalita significativa. Se MSTL rimuove varianza
che non e' stagionale (>80% residua, ACF debole), sta facendo danni. La diagnostica
permette di saltare la destagionalizzazione per quei canali.

---

## Fase 4: Redesign completo della pipeline (2026-03-27)

Riscrittura basata su revisione architetturale. Singola configurazione (no piu A/B/C/D).

### Step 01 — Preprocessing (risultati run 2026-03-27)

| Parametro | Valore |
|-----------|--------|
| Input rows | 43,814 |
| Output rows (hourly) | 43,795 |
| Output days (windows) | **1,819** |
| Giorni incompleti droppati | 7 |
| Canali totali | 8 (1 LMP + 7 ILR) |

#### MSTL diagnostics — tutti i canali detrended

| Canale | Var residua | ACF 24h | ACF 168h | Decisione |
|--------|-------------|---------|----------|-----------|
| arcsinh_lmp | 12.3% | 0.9996 | 0.9991 | DETREND |
| ilr_1 (gas/nuc/coal vs hydro/wind/sol) | 11.7% | 0.9997 | 0.9992 | DETREND |
| ilr_2 (fossil vs non-fossil disp.) | 8.1% | 0.9997 | 0.9984 | DETREND |
| ilr_3 (gas vs coal+oil) | 55.9% | 0.9981 | 0.9978 | DETREND |
| ilr_4 (coal vs oil) | 6.8% | 1.0001 | 0.9959 | DETREND |
| ilr_5 (nuclear vs other) | 22.0% | 0.9988 | 0.9950 | DETREND |
| ilr_6 (hydro vs wind+solar) | 40.7% | 0.9967 | 0.9930 | DETREND |
| ilr_7 (solar vs wind) | 19.8% | 1.0000 | 0.9965 | DETREND |

Nessun canale ha >80% varianza residua con ACF debole. La soglia diagnostica non e' stata
attivata — tutti i canali hanno stagionalita significativa da rimuovere.

Nota: ilr_3 (gas vs coal+oil) ha la varianza residua piu alta (55.9%) — la distinzione
gas/fossili solidi e' meno stagionale. ilr_6 (hydro vs intermittenti) segue a 40.7%.

#### Robust z-score post-MSTL

| Canale | Mediana | Std (MAD-scaled) |
|--------|---------|------------------|
| mstl_resid_arcsinh_lmp | 0.0 | 1.92 |
| mstl_resid_ilr_1 | 0.0 | 1.77 |
| mstl_resid_ilr_2 | 0.0 | 1.78 |
| mstl_resid_ilr_3 | 0.0 | 1.69 |
| mstl_resid_ilr_4 | 0.0 | 1.89 |
| mstl_resid_ilr_5 | 0.0 | 2.45 |
| mstl_resid_ilr_6 | 0.0 | 2.52 |
| mstl_resid_ilr_7 | 0.0 | 1.81 |

Std MAD-scaled non e' esattamente 1.0 perche MAD e std differiscono per distribuzioni
non-gaussiane. Il rapporto std/MAD ~ 1.7-2.5 indica code pesanti (atteso per dati di
mercato elettrico).

#### arcsinh(LMP / MAD)

MAD(LMP) = 15.08 $/MWh. Il fattore 1/MAD normalizza la dispersione prima della
compressione arcsinh, producendo una distribuzione piu uniforme per MOMENT.
Confronto: arcsinh(41/15.08) = 1.36 vs arcsinh(41) = 4.40 — range piu compresso.

### Step 02 — MOMENT Multi-Channel Embedding (risultati run 2026-03-27)

8 canali indipendenti, ciascuno 24h zero-padded a 512 con input_mask.

| Canale | Embedding dim | Norma media L2 | Tempo (GPU) |
|--------|---------------|-----------------|-------------|
| arcsinh_lmp | 1024 | 2.94 | 12.9s |
| ilr_1 | 1024 | 2.89 | 12.7s |
| ilr_2 | 1024 | 3.03 | 12.7s |
| ilr_3 | 1024 | 2.84 | 12.8s |
| ilr_4 | 1024 | 2.88 | 12.9s |
| ilr_5 | 1024 | 2.98 | 13.0s |
| ilr_6 | 1024 | 2.96 | 13.0s |
| ilr_7 | 1024 | 2.94 | 13.0s |

Totale: 1819 giorni x **8192D**. Norme uniformi tra canali (~2.84-3.03) -- MOMENT tratta
tutti i canali in modo comparabile nonostante la diversa natura dei segnali. Nessun canale
degenere (norme >0, varianza >0 su tutte le dimensioni).

Test critico: MOMENT con 24h padded a 512 + input_mask produce embedding non-degenere
(std=0.036 nel test dummy, confermato con dati reali: varianza media per dim > 0).

### Step 03 — Dimensionality Reduction (risultati run 2026-03-27)

| Stadio | Input | Output | Note |
|--------|-------|--------|------|
| Block PCA LMP | 1024D | 20D | 64.1% varianza |
| Block PCA ILR | 7168D (7x1024) | 30D | 34.6% varianza |
| Cross-channel cosine | 8 canali | 28D (C(8,2) paia) | media cos=0.82 |
| **Concatenazione** | — | **78D** | 50 PCA + 28 cosine |
| Diffusion Maps | 78D | **5D** | spectral gap automatico |

**Spectral gap automatico seleziona 5 componenti** — molto piu netto che nell'architettura
precedente (dove su 1031D dava 3 con ratio 1.04). La riduzione a 78D via block PCA + cosine
crea uno spazio piu strutturato per le Diffusion Maps.

Top 10 eigenvalues: [0.852, 0.784, 0.722, 0.720, 0.689, 0.654, 0.631, 0.622, 0.612, 0.611]
Gap principale tra componente 5 (0.689) e 6 (0.654) — ratio 1.05.

Cross-channel cosine media = 0.82 indica che i canali sono abbastanza correlati (MOMENT
produce embedding simili per segnali diversi). La varianza (std=0.05) e' dove sta
l'informazione discriminante — i giorni dove i canali divergono.

### Difetti identificati e fix (2026-03-27)

| # | Difetto | Evidenza | Fix |
|---|---------|----------|-----|
| 1 | 24h su 512 slot = 95% padding | MOMENT pre-addestrato su serie piene | Context 168h (1 settimana), pad a 512: 33% reale |
| 2 | Canali indipendenti, cosine similarity insufficiente | Solo 28D, cattura solo se embedding simili non come interagiscono | Aggiungere L2 distance + correlazione: 3x28=84D |
| 3 | Block PCA 50D butta 65% varianza ILR | pca_ilr_dim=30 forzato cattura solo 34.6% | Marchenko-Pastur automatico per blocco |
| 4 | Diffusion Maps collassa 78D->5D con gap debole | Spectral gap ratio 1.05, K=4 poi merge a K=2 | Eliminare DiffMaps, HDBSCAN diretto su feature space |
| 5 | Tukey merge su solo LMP medio giornaliero | Appiattisce spike: giorno con spike 500+23h a 30 = media 50 | Tukey multivariato: LMP medio + P95 + volatilita |

### Caratterizzazione dei 9 regimi (exp C, pipeline finale 2026-03-28)

Pipeline: arcsinh(LMP) -> MSTL(24h+168h) -> MOMENT(512h, 6h stride) -> DiffMaps(6D, silhouette) -> ToMATo -> Tukey merge -> K=9

#### Regimi identificati

| Regime | LMP medio | n (%) | Stagione dominante | Gas% | Oil% | Interpretazione |
|--------|-----------|-------|-------------------|------|------|-----------------|
| R0 | 29 $/MWh | 461 (6.5%) | Primavera+Autunno | 54.3 | 0.1 | Off-peak primaverile |
| R1 | 39 $/MWh | 1272 (17.8%) | Autunno+Primavera | 56.0 | 0.2 | Baseload mite |
| R2 | 48 $/MWh | 2006 (28.1%) | Distribuito | 56.2 | 0.2 | Normale (dominante) |
| R3 | 55 $/MWh | 1251 (17.5%) | Estate+Primavera | 56.9 | 0.5 | Domanda moderata |
| R4 | 64 $/MWh | 946 (13.3%) | Inverno | 54.4 | 0.6 | Domanda invernale |
| R5 | 80 $/MWh | 671 (9.4%) | 72% Inverno | 49.6 | 2.1 | Stress invernale |
| R6 | 101 $/MWh | 252 (3.5%) | Inverno+Estate | 56.0 | 2.3 | Spike moderato |
| R7 | 120 $/MWh | 148 (2.1%) | 97% Inverno | 49.0 | 3.2 | Spike invernale |
| R8 | 147 $/MWh | 125 (1.8%) | 100% Inverno | 50.6 | 4.4 | Spike estremo |

#### Pattern emergenti

1. **Gradiente invernale**: R5-R8 sono progressivamente invernali (72% -> 100%)
   con LMP crescente. I regimi di spike sono fenomeni invernali (cold snap, heating demand).

2. **Olio come indicatore di stress**: la quota olio cresce da 0.1% (R0) a 4.4% (R8).
   L'olio e' il combustibile di backup piu costoso — si attiva solo in emergenza quando
   gas e nucleare non bastano. Correla con il livello di stress del sistema.

3. **Gas inversamente correlato allo stress**: gas scende nei regimi di spike (57% in R3
   vs 49% in R7). In condizioni di scarsita il gas marginale e' gia al massimo e entrano
   combustibili piu costosi (olio, coal).

4. **R0 primaverile/pomeridiano**: 41% ore 12-18, 53% primavera. Coincide con
   produzione solare e domanda bassa — il regime "green/off-peak".

5. **R6 bimodale**: 58% inverno + 33% estate. Cattura sia cold snap sia heat wave —
   entrambi causano spike moderati per eccesso di domanda (riscaldamento e raffrescamento).

#### Matrice di transizione

Probabilita P[i,j] di passare dal regime i al regime j (finestre consecutive, stride 6h).

|     | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|-----|----|----|----|----|----|----|----|----|----|
| R0 | **0.36** | 0.46 | 0.08 | 0.07 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| R1 | 0.11 | **0.49** | 0.21 | 0.13 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| R2 | 0.05 | 0.12 | **0.56** | 0.16 | 0.09 | 0.01 | 0.01 | 0.00 | 0.00 |
| R3 | 0.04 | 0.09 | 0.29 | **0.42** | 0.11 | 0.04 | 0.02 | 0.00 | 0.00 |
| R4 | 0.01 | 0.09 | 0.18 | 0.13 | **0.51** | 0.06 | 0.02 | 0.00 | 0.00 |
| R5 | 0.00 | 0.00 | 0.03 | 0.09 | 0.08 | **0.74** | 0.04 | 0.01 | 0.00 |
| R6 | 0.00 | 0.01 | 0.07 | 0.10 | 0.06 | 0.15 | **0.58** | 0.01 | 0.02 |
| R7 | 0.00 | 0.02 | 0.05 | 0.01 | 0.00 | 0.04 | 0.01 | **0.85** | 0.03 |
| R8 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.04 | 0.05 | **0.89** |

La matrice e' fortemente diagonale: i regimi sono persistenti. Le transizioni avvengono
prevalentemente tra regimi adiacenti (R0<->R1, R1<->R2, etc.) — il mercato transita
gradualmente tra livelli di prezzo, non salta da R0 a R8.

I regimi di spike (R7, R8) hanno le auto-transizioni piu alte (0.85, 0.89) — gli eventi
di stress persistono una volta iniziati. Durata media: R7=39h, R8=54h.

#### Persistenza per regime

| Regime | Durata media | Mediana | Max |
|--------|-------------|---------|-----|
| R0 | 9h | 6h | 102h |
| R1 | 12h | 6h | 246h |
| R2 | 14h | 12h | 336h |
| R3 | 10h | 6h | 342h |
| R4 | 12h | 6h | 456h |
| R5 | 23h | 6h | 636h |
| R6 | 14h | 6h | 258h |
| R7 | 39h | 12h | 216h |
| R8 | 54h | 6h | 522h |

Nota: le durate sono in finestre x 6h (stride). L'overlap del 98.8% tra finestre
consecutive fa si che i cambi di regime siano graduali.

---

## Test di sensibilita (pipeline finale, 2026-03-28)

Tutti i test usano la stessa pipeline: MOMENT(512h, 6h) -> Diffusion Maps (n_comp auto
via silhouette, sweep 2-20) -> ToMATo (grid logDTM) -> Tukey HSD merge (alpha=0.05).
I test variano un fattore alla volta rispetto alla configurazione base (exp C).

### S1. Trasformazione del prezzo: arcsinh vs log_return

Fattore testato: input a MOMENT. arcsinh preserva il livello, log_return lo rimuove.

| | Exp C (arcsinh) | Exp A (log_return) |
|-|-----------------|-------------------|
| Input | mstl_resid_arcsinh | mstl_resid_lr |
| n_comp (auto sil) | 6 | 2 |
| K (pre-merge) | 8 | 13 |
| Silhouette | 0.285 | 0.176 |
| K (post-merge) | **9** | (non calcolato, eta^2~0.02) |
| eta^2 (sweep) | **0.182** | 0.023 |

arcsinh vince nettamente. log_return produce cluster geometricamente separati (K=13)
ma economicamente vuoti (eta^2=0.02 — i regimi non distinguono livelli di prezzo).
Questo conferma che i regimi di mercato sono definiti dal livello, non dalla variazione.

### S2. Destagionalizzazione: con MSTL vs senza

Fattore testato: se applicare MSTL(24h+168h) prima di MOMENT.

| | Exp C (con MSTL) | Exp E (senza MSTL) |
|-|-----------------|-------------------|
| Input | mstl_resid_arcsinh | arcsinh_lmp |
| Var residua / totale | 12.2% | 100% |
| n_comp (auto) | 6 | 16 |
| K (post-merge) | **9** | 4 |
| eta^2 (sweep) | **0.182** | 0.189 |
| eta^2 (post-merge) | **0.267** | (basso, K=4 grossolani) |

Senza MSTL, MOMENT codifica la stagionalita dominante e produce pochi regimi grossolani
(estate/inverno). Con MSTL, la stagionalita e' rimossa e MOMENT cattura la struttura fine
(spike, stress, livelli di prezzo). L'eta^2 nel sweep e' simile (~0.18) ma il risultato
post-merge e' molto superiore con MSTL (9 regimi distinti vs 4 grossolani).

### S3. Fuel mix: con ILR vs senza ILR nell'embedding

Fattore testato: se concatenare 7 coordinate ILR destagionalizzate all'embedding MOMENT.

| | Exp C (no ILR, 1024D) | Exp D (con ILR, 1031D) |
|-|----------------------|----------------------|
| n_comp (auto) | 6 | 6 |
| eta^2 (sweep) | 0.182 | 0.184 |
| K (post-merge) | 9 | 9 |
| eta^2 (post-merge) | **0.267** | 0.266 |

Nessuna differenza significativa. 7 scalari ILR su 1031 dimensioni pesano lo 0.7% —
le Diffusion Maps li ignorano. L'ILR e' piu utile nella caratterizzazione post-clustering
(dove il fuel mix spiega i regimi) che come input al clustering.

### S4. Context e stride di MOMENT

Fattore testato: lunghezza finestra e passo di campionamento.

| ctx (h) | stride (h) | N | n_comp | K (pre) | sil | eta^2 (pre) |
|---------|-----------|------|--------|---------|-----|-------------|
| 72 | 6 | 7280 | 2 | 21 | 0.198 | 0.089 |
| 168 | 6 | 7248 | 2 | 23 | 0.234 | 0.093 |
| 336 | 6 | 7192 | 2 | 23 | 0.256 | 0.081 |
| **512** | **6** | **7132** | **6** | **8** | **0.285** | **0.161** |
| 512 | 12 | 3566 | 2 | 13 | 0.382 | 0.138 |
| 512 | 24 | 1784 | 2 | 12 | 0.427 | 0.112 |

ctx=512h e stride=6h producono l'eta^2 piu alto. Context piu lungo e' migliore nonostante
l'ACF del residuo sia trascurabile oltre 31h — MOMENT cattura dipendenze non-lineari
a scala piu ampia che l'ACF lineare non rileva.

### S5. Feature engineering vs MOMENT (controprova, 2026-03-28)

19 feature statistiche calcolate sulla stessa finestra 512h: mean, std, skew, kurt,
min, max, range, median, p5, p95, iqr, acf_1h, acf_6h, acf_24h, acf_168h, vol_24h,
lmp_mean, lmp_p95, lmp_std.

Stessa pipeline downstream: Diffusion Maps (silhouette, sweep 2-20) -> ToMATo -> Tukey merge.

| Metrica | MOMENT zero-shot | MOMENT fine-tuned | **Feature eng.** |
|---------|-----------------|-------------------|------------------|
| Input dim | 1024D | 1024D | **19D** |
| eta^2 (sweep best) | 0.182 | 0.185 | **0.413** |
| silhouette (sweep) | 0.285 | 0.316 | **0.402** |
| n_comp (auto) | 6 | 2 | 7 |
| K (post-merge) | 9 | (in corso) | **10** |
| eta^2 (post-merge) | 0.267 | — | **0.409** |
| Transition rate | 47% | — | **3%** |
| Sojourn | 13h | — | **197h (8 giorni)** |
| Tukey 100% | si | — | **si** |
| Tempo embedding | 65s GPU | 270s GPU+FT | **4s CPU** |

**Conclusione.** Le feature hand-crafted domain-specific producono regimi con eta^2
quasi doppio, stabilita temporale 15x superiore, e nessuna dipendenza da GPU.
Il fine-tuning di MOMENT su dati ISONE non migliora rispetto allo zero-shot
(eta^2 sweep 0.185 vs 0.182).

Per la detection di regimi in mercati elettrici, la conoscenza del dominio incorporata
in feature statistiche classiche e' piu informativa delle rappresentazioni generiche
apprese da foundation model pre-addestrati su serie temporali eterogenee.

---

## Confronto 3 approcci — tabella riassuntiva (2026-03-29)

| Metrica | A: MOMENT | B: FE | C: COMBO |
|---------|-----------|-------|----------|
| Input | 1024D | 19D | 74D (19+55) |
| n_comp (auto sil) | 6 | 7 | 9 |
| K post-merge | 9 | 10 | 7 |
| eta^2 | 0.267 | **0.409** | 0.271 |
| Silhouette | -0.018 | 0.023 | **0.057** |
| Tukey 100% | si | si | si |
| Trans. rate | 47% | **3%** | 8% |
| Sojourn | 13h | **197h** | 79h |
| ARI bootstrap | **0.79** | 0.68 | 0.74 |

Nota: nell'approccio C, MOMENT 1024D e' compresso a 55D via PCA Marchenko-Pastur
prima della concatenazione con le 19 feature. Bilanciamento interno, non step separato.

ILR (fuel mix) non entra nel clustering per nessun approccio. Usato solo nella
caratterizzazione post-clustering per dare interpretazione economica ai regimi.
Test di sensibilita (approccio D=COMBO+ILR, eta^2=0.245) conferma che l'ILR
non migliora la separazione economica quando aggiunto all'input del clustering.

---

## Outline del paper

```
1. INTRODUCTION
   - Electricity markets: regimi di prezzo (spike, baseload, stress)
   - Problema: identificazione non supervisionata, K non noto a priori
   - Gap: foundation models vs feature classiche per regime detection
   - Contributo: confronto sistematico di 4 approcci su ISONE 2021-2025

2. DATA
   - ISONE LMP orario (Massachusetts Hub) + EIA fuel mix (8 tipi)
   - 43,794 ore, 5 anni
   - Proprieta: prezzi negativi, code pesanti, doppia stagionalita, dati composizionali

3. PREPROCESSING
   - arcsinh(LMP): stabilizzazione varianza, gestisce negativi [Ziel & Weron 2018]
   - MSTL(24h+168h): rimuove stagionalita nota, tiene struttura residua
   - ILR (base SBP): dati composizionali fuel -> R^7 [Egozcue 2003]
   - Finestre sliding: 512h contesto, 6h stride -> 7132 finestre

4. METHODS
   4.1 Rappresentazioni (i 3 approcci confrontati)
       A) MOMENT: foundation model, 1024D, apprende dinamica temporale
       B) Feature engineering: 19 statistiche domain-specific
       C) Combinato: FE(19D) + MOMENT compresso(55D) = 74D
       Compressione MOMENT: PCA Marchenko-Pastur 1024D -> 55D (denoising)
       Bilanciamento: StandardScaler su tutto il concatenato
       ILR (fuel mix): non nell'input, solo per caratterizzazione post-clustering

   4.2 Diffusion Maps [Coifman & Lafon 2006]
       - Scopo: manifold learning non-lineare, preserva geometria densita
       - Kernel adattivo, alpha=1 (Laplace-Beltrami)
       - n_comp: automatico via sweep silhouette [2..20] (geometrico, no bias prezzo)

   4.3 ToMATo [Chazal et al. 2013]
       - Scopo: clustering topologico, K dal persistence diagram
       - Densita logDTM, grid search k, max_gap per K
       - No assunzioni parametriche (no Gaussiane, no K a priori)

   4.4 Tukey HSD merge
       - Scopo: fondere cluster topologici non distinguibili economicamente
       - Merge iterativo su LMP, alpha=0.05
       - Garanzia: ogni regime sopravvissuto significativamente diverso sugli altri

5. RESULTS
   5.1 Tabella confronto (eta^2, silhouette, K, sojourn, ARI, trans. rate)
   5.2 Caratterizzazione regimi per approccio (LMP, fuel mix, stagionalita)
   5.3 Matrici di transizione (persistenza, transizioni tra regimi)
   5.4 Cosa cattura MOMENT che FE non cattura (ARI=0.09 tra A e B)
   5.5 Stabilita bootstrap (ARI su 20 run, 80% subsample)
   5.6 Analisi di sensibilita (arcsinh vs log_return, MSTL si/no, context/stride)

6. DISCUSSION
   - FE vince su separazione economica (eta^2), MOMENT su stabilita geometrica (ARI)
   - Approcci combinati bilanciano entrambi gli aspetti
   - ILR migliora geometria cluster (silhouette) non separazione prezzo
   - Foundation models: opachi ma catturano dinamica temporale oltre le statistiche aggregate
   - Domain knowledge vs rappresentazioni apprese: complementari, non sostitutive

7. CONCLUSION
   - 4 approcci, stessa pipeline downstream, confronto onesto
   - Regimi interpretabili: baseload, stress, spike (confermati da fuel mix e stagionalita)
   - Pratico: FE per semplicita, Combinato per completezza
```

---

## Pipeline finale consolidata (2026-03-30)

```
PREPROCESSING
  LMP -> arcsinh -> MSTL(24h, 168h, 8760h) -> mstl_resid_arcsinh
  Fuel 8 shares -> ILR SBP -> MSTL(24h, 168h, 8760h) -> mstl_resid_ilr_1..7
  windows=[25, 169, 8761] (regola period+1, Cleveland et al. 1990)
  Sliding windows: 512h context, 6h stride -> 7132 finestre

RAPPRESENTAZIONE (3 approcci)
  A: MOMENT-1-large (1024D) — foundation model, dinamica temporale
  B: 19 feature hand-crafted (19D) — statistiche domain-specific
  C: FE(19D) + MOMENT PCA-MP(55D) = 74D — combinato

RIDUZIONE DIMENSIONALE
  Diffusion Maps (Coifman & Lafon 2006)
  alpha=1 (Laplace-Beltrami), k=sqrt(n), kernel adattivo
  n_comp: automatico via sweep silhouette [2..20]

CLUSTERING (2 metodi)
  ToMATo (Chazal et al. 2013): grid logDTM k=[20..150], max_gap, + Tukey HSD merge
  Ward (baseline): clustering gerarchico, + Tukey HSD merge

POST-ANALYSIS
  Caratterizzazione regimi con ILR (fuel mix centroid, stagionalita, profili 24h)
  Matrice di transizione tra regimi
  Bootstrap ARI (stabilita)

= 3 rappresentazioni x 2 clustering = 6 configurazioni
```

### Design sperimentale

| | ToMATo + Tukey | Ward + Tukey |
|-|---------------|-------------|
| A: MOMENT 1024D | config A1 | config A2 |
| B: FE 19D | config B1 | config B2 |
| C: COMBO 74D | config C1 | config C2 |

Tutte le 6 configurazioni condividono: preprocessing, sliding windows, Diffusion Maps.
Variano: rappresentazione (input) e metodo di clustering (finale).

### Scelte consolidate con razionale

| Scelta | Razionale | Evidenza |
|--------|-----------|----------|
| arcsinh(LMP) | Gestisce negativi, comprime spike, no parametri | Testato vs log_return: eta^2 0.28 vs 0.02 |
| MSTL 24h+168h+8760h su tutti i canali | Rimuove stagionalita nota | Testato: senza MSTL K=4 grossolani; con MSTL K=9-10 fini |
| windows period+1 | Regola standard (Cleveland 1990) | Non calibrato — scelta conservativa |
| context=512h stride=6h | Griglia [72,168,336,512]x[6,12,24] | 512/6 vince su eta^2 |
| ILR solo post-analysis | 7 scalari su 1031D = 0.7% | Con ILR: eta^2=0.245; senza: 0.267 |
| n_comp via silhouette | Criterio geometrico, no circolarita | Stabile su 4/5 seed |
| Tukey merge alpha=0.05 | Fonde cluster non distinti | 100% coppie significative post-merge |
| Ward come baseline | Standard in letteratura energy markets | Da confrontare con ToMATo |

---

## Domande aperte

- Rieseguire step01 con MSTL annuale anche per LMP (codice gia aggiornato)
- Rieseguire step02 (embedding MOMENT) sul nuovo preprocessed
- Rieseguire FE sul nuovo preprocessed
- Implementare Ward + Tukey merge
- Caratterizzazione regimi per tutte le 6 configurazioni
- Stesura paper
