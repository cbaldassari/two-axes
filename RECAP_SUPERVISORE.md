# RECAP PER SUPERVISORE — NEPOOL TDA Regime Detection
**Ultimo aggiornamento: 31 marzo 2026**

---

## 1. OBIETTIVO DEL LAVORO

Identificare **regimi strutturali** del mercato elettrico ISO New England in modo **completamente non supervisionato** (K non fissato a priori), confrontando sistematicamente:
- 3 **rappresentazioni** delle serie temporali (foundation model vs feature engineering vs combinato)
- 2 **metodi di clustering** (topologico vs gerarchico)

= 6 configurazioni sulla stessa pipeline downstream.

**Domanda di ricerca centrale**: un foundation model pre-addestrato (MOMENT, 340M parametri, ICML 2024) può sostituire o complementare feature hand-crafted per la detection di regimi in mercati elettrici?

---

## 2. DATI

### 2.1 Fonte e struttura

| Parametro | Valore |
|-----------|--------|
| Mercato | ISO New England (ISONE) |
| Nodo | Massachusetts Hub (principale hub di prezzo del New England) |
| Variabile target | LMP (Locational Marginal Price, $/MWh) — prezzo orario spot |
| Fuel mix | 8 tipi da EIA (Energy Information Administration): gas naturale, nucleare, idroelettrico, eolico, solare, carbone, olio, altro |
| Frequenza | Oraria |
| Periodo | 2021-01-01 → 2025-12-31 |
| Righe raw | 43.814 |
| Righe dopo pulizia | 43.795 (19 rimosse) |
| Finestre sliding | 7.214 (512h contesto, 6h stride) |

**Perché ISONE**: il mercato ISO New England è un mercato deregolato (wholesale) dove il prezzo è determinato in tempo reale dall'equilibrio domanda-offerta. Presenta tre caratteristiche che lo rendono ideale per lo studio dei regimi: (1) forte dipendenza dal gas naturale (~55% della generazione), che lega il prezzo elettrico alla volatilità del gas; (2) cold snap invernali che causano spike di prezzo estremi (la domanda di riscaldamento compete con la generazione elettrica per il gas); (3) espansione delle rinnovabili (eolico e solare crescenti) che introduce periodi di prezzo molto basso. Il periodo 2021-2025 è particolarmente ricco: copre la ripresa post-COVID, la crisi energetica 2022 (prezzi gas alle stelle dopo l'invasione russa dell'Ucraina), l'espansione solare, e diversi eventi meteo estremi.

### 2.2 Distribuzione del prezzo LMP

| Statistica | Valore |
|-----------|--------|
| Minimo | 9.15 $/MWh |
| 5° percentile | ~18 $/MWh |
| Mediana | 41.05 $/MWh |
| Media | 55.51 $/MWh |
| 95° percentile | ~160 $/MWh |
| Massimo | 474.90 $/MWh |
| Prezzi negativi | 0 (nel periodo considerato) |
| Prezzi > 200 $/MWh | 632 ore (1.44%) |

La distribuzione è fortemente asimmetrica a destra (media >> mediana): la coda destra contiene gli spike di prezzo che definiscono i regimi di stress. Il rapporto media/mediana di 1.35 indica che gli eventi estremi alzano la media significativamente.

**Nota**: nel dataset ISONE 2021-2025 non ci sono prezzi negativi. Questo può sembrare sorprendente dato che i mercati europei ne hanno frequentemente. In New England, i prezzi negativi sono rari perché il nucleare (che non può ridurre rapidamente) rappresenta solo il ~27% e le rinnovabili (wind+solar) sono ancora ~5%. In futuro, con l'espansione del solare, ci si aspetta che emergano. La scelta di arcsinh è comunque corretta: gestisce i negativi senza costo aggiuntivo, ed è forward-looking per applicazioni su altri mercati (PJM, ERCOT, mercati europei).

### 2.3 Fuel mix: struttura e zeri

Le 8 quote di generazione per combustibile sono **dati composizionali**: sommano a 1 (100%) in ogni ora. Questo ha implicazioni importanti per l'analisi statistica (vedi §3.4).

| Combustibile | Quota media | Zeri (su 43.814) | Interpretazione degli zeri |
|-------------|-------------|-------------------|---------------------------|
| Gas naturale | 55.7% | 0 | Sempre attivo — baseload e picco |
| Nucleare | 26.5% | 1 | Quasi sempre attivo — baseload |
| Idroelettrico | 7.0% | 0 | Sempre presente — run-of-river |
| Eolico | 3.8% | 16 (0.04%) | Raro a zero — qualche ora di calma |
| Solare | 1.1% | 11.356 (25.9%) | **Zero ogni notte** — assenza naturale |
| Carbone | 0.3% | 22.622 (51.6%) | **Spesso zero** — phase-out in corso |
| Olio | 0.3% | 28.280 (64.5%) | **Quasi sempre zero** — solo emergenza |
| Altro | 5.1% | 0 | Include biomassa, rifiuti, import |

Gli zeri nel fuel mix sono un punto critico. Il solare è zero ogni notte (strutturale), carbone e olio sono zero nella maggioranza delle ore (perché troppo costosi/inquinanti e attivati solo in emergenza). Questi zeri devono essere gestiti prima della trasformazione ILR (che richiede log delle proporzioni).

### 2.4 Dati mancanti e pulizia

| Problema | Quantità | Trattamento |
|----------|----------|-------------|
| **Fuel share NA** | 6 righe (0.01%) su tutti gli 8 combustibili simultaneamente | Le 6 righe hanno dati LMP validi ma fuel mix completamente mancante. Gestite nel preprocessing: NaN fill con mediana per MSTL, NaN ripristinati nel residuo |
| **MW generazione negativi** | 5 righe (1 per hydro, gas, nuclear, other, wind) | Righe rimosse (MW negativo è fisicamente impossibile — errore di misurazione) |
| **Duplicati temporali** | 14 righe | Tenuto il primo, rimosso il duplicato |
| **Ore fuori range** | 0 | Dataset pulito, nessuna ora fuori 2021-2025 |
| **Totale rimossi** | 19 righe (0.04%) | 43.814 → 43.795 |

La pulizia è conservativa: rimuoviamo solo le righe con errori fisici (MW negativi) o duplicati. I 6 NA nel fuel mix non sono rimossi — i dati LMP sono validi e i NA sono gestiti nel preprocessing ILR (fill con mediana per MSTL, NaN ripristinati nei residui, ignorati nel clustering che usa solo il prezzo).

### 2.5 Perché sliding windows 512h / 6h stride

Un singolo punto orario (prezzo + fuel mix) non definisce un regime. Serve una **finestra temporale** che catturi il "comportamento" del mercato in quel periodo. La scelta della dimensione della finestra e del passo di campionamento è fondamentale.

| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| **Contesto** | 512h (~21 giorni) | Massimo input nativo di MOMENT. Copre ≥3 cicli settimanali. Griglia testata: 72h/168h/336h/512h — 512h produce η² migliore (0.161 vs 0.089 per 72h) |
| **Stride** | 6h | ~4 finestre/giorno. Bilancia risoluzione (cattura transizioni intra-giornaliere) vs ridondanza (overlap 97%) |
| **N finestre** | 7.214 | (43.795 - 512) / 6 + 1 |
| **Timestamp** | Ultima ora della finestra | Convenzione: la finestra "rappresenta" il mercato fino a quel momento |

**Griglia di calibrazione contesto × stride** (esperimento con pipeline completa):

| ctx (h) | stride (h) | N finestre | η² (pre-merge) | Silhouette |
|---------|-----------|-----------|---------------|------------|
| 72 | 6 | 7.280 | 0.089 | 0.198 |
| 168 | 6 | 7.248 | 0.093 | 0.234 |
| 336 | 6 | 7.192 | 0.081 | 0.256 |
| **512** | **6** | **7.132** | **0.161** | **0.285** |
| 512 | 12 | 3.566 | 0.138 | 0.382 |
| 512 | 24 | 1.784 | 0.112 | 0.427 |

Il contesto 512h vince su η² (informazione economica) nonostante l'ACF del residuo scenda sotto 0.05 a 31h. Questo suggerisce che MOMENT cattura dipendenze non-lineari a scala più lunga di quanto l'ACF lineare rilevi. Lo stride 6h massimizza il numero di finestre (più punti per il clustering), a costo di forte overlap.

---

## 3. PREPROCESSING — SCELTE E RAZIONALI

Il preprocessing trasforma i dati grezzi (LMP orario + 8 fuel shares) in una rappresentazione adatta al clustering. Ogni scelta è motivata da proprietà matematiche dei dati elettrici e validata con test di sensitività.

### 3.1 Trasformazione del prezzo: arcsinh(LMP)

#### Il problema

I prezzi LMP del mercato elettrico presentano proprietà che escludono le trasformazioni standard:

| Proprietà | Implicazione per la trasformazione |
|-----------|-----------------------------------|
| **Distribuzione asimmetrica** (media/mediana = 1.35) | Servono trasformazioni che comprimano la coda destra |
| **Code pesanti** (474 $/MWh massimo, 632 ore > 200) | Senza compressione, gli spike dominano le distanze nel clustering |
| **Possibili prezzi negativi** (0 in questo dataset, ma frequenti in altri mercati) | log(x) non è definito per x ≤ 0 |
| **Range ampio** (9 → 475 $/MWh, rapporto 52:1) | La scala assoluta influenza il clustering se non normalizzata |

#### Le alternative considerate

| Trasformazione | Negativi? | Zeri? | Compressione | Parametri | Problema |
|----------------|-----------|-------|-------------|-----------|----------|
| log(x) | No | No | Sì | 0 | Indefinita per ≤ 0 |
| log(x + c) | Sì | Sì | Sì | 1 (c arbitrario) | c cambia la compressione; scelta soggettiva |
| Box-Cox | No | No | Sì | 1 (λ) | Richiede x > 0, λ stimato dai dati |
| Standardizzazione (z-score) | Sì | Sì | No | 2 (μ, σ) | Non comprime le code |
| log-return Δarcsinh | Sì | Sì | Sì | 0 | **Rimuove il livello** → η² = 0.023 |
| **arcsinh(x)** | **Sì** | **Sì** | **Sì** | **0** | **Nessuno** |

#### arcsinh: definizione e comportamento

arcsinh(x) = ln(x + √(x² + 1))

| LMP ($/MWh) | arcsinh(LMP) | Comportamento |
|-------------|-------------|---------------|
| -10 | -3.00 | Simmetrica per negativi |
| 0 | 0.00 | Zero preservato |
| 10 | 2.99 | Quasi lineare (≈ x) |
| 41 (mediana) | 4.40 | Compressione moderata |
| 100 | 5.30 | Compressione logaritmica |
| 200 | 6.09 | Forte compressione |
| 475 (max) | 6.86 | 475 → 6.86 (69× compresso) |

La compressione è graduale: per |x| < 1, arcsinh ≈ x (lineare); per |x| >> 1, arcsinh ≈ log(2|x|) (logaritmica). La transizione è morbida, senza soglia arbitraria.

Riferimenti: Ziel & Weron (2018) adottano arcsinh per prezzi day-ahead europei. Uniejewski et al. (2019) confermano la superiorità su log+shift per mercati con prezzi negativi.

#### Test di sensitività: arcsinh vs log-return

| Metrica | arcsinh (baseline) | log-return |
|---------|-------------------|------------|
| η² (varianza LMP spiegata) | **0.182** | 0.023 |
| K (regimi post-merge) | **9** | non calcolato |
| Interpretazione | I regimi separano livelli di prezzo | I regimi non distinguono livelli di prezzo |

**Conclusione**: il log-return (differenza prima di arcsinh) rimuove l'informazione sul livello del prezzo, che è esattamente ciò che definisce i regimi. Per il forecasting il log-return è preferibile (stazionarietà); per il regime detection, arcsinh preserva l'informazione critica.

### 3.2 Destagionalizzazione: MSTL

#### Il problema

La serie arcsinh(LMP) ha forte stagionalità a tre scale:

| Scala | Periodo | ACF | Causa fisica |
|-------|---------|-----|-------------|
| **Giornaliera** | 24h | 0.890 | Ciclo domanda notte/giorno (picco pomeridiano) |
| **Settimanale** | 168h | 0.678 | Pattern feriale/festivo (domanda più bassa nel weekend) |
| **Annuale** | 8760h | 0.108 | Stagioni (inverno freddo → gas heating, estate calda → AC) |

Se non rimossa, la stagionalità domina il clustering: il modello troverebbe "estate vs inverno" (banale) anziché regimi strutturali di mercato (spike, stress, baseload). La stagionalità è un pattern noto e deterministico — non contiene informazione sui regimi.

#### MSTL: come funziona

MSTL (Multiple Seasonal-Trend decomposition using LOESS, De Livera et al. 2011) decompone una serie in:

```
y(t) = T(t) + S_24(t) + S_168(t) + S_8760(t) + R(t)
```

dove T è il trend, S_k sono le componenti stagionali a periodo k, e R è il residuo. Ogni componente è stimata iterativamente con LOESS (LOcally Estimated Scatterplot Smoothing) — un metodo di regressione locale ponderata.

| Parametro MSTL | Valore | Razionale |
|----------------|--------|-----------|
| **Periodi** | [24, 168, 8760] | Giorno, settimana, anno |
| **Finestre LOESS** | [25, 169, 8761] | Regola period+1 (Cleveland et al. 1990) |
| **inner_iter** | 1 | Single-pass, sufficiente per rimuovere stagionalità (3× più veloce del default 2) |
| **outer_iter** | 0 | Nessuna robustificazione (non serve: non ci sono outlier nella stagionalità) |

#### Il residuo MSTL

| Canale | Varianza residua / totale | Interpretazione |
|--------|--------------------------|-----------------|
| arcsinh(LMP) | 24.6% | Il residuo contiene ~1/4 della varianza: shock di prezzo, cambi di regime, variabilità non stagionale |
| ilr_1 (disp vs var) | ~12% | La composizione del fuel mix è fortemente stagionale |
| ilr_3 (gas vs solid) | ~56% | La distinzione gas/fossili solidi è meno stagionale |
| ilr_6 (hydro vs intermittent) | ~41% | Idro vs eolico+solare: moderatamente stagionale |

Il residuo del prezzo contiene il 24.6% della varianza totale di arcsinh(LMP). Questo 25% include: gli shock di prezzo (spike), i cambi di regime (transizioni), e la variabilità non spiegata dalla stagionalità.

#### Perché la componente annuale per LMP?

L'ACF annuale di arcsinh(LMP) è debole (0.108). Per i canali ILR è molto più forte (0.40 per ilr_1). La componente annuale per il prezzo è stata inclusa per **consistenza** della pipeline (tutti i canali trattati allo stesso modo). Test di sensitività: la differenza nel residuo è trascurabile (std cambia di < 0.002).

#### Test di sensitività: con MSTL vs senza

| Metrica | Con MSTL (baseline) | Senza MSTL |
|---------|-------------------|------------|
| η² (sweep) | **0.182** | 0.189 |
| K (post-merge) | **9** (struttura fine) | 4 (grossolano) |
| Interpretazione | Regimi di mercato strutturali | Essenzialmente "estate vs inverno" |

**Conclusione**: senza MSTL, il clustering cattura la stagionalità (K=4, banale). Con MSTL, cattura la struttura residua dei regimi (K=9-10, informativa). L'η² nel sweep è simile (~0.18) ma il risultato post-merge è radicalmente diverso.

### 3.3 Gestione dei dati mancanti e degli zeri

#### Missing values (6 righe fuel mix)

Le 6 righe con fuel mix completamente mancante (0.01% del dataset) sono gestite in modo conservativo:

| Fase | Trattamento | Razionale |
|------|-------------|-----------|
| **ILR** | Le 6 righe hanno NaN in tutte le 7 coordinate ILR | Impossibile calcolare log-ratio con dati mancanti |
| **MSTL fit** | NaN riempiti con la **mediana** del canale prima del fit | MSTL richiede serie completa; la mediana è robusta agli outlier |
| **Residuo MSTL** | NaN **ripristinati** nelle posizioni originali dopo il fit | Non vogliamo che il fill-value sia trattato come dato reale |
| **Clustering** | Le finestre che contengono le 6 righe usano i residui LMP (validi) | Il clustering usa il prezzo, non l'ILR. Le 6 righe non influenzano i cluster |
| **Post-analisi ILR** | Le 6 righe sono escluse (NaN nel fuel mix) | Il fuel mix non è disponibile per queste ore |

#### Zeri nel fuel mix (solare, carbone, olio)

I dati composizionali con zeri strutturali sono un problema noto nella statistica composizionale [Martín-Fernández et al. 2003]. Il logaritmo di zero è indefinito, e l'ILR richiede il log delle proporzioni.

| Combustibile | Ore a zero | % | Tipo di zero | Trattamento |
|-------------|-----------|---|-------------|-------------|
| Solare | 11.356 | 25.9% | **Strutturale** (notte) | Sostituzione moltiplicativa δ=0.0001 |
| Carbone | 22.622 | 51.6% | **Essenziale** (non attivato) | Sostituzione moltiplicativa δ=0.0001 |
| Olio | 28.280 | 64.5% | **Essenziale** (solo emergenza) | Sostituzione moltiplicativa δ=0.0001 |
| Eolico | 16 | 0.04% | **Arrotondamento** | Sostituzione moltiplicativa δ=0.0001 |

**Sostituzione moltiplicativa** [Martín-Fernández et al. 2003]: gli zeri sono sostituiti con δ=0.0001, e tutte le proporzioni sono rinormalizzate a somma 1. Questo è lo standard per dati composizionali con zeri — preserva la struttura relativa delle proporzioni non-zero.

**Distorsione nota**: la sostituzione degli zeri introduce una piccola distorsione nelle coordinate ILR per le ore notturne (solare=0) e per le ore senza carbone/olio. Questa distorsione è minima (δ=0.0001 corrisponde a 0.01% di generazione) e coerente con l'idea che "zero generazione" non significa "capacità zero" — è un valore limite.

### 3.4 Fuel mix composizionale: trasformazione ILR

#### Perché non usare le proporzioni raw

Le 8 quote di generazione sommano a 100% in ogni ora. Questo vincolo (dati composizionali, spazio del simplesso S⁷) rende le operazioni statistiche standard (media, varianza, distanza euclidea, correlazione) **inappropriate** [Aitchison 1986]. Per esempio:
- La "media" di due composizioni (50%, 50%) e (50%, 50%) è (50%, 50%) — corretto.
- Ma la "media" di (99%, 1%) e (1%, 99%) dovrebbe essere (50%, 50%), non il centroide aritmetico.
- La correlazione tra due quote è spuria: se gas sale, le altre devono scendere (vincolo di somma), creando correlazioni negative artificiali.

La trasformazione ILR (Isometric Log-Ratio, Egozcue et al. 2003) proietta le 8 quote in 7 coordinate in R⁷ dove le operazioni standard sono matematicamente valide.

#### La base SBP (Sequential Binary Partition)

La base SBP è una partizione gerarchica dei fuel types che produce coordinate interpretabili:

```
Livello 1: [gas, nuc, coal, oil, other] vs [hydro, wind, solar]     → ilr_1 (dispatchable vs variable)
Livello 2a: [gas, coal, oil] vs [nuc, other]                         → ilr_2 (fossil vs non-fossil disp.)
Livello 2b: [hydro] vs [wind, solar]                                  → ilr_6 (hydro vs intermittent)
Livello 3a: [gas] vs [coal, oil]                                      → ilr_3 (gas vs solid fossil)
Livello 3b: [nuc] vs [other]                                          → ilr_5 (nuclear vs other)
Livello 3c: [solar] vs [wind]                                         → ilr_7 (solar vs wind)
Livello 4:  [coal] vs [oil]                                           → ilr_4 (coal vs oil)
```

La formula per ogni coordinata è un balance:

```
ilr_k = √(r_p · r_n / (r_p + r_n)) · (mean(log(positive)) - mean(log(negative)))
```

dove r_p e r_n sono il numero di componenti nei due gruppi.

#### Le 7 coordinate ILR — significato fisico

| Coord. | Contrasto | Cosa misura | Range tipico |
|--------|-----------|-------------|-------------|
| **ilr_1** | dispatchable vs variable | Quanto il sistema dipende da fonti controllabili vs rinnovabili/idro | Da -0.6 (off-peak, più rinnovabili) a +1.7 (spike, più dispatchable) |
| **ilr_2** | fossile vs non-fossile disp. | Quanto il dispatchable è fossile (gas,coal,oil) vs nucleare+altro | Da -4.4 (più nucleare) a -2.1 (più fossile) |
| **ilr_3** | gas vs solid fossil | Se il fossile è gas o carbone+olio | Da 4.1 (più carbone/olio) a 6.7 (quasi tutto gas) |
| **ilr_4** | coal vs oil | Se il fossile solido è carbone o olio | Da -0.7 (più olio) a +0.3 (più carbone) |
| **ilr_5** | nuclear vs other | Se il non-fossile disp. è nucleare o altro (biomassa) | ~1.1-1.2 (stabile, dominato dal nucleare) |
| **ilr_6** | hydro vs intermittent | Se il variabile è idro o eolico+solare | ~1.7-2.1 (idro dominante sul variabile) |
| **ilr_7** | solar vs wind | Se l'intermittente è solare o eolico | Da -2.8 (eolico, inverno/notte) a -1.7 (più solare, estate/giorno) |

#### ILR nel clustering: perché solo in post-analisi

L'ILR è stato testato come input al clustering (concatenato all'embedding MOMENT):

| Configurazione | η² | K |
|---------------|-----|---|
| Senza ILR (baseline) | **0.267** | 9 |
| Con ILR (7D concat) | 0.245 | 9 |

L'ILR peggiora leggermente la separazione economica. Motivo: 7 scalari su 1031 dimensioni totali pesano lo 0.7% — le Diffusion Maps li ignorano. L'ILR è molto più utile nella **caratterizzazione post-clustering**: dato un regime, si esamina il fuel mix medio per dare interpretazione economica (es. "regime spike invernale con attivazione olio di backup").

### 3.5 Riepilogo del flusso di preprocessing

```
INPUT: 43.814 righe orarie (LMP + 8 fuel shares)
  │
  ├─ Pulizia: -19 righe (duplicati, MW negativi) → 43.795
  │
  ├─ Prezzo:
  │    LMP → arcsinh(LMP)
  │         → MSTL(24h+168h+8760h) → mstl_resid_arcsinh
  │              (12% var totale = shock + regimi + rumore)
  │
  ├─ Fuel mix:
  │    8 shares → zero replacement (δ=0.0001)
  │            → ILR SBP (8→7 coord.)
  │            → MSTL(24h+168h+8760h) → mstl_resid_ilr_1..7
  │              (6 NaN gestiti: fill mediana → MSTL → NaN ripristinati)
  │
  └─ Sliding windows:
       mstl_resid_arcsinh → finestre 512h, stride 6h → 7.214 finestre
       LMP raw → stesse finestre (per feature engineering)

OUTPUT: 7.214 finestre × {512 valori residuo + 512 valori LMP}
```

---

## 4. RAPPRESENTAZIONI — SCELTE E RAZIONALI

Il cuore dello studio è il confronto tra tre modi alternativi di descrivere ogni finestra di 512 ore. Le tre rappresentazioni condividono l'input (le stesse finestre sliding di residui MSTL) e la stessa pipeline downstream (Diffusion Maps → clustering → Tukey merge). L'unica differenza è come la finestra viene trasformata in un vettore numerico.

### 4.1 A: MOMENT — Foundation Model (1024D)

#### Cos'è MOMENT

MOMENT (Goswami et al., ICML 2024, Carnegie Mellon) è un foundation model per serie temporali basato su un encoder T5 con 340 milioni di parametri, pre-addestrato con **masked reconstruction** su Time-Series Pile — una collezione diversificata di serie temporali pubbliche (sensori, traffico, meteo, produzione industriale, finanza). A differenza di Chronos (Amazon, forecasting-only) e TimesFM (Google, decoder-only), MOMENT offre una modalità `embedding` nativa: dato un vettore di 512 valori, produce un vettore di 1024 dimensioni che codifica la "struttura temporale" della serie dal punto di vista del modello.

| Caratteristica | Valore |
|---------------|--------|
| Architettura | T5 encoder |
| Parametri | 340M |
| Pre-training | Masked reconstruction su Time-Series Pile |
| Input | 1 canale × 512 punti |
| Output (embedding mode) | 1024 dimensioni |
| Training data | ~1B time points eterogenei (non finanziari, non energetici) |

#### Cosa cattura e cosa non cattura

L'embedding MOMENT codifica pattern temporali complessi: la *forma* della serie (rampe, oscillazioni, plateau), le dipendenze non-lineari tra scale temporali, i pattern di transizione. Tuttavia:

- **Non conosce i mercati elettrici**: il training set non include dati energetici. I pattern che cattura sono generici (trend, stagionalità residua, autocorrelazione) — non conosce la relazione prezzo-fuel o la struttura merit-order.
- **Non vede il livello assoluto**: l'embedding è invariante a traslazioni verticali (per design del T5). Questo significa che non distingue una finestra a 40$/MWh da una a 140$/MWh se la *forma* è simile.
- **Opaco**: non sappiamo cosa significano le 1024 dimensioni. L'interpretabilità richiede analisi post-hoc (correlazione con feature fisiche, random forest per importanza).

#### Scelte di design

| Scelta | Razionale | Alternativa testata |
|--------|-----------|---------------------|
| **MOMENT-1-large** | Unico foundation model con embedding nativo per representation learning | Chronos-2 (Amazon): testato in fase 1, R² negativo su tutte le feature, embedding opaco e inutilizzabile |
| **Single channel** (solo residuo arcsinh) | MOMENT è univariato per design | Multi-channel (8 canali): testato nella fase 4, non migliora. Costo 8× senza beneficio |
| **512h context** | Massimo nativo di MOMENT. La griglia [72, 168, 336, 512] mostra che 512h produce η² migliore | 168h: η²=0.093 vs 0.161 per 512h |
| **Zero-shot** (no fine-tuning) | Confronto equo: il foundation model vs feature engineering, entrambi senza adattamento | Fine-tuning testato: η² sweep 0.185 vs 0.182 zero-shot — nessun miglioramento |
| **Batch size 64** | Massimo che entra in VRAM GPU locale | — |

### 4.2 B: Feature Engineering — 19 Statistiche Domain-Specific (19D)

#### Filosofia

L'approccio opposto a MOMENT: anziché lasciare che un modello pre-addestrato decida cosa è rilevante, il ricercatore specifica esplicitamente 19 statistiche che catturano aspetti noti della dinamica dei prezzi elettrici. Ogni dimensione ha un significato economico preciso e interpretabile.

#### Le 19 feature dettagliate

| # | Feature | Formula | Calcolata su | Cosa cattura |
|---|---------|---------|-------------|-------------|
| 1 | mean | μ = (1/n)Σxᵢ | residuo arcsinh | Livello medio della finestra (↑ in spike) |
| 2 | std | σ = √(Var) | residuo arcsinh | Dispersione (↑ in regimi volatili) |
| 3 | skew | γ₁ = E[(x-μ)³]/σ³ | residuo arcsinh | Asimmetria (+ = coda destra = spike) |
| 4 | kurtosis | γ₂ = E[(x-μ)⁴]/σ⁴ - 3 | residuo arcsinh | Code pesanti (↑ = più eventi estremi) |
| 5 | min | min(xᵢ) | residuo arcsinh | Valore minimo nella finestra |
| 6 | max | max(xᵢ) | residuo arcsinh | Valore massimo (↑ in spike) |
| 7 | range | max - min | residuo arcsinh | Escursione (↑ = più variabile) |
| 8 | median | Q50 | residuo arcsinh | Centro robusto della distribuzione |
| 9 | p5 | P5 | residuo arcsinh | Coda sinistra (prezzi bassi nella finestra) |
| 10 | p95 | P95 | residuo arcsinh | Coda destra (spike nella finestra) |
| 11 | iqr | Q75 - Q25 | residuo arcsinh | Dispersione robusta |
| 12 | acf_1h | ρ(1) | residuo arcsinh | Persistenza ora-ora (↑ = prezzo "sticky") |
| 13 | acf_6h | ρ(6) | residuo arcsinh | Persistenza intra-giornaliera |
| 14 | acf_24h | ρ(24) | residuo arcsinh | Persistenza giornaliera (ciclo residuo) |
| 15 | acf_168h | ρ(168) | residuo arcsinh | Persistenza settimanale (ciclo residuo) |
| 16 | vol_24h | mean(|Δx| per blocchi 24h) | residuo arcsinh | Volatilità intra-giornaliera media |
| 17 | lmp_mean | μ(LMP) | **LMP raw $/MWh** | Livello di prezzo assoluto |
| 18 | lmp_p95 | P95(LMP) | **LMP raw $/MWh** | Spike di prezzo assoluto |
| 19 | lmp_std | σ(LMP) | **LMP raw $/MWh** | Volatilità di prezzo assoluta |

**Nota critica**: le feature 17-19 sono calcolate sul LMP raw (non sul residuo). Questo è deliberato: il livello assoluto del prezzo è l'informazione più discriminante per i regimi. Il residuo MSTL cattura la struttura *relativa* (shock vs normalità), ma il livello *assoluto* (27$/MWh vs 137$/MWh) è ciò che definisce economicamente un regime. L'inclusione di queste 3 feature spiega in gran parte perché FE produce η²=0.43 vs 0.18 di MOMENT (che non vede il livello).

#### Cosa cattura e cosa non cattura

**Cattura**: livello, dispersione, forma della distribuzione, persistenza temporale a 4 scale, volatilità intra-day. Ogni feature corrisponde a un concetto economico noto.

**Non cattura**: la *forma* della serie nella finestra. Due finestre con la stessa media/std/skew ma profilo temporale diverso (es. spike singolo vs plateau alto) produrranno vettori simili. MOMENT potrebbe distinguerle.

### 4.3 C: COMBO — Combinato (72D)

#### Logica

Se MOMENT cattura la dinamica temporale e FE cattura le statistiche aggregate, combinarli dovrebbe produrre una rappresentazione più completa. In pratica:

1. MOMENT 1024D → **PCA Marchenko-Pastur** → 53D (denoising: solo il 5.2% delle componenti contiene segnale sopra la soglia RMT)
2. FE 19D (invariato)
3. Concatenazione: 53 + 19 = 72D
4. **StandardScaler** su tutto il concatenato (media 0, varianza 1 per dimensione)

#### PCA Marchenko-Pastur: perché e come

La Random Matrix Theory (Marchenko & Pastur 1967) fornisce una soglia per distinguere segnale da rumore negli autovalori di una matrice di covarianza campionaria. Per dati standardizzati con n osservazioni e p variabili:

```
λ_max = (1 + √(p/n))²
```

Tutti gli autovalori > λ_max contengono segnale reale; quelli ≤ λ_max sono compatibili con rumore puro. Per MOMENT (p=1024, n=7214): λ_max ≈ 1.31, 53 componenti sopra soglia (85% varianza spiegata). Le restanti 971 dimensioni sono rumore statistico.

#### Risultato

| Metrica | A: MOMENT | B: FE | C: COMBO |
|---------|-----------|-------|----------|
| η² | 0.175 | **0.433** | 0.378 |
| K | 8 | 10 | 9 |
| Sojourn | 14h | 222h | 27h |

Il COMBO (η²=0.378) è intermedio: migliore di MOMENT puro ma peggiore di FE. **MOMENT aggiunge rumore più che segnale** quando combinato con FE. Le 53 componenti PCA non sono allineate con la struttura dei regimi di prezzo — il segnale utile per il clustering è già tutto nelle 19 feature hand-crafted.

---

## 5. PIPELINE DOWNSTREAM — SCELTE E RAZIONALI

La pipeline downstream è identica per le 3 rappresentazioni e i 2 metodi di clustering. Questo rende il confronto equo: l'unica variabile che cambia è l'input (1024D vs 19D vs 72D) e il metodo di clustering (ToMATo vs Ward).

### 5.1 Diffusion Maps — Riduzione dimensionale non-lineare

#### Il problema

I vettori di rappresentazione vivono in spazi ad alta dimensione (1024D per MOMENT, 19D per FE, 72D per COMBO). In alta dimensione le distanze euclidee perdono potere discriminante (curse of dimensionality) e i metodi di clustering basati sulla densità (ToMATo) soffrono. Serve una riduzione dimensionale che preservi la geometria rilevante per il clustering.

#### Perché Diffusion Maps e non PCA/UMAP

| Metodo | Tipo | Preserva | Compatibilità con ToMATo | Problema |
|--------|------|----------|--------------------------|----------|
| **PCA** | Lineare | Varianza globale | Bassa — la topologia non-lineare è proiettata male | η² 3× inferiore a DiffMaps su stessi dati |
| **UMAP** | Non-lineare | Vicinanza locale | Bassa — distorce la topologia globale, crea cluster artificiali | Introdotto nella pipeline precedente e scartato per circolarità |
| **t-SNE** | Non-lineare | Vicinanza locale | Molto bassa — solo visualizzazione, distanze globali prive di significato | Non adatto a downstream analysis |
| **Diffusion Maps** | Non-lineare | **Distanza geodetica sul manifold** | **Alta — preserva la landscape di densità** | Costo O(n² × k_nn) |

Le Diffusion Maps (Coifman & Lafon 2006) costruiscono un grafo di vicinanza pesato (kernel gaussiano), calcolano la matrice di transizione di un random walk sul grafo, e usano gli autovettori come coordinate. Le coordinate diffusive approssimano la **distanza geodetica** sul manifold dei dati — cioè la distanza "camminando lungo la superficie" dei dati anziché in linea retta. Questo è essenziale per ToMATo, che opera sulla landscape di densità del manifold.

#### Parametri e implementazione

| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| **Kernel** | Gaussiano adattivo, σ = mediana delle distanze al k-esimo vicino | Standard. σ adattivo rende il kernel robusto alla scala locale |
| **α = 1** (normalizzazione Laplace-Beltrami) | Separa geometria del manifold dalla densità dei punti. Senza α=1, le regioni dense dominano le coordinate | |
| **k_nn** | max(15, √n) ≈ 85 | Bilancia connettività locale (15 minimo) e globale (√n scala con il dataset) |
| **n_comp** | Automatico via sweep silhouette [2..20] | Per ogni candidato: DiffMaps → ToMATo → silhouette. Si sceglie il n_comp con silhouette massima (criterio puramente geometrico, senza usare LMP — nessuna circolarità) |
| **PCA denoising** | Marchenko-Pastur prima di DiffMaps se dim input > 100 | Per MOMENT (1024D): PCA 1024→53D prima di DiffMaps. Riduce il rumore che altrimenti confonde il kernel gaussiano |

#### Risultati n_comp selection

| Rappresentazione | Input dim | PCA denoising | n_comp selezionato | Silhouette | Interpretazione |
|-----------------|----------|---------------|-------------------|------------|-----------------|
| A: MOMENT | 1024 → 53 (PCA) | Sì | **2** | 0.283 | Struttura semplice: manifold 2D |
| B: FE | 19 | No | **11** | 0.410 | Struttura ricca: 11 dimensioni informative |
| C: COMBO | 72 | No | **2** | 0.376 | Dominato da MOMENT (PCA compresso) |

**Osservazione importante**: MOMENT e COMBO collassano a 2D. Questo spiega perché Ward (che assume cluster convessi) produce solo K=3 su queste rappresentazioni — in 2D ci sono pochi bacini distinguibili. FE produce 11 dimensioni informative → più struttura → Ward trova K=12 (poi merge a 7).

### 5.2 Clustering: ToMATo vs Ward

Usiamo due metodi di clustering con filosofie opposte per robustezza del confronto.

#### 5.2.1 ToMATo — Clustering topologico

ToMATo (Topological Mode Analysis Tool, Chazal, Guibas, Oudot & Skraba 2013) è un algoritmo di clustering basato sulla persistenza topologica. Il suo funzionamento:

1. **Stima della densità**: logDTM (logarithm of the Distance To a Measure), una stima robusta della densità locale basata sulle distanze ai k vicini più prossimi. Più robusto di KDE per dati in alta dimensione.

2. **Grafo di vicinanza**: k-NN con il parametro k selezionato via grid search.

3. **Identificazione dei modi**: i picchi della funzione di densità corrispondono ai "modi" (cluster). Ogni punto è assegnato al modo nel cui bacino di attrazione "cade" seguendo il gradiente ascendente della densità.

4. **Persistence diagram**: ogni modo ha una "persistenza" = differenza tra il suo picco di densità e il punto in cui si fonde con un modo più alto. I modi con alta persistenza sono robusti; quelli con bassa persistenza sono rumore topologico.

5. **Selezione di K**: il gap massimo nel persistence diagram (ordinato per persistenza decrescente) separa i modi significativi dal rumore. K = indice del gap + 1.

| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| **density_type** | logDTM | Robusto in alta dimensione, standard per TDA |
| **graph_type** | k-NN | Efficiente, ben definito |
| **k (DTM)** | Grid search {20, 40, 60, 80, 100, 150} | k controlla la risoluzione della densità. Troppo basso = rumore, troppo alto = oversmoothing |
| **Selezione k** | Max gap_norm (gap / mediana persistenze) | Normalizzato per confrontabilità tra diversi k |
| **Selezione K** | Max-gap nel persistence diagram del k selezionato | Standard in TDA |

#### 5.2.2 Ward — Clustering gerarchico (baseline)

Ward linkage (Ward 1963) è il metodo gerarchico più comune in letteratura sui mercati energetici. Minimizza la varianza intra-cluster ad ogni passo di merge, producendo cluster convessi e bilanciati. Lo includiamo come **baseline tradizionale** per confronto.

| Parametro | Valore | Razionale |
|-----------|--------|-----------|
| **Metodo** | Ward (varianza minima) | Standard in letteratura energy markets |
| **Distanza** | Euclidea | Coerente con Ward |
| **Selezione K** | Max gap nelle distanze di merge (ultimo 5%) | Automatico, basato sulla struttura del dendrogramma |

#### Confronto ToMATo vs Ward

| Aspetto | ToMATo | Ward |
|---------|--------|------|
| Assunzione sulla forma dei cluster | **Nessuna** — segue la densità | Cluster convessi (ellissoidali) |
| Selezione K | Persistence diagram (topologico) | Gap nelle distanze di merge |
| Robustezza in bassa dim | **Alta** — trova struttura fine | Bassa — collassa a K=3 in 2D |
| Robustezza in alta dim | Alta (opera su densità locale) | Alta (Ward è stabile) |
| Costo | Più alto (grid search + Tukey merge con molti cluster) | Basso (linkage + cut) |
| K tipico (pre-merge) | 53-132 | 3-12 |
| K tipico (post-merge) | 8-10 | 3-7 |

### 5.3 Tukey HSD merge — Consolidamento economico

#### Il problema

ToMATo può produrre molti cluster topologicamente distinti ma **economicamente indistinguibili** — due picchi di densità a livello di prezzo simile. Ward può produrre cluster geometricamente separati ma con sovrapposizione nei prezzi.

#### La soluzione

Il test di Tukey HSD (Honestly Significant Difference, α=0.05) verifica la significatività della differenza di LMP medio tra ogni coppia di cluster. L'algoritmo di merge:

1. Calcola Tukey HSD su tutte le coppie di cluster
2. Identifica la coppia con p-value più alto tra quelle non significative
3. Fonde i due cluster
4. Ricalcola Tukey HSD sulle nuove etichette
5. Ripete fino a che tutte le coppie sopravvissute sono significativamente diverse

**Non c'è circolarità**: i cluster sono formati nello spazio diffusivo (senza usare LMP), poi validati e consolidati tramite LMP. Il merge è un filtro *a posteriori*, non un input alla formazione.

| Configurazione | K pre-merge | K post-merge | Iterazioni |
|---------------|-------------|-------------|------------|
| A:MOMENT+ToMATo | 65 | 8 | 57 |
| B:FE+ToMATo | 53 | 10 | 43 |
| C:COMBO+ToMATo | 132 | 9 | 123 |
| A:MOMENT+Ward | 3 | 3 | 0 |
| B:FE+Ward | 12 | 7 | 5 |
| C:COMBO+Ward | 3 | 3 | 0 |

In tutte le 6 configurazioni post-merge, **100% delle coppie sono statisticamente distinte** (p < 0.05).

---

## 6. RISULTATI COMPLETI

### 6.1 Tabella confronto principale

| Config | K | η² | Sil | DB | CH | Dunn | Trans% | Sojourn | Cramér | Tukey | bARI |
|--------|---|-----|-----|-----|-----|------|--------|---------|--------|-------|------|
| A:MOMENT+ToMATo | 8 | 0.175 | -0.053 | 3.06 | 800 | 0.027 | 43.0% | 14h | 0.394 | 28/28 | 0.17±0.02 |
| A:MOMENT+Ward | 3 | 0.110 | 0.492 | 0.71 | 7366 | 0.410 | 5.1% | 118h | 0.650 | 3/3 | 0.54±0.18 |
| **B:FE+ToMATo** | **10** | **0.433** | 0.017 | 2.87 | 287 | 0.024 | **2.7%** | **222h** | 0.416 | **45/45** | 0.28±0.01 |
| B:FE+Ward | 7 | 0.282 | 0.248 | 1.29 | 709 | 0.058 | 1.0% | 585h | 0.393 | 21/21 | 0.41±0.12 |
| C:COMBO+ToMATo | 9 | 0.378 | -0.184 | 8.52 | 478 | 0.016 | 21.9% | 27h | 0.386 | 36/36 | 0.18±0.03 |
| C:COMBO+Ward | 3 | 0.211 | 0.589 | 0.76 | 8597 | 0.346 | 1.5% | 401h | 0.505 | 3/3 | 0.89±0.10 |

**Legenda metriche**:
- **η² (eta-squared)**: frazione della varianza totale del LMP spiegata dalla suddivisione in regimi (ANOVA). Più alto = regimi più distinti economicamente. Metrica primaria.
- **Sil (silhouette)**: misura se i cluster sono compatti e separati nello spazio Diffusion Maps. Valori negativi indicano sovrapposizione geometrica (possibile con cluster non convessi — non invalida ToMATo).
- **DB (Davies-Bouldin)**: rapporto dispersione/distanza tra centroidi. Più basso = meglio.
- **CH (Calinski-Harabasz)**: varianza inter/intra cluster. Più alto = meglio. Favorisce K basso.
- **Dunn**: min distanza inter-cluster / max diametro intra-cluster. Più alto = meglio.
- **Trans%**: frequenza di cambio regime tra finestre consecutive. Basso = regimi stabili nel tempo.
- **Sojourn**: permanenza media in un regime (ore). Alto = regimi persistenti (economicamente interpretabili come "periodi").
- **Cramér V**: associazione regime ↔ stagione. Alto = forte contenuto stagionale.
- **Tukey**: coppie di regimi con media LMP statisticamente diversa (Tukey HSD, α=0.05).
- **bARI**: bootstrap Adjusted Rand Index (20 run, 80% subsample). Misura quanto il clustering è stabile a perturbazioni del campione. 1 = perfettamente stabile, 0 = casuale.

#### Lettura dei risultati

**B:FE+ToMATo** è la configurazione migliore per la detection di regimi economici:
- η² più alto (0.433): i regimi spiegano il 43% della varianza del prezzo
- K=10: struttura ricca con 10 regimi interpretabili
- Trans%=2.7%: i regimi sono molto stabili (cambiano raramente)
- Sojourn=222h (~9 giorni): i regimi durano in media 9 giorni — coerente con la nozione di "periodo di mercato"
- Tukey 45/45: tutte le 45 coppie di regimi sono statisticamente distinte su LMP

**Ward** ha silhouette e bARI migliori ma K più basso (3-7): produce cluster geometricamente puliti ma economicamente meno informativi. In particolare, su MOMENT e COMBO (2D), Ward vede solo 3 regimi grossolani.

### 6.2 I 10 regimi B:FE+ToMATo (vincitore)

I regimi sono ordinati per LMP medio crescente e caratterizzati con fuel mix, stagionalità, e ILR.

| Regime | LMP medio | LMP std | n | % | Inv% | Pri% | Est% | Aut% | Gas% | Nuc% | Oil% | Sojourn | Interpretazione |
|--------|----------|---------|---|---|------|------|------|------|------|------|------|---------|-----------------|
| R5 | $26.7 | $8.1 | 807 | 11.2 | 0.9 | 64.2 | 19.6 | 15.4 | 52.8 | 28.7 | 0.0 | 255h | Off-peak primaverile |
| R0 | $33.2 | $16.8 | 1189 | 16.5 | 5.9 | 30.1 | 27.3 | 36.7 | 58.2 | 24.1 | 0.1 | 159h | Basso, autunnale |
| R6 | $39.2 | $23.6 | 852 | 11.8 | 18.5 | 17.8 | 33.1 | 30.5 | 55.9 | 26.0 | 0.2 | 128h | Sotto-media |
| R1 | $45.2 | $22.4 | 1231 | 17.1 | 1.8 | 25.2 | 28.8 | 44.2 | 56.7 | 27.2 | 0.2 | 321h | Baseload normale |
| R3 | $53.3 | $29.4 | 573 | 7.9 | 27.2 | 11.0 | 28.4 | 33.3 | 56.8 | 26.2 | 0.4 | 246h | Moderato |
| R2 | $65.0 | $36.7 | 779 | 10.8 | 22.5 | 23.5 | 42.0 | 12.1 | 57.6 | 25.5 | 0.5 | 360h | Domanda estiva |
| R4 | $75.2 | $40.6 | 574 | 8.0 | **54.7** | 21.6 | 14.6 | 9.1 | 51.8 | 28.7 | **0.9** | 164h | Domanda invernale |
| R8 | $83.0 | $40.8 | 309 | 4.3 | 24.6 | 8.4 | 28.8 | 38.2 | 57.4 | 26.8 | 0.8 | 206h | Stress |
| R7 | $91.3 | $59.0 | 483 | 6.7 | **71.6** | 16.4 | 12.0 | 0.0 | 49.2 | 29.2 | **2.8** | 414h | Spike invernale |
| **R9** | **$136.9** | **$51.0** | **417** | **5.8** | **93.8** | **6.2** | **0.0** | **0.0** | **50.9** | **27.9** | **4.4** | **626h** | **Spike estremo** |

#### Descrizione narrativa dei regimi

**R5 — Off-peak primaverile** ($27/MWh, 11.2% delle finestre). Il regime con prezzo più basso. Concentrato in primavera (64%) — il periodo in cui la domanda è bassa (temperature miti), il solare contribuisce di più, e l'idroelettrico è alimentato dal disgelo. Gas al minimo (53%), nucleare al massimo (29%). Volatilità molto bassa (std $8). Questo regime corrisponde al "surplus di generazione": l'offerta supera comodamente la domanda.

**R1 — Baseload normale** ($45/MWh, 17.1%). Il regime più frequente assieme a R0. Distribuito tra estate (29%) e autunno (44%), con quasi nessuna presenza invernale (2%). Rappresenta le condizioni "normali" del mercato: domanda coperta dal gas (57%) e nucleare (27%) senza stress. Sojourn 321h (~13 giorni) — il regime più persistente tra quelli non-estremi.

**R2 — Domanda estiva** ($65/MWh, 10.8%). Il principale regime estivo (42% estate). Prezzi elevati per la domanda di raffrescamento (AC), ma senza stress del sistema. Gas al massimo stagionale (58%), olio assente. Il LMP è alto perché la domanda termica estiva spinge il prezzo marginale.

**R4 — Domanda invernale** ($75/MWh, 8.0%). Primo regime con dominanza invernale (55%). L'olio inizia a comparire (0.9%) — segnale che la domanda si avvicina alla capacità disponibile. Gas scende al 52%: in inverno il gas è conteso tra riscaldamento e generazione elettrica.

**R7 — Spike invernale** ($91/MWh, 6.7%). Fortemente invernale (72%), autunno assente (0%). L'olio sale al 2.8% — le centrali a olio, le più costose, vengono attivate. Gas scende al 49%: il gas disponibile non basta, si ricorre a combustibili alternativi. Sojourn 414h (~17 giorni): gli spike invernali sono eventi prolungati.

**R9 — Spike estremo** ($137/MWh, 5.8%). Il regime più costoso. 94% inverno, 0% estate, 0% autunno. L'olio raggiunge il 4.4% — il massimo. Gas al 51%. Sojourn **626h = 26 giorni**: gli eventi estremi (cold snap prolungati, come il Bomb Cyclone di febbraio 2022) persistono per settimane. Questo regime cattura le emergenze del sistema quando la domanda di riscaldamento supera la capacità.

### 6.3 Pattern emergenti (senza supervisione)

Tutti i pattern seguenti emergono dal clustering senza essere stati specificati come obiettivo. Il clustering opera sui residui MSTL e sulle feature statistiche — non conosce le stagioni, il fuel mix, o il significato dell'olio.

| Pattern | Evidenza | Significato economico |
|---------|----------|----------------------|
| **Gradiente invernale** | R4→R7→R9: inverno 55%→72%→94%, LMP 75→91→137 | Il sistema passa gradualmente da stress moderato a emergenza. Non è un salto discreto: è una scala continua di severity |
| **Oil come stress indicator** | 0.0% (R5) → 0.9% (R4) → 2.8% (R7) → 4.4% (R9) | L'olio è il combustibile marginale in emergenza: il più costoso, il più inquinante, attivato solo quando tutto il resto non basta. La sua quota cresce monotonicamente con il livello di prezzo |
| **Gas inversamente correlato allo stress** | 58% (R0) → 49% (R7) | In condizioni normali il gas è il marginale dominante. In condizioni di stress il gas è conteso (riscaldamento vs generazione) e viene parzialmente sostituito da carbone e olio |
| **Nucleare stabile** | 24-29% in tutti i regimi | Il nucleare è baseload: produce a capacità costante indipendentemente dal prezzo. La variazione ±3% riflette outage programmati, non risposte al mercato |
| **Primavera = off-peak** | R5: 64% primavera, LMP $27 | Temperature miti, solare crescente, idro da disgelo, domanda bassa. Il "regime d'oro" per i consumatori |
| **R9 persistente** | Sojourn 626h = 26 giorni | Le emergenze non sono eventi puntuali: un cold snap che attiva il regime di spike può durare settimane. Informazione cruciale per hedging e risk management |
| **R8 bimodale (stress)** | 25% inverno + 29% estate + 38% autunno | Cattura sia cold snap sia heat wave. Entrambi causano stress per eccesso di domanda (riscaldamento e raffrescamento). L'unico regime con distribuzione stagionale trimodale |

### 6.4 Analisi ILR per regime

Le coordinate ILR descrivono la composizione del fuel mix in modo matematicamente rigoroso. La tabella seguente mostra le 3 coordinate più discriminanti tra i regimi.

| Regime | ilr_1 (disp/var) | ilr_3 (gas/solid) | ilr_7 (sol/wind) | Interpretazione composizionale |
|--------|------------------|-------------------|-------------------|-------------------------------|
| R5 (off-peak) | **-0.61** | **6.74** | -1.71 | Più variabili (rinnovabili), quasi tutto gas tra i fossili, mix solare/eolico |
| R1 (baseload) | +0.03 | 6.38 | -1.91 | Bilanciato, gas dominante, più eolico che solare |
| R4 (inverno) | +0.40 | 5.79 | **-2.45** | Più dispatchable, fossili solidi emergono, eolico domina (solare assente in inverno) |
| R7 (spike) | +0.53 | **5.46** | -2.43 | Fortemente dispatchable, carbone+olio significativi, tutto eolico |
| R9 (estremo) | **+1.69** | **4.10** | **-2.80** | Massimo dispatchable, massimo fossile solido, tutto eolico notturno/invernale |

**Osservazioni chiave**:
- **ilr_1** (dispatchable vs variable) varia da -0.6 a +1.7 tra regimi: in condizioni di stress il sistema dipende sempre più da fonti controllabili (gas, nucleare, carbone, olio) e meno da rinnovabili/idro
- **ilr_3** (gas vs solid fossil) scende da 6.7 a 4.1: nei regimi di spike, carbone e olio prendono quota rispetto al gas — il merit order spinge verso combustibili più costosi
- **ilr_7** (solar vs wind) diventa sempre più negativo: i regimi invernali sono dominati dall'eolico (il solare è quasi assente in inverno nel New England)

### 6.5 Cross-ARI tra rappresentazioni

|  | A: MOMENT | B: FE | C: COMBO |
|--|-----------|-------|----------|
| A: MOMENT | 1.000 | **0.070** | 0.144 |
| B: FE | 0.070 | 1.000 | 0.167 |
| C: COMBO | 0.144 | 0.167 | 1.000 |

L'ARI tra MOMENT e FE è **0.07** — vicino a zero, che corrisponde a indipendenza statistica. Le due rappresentazioni trovano struttura **quasi ortogonale** negli stessi dati. Questo non significa che MOMENT "fallisce": significa che cattura aspetti diversi della dinamica del mercato.

MOMENT identifica regimi basati sulla *forma* delle serie temporali (pattern di transizione, oscillazioni, persistenza non-lineare). FE identifica regimi basati sulle *statistiche aggregate* (livello di prezzo, dispersione, code). Per la detection di regimi definiti dal livello di prezzo, FE è superiore. Per analisi della dinamica temporale (es. stile di mean-reversion, pattern di risposta a shock), MOMENT potrebbe essere superiore — ma questa domanda è oltre lo scope del presente lavoro.

### 6.6 Matrice di transizione (B:FE+ToMATo)

La matrice di transizione P[i→j] mostra la probabilità di passare dal regime i al regime j tra finestre consecutive (stride 6h). La matrice è:

- **Fortemente diagonale**: i regimi sono persistenti (permanenza media 14-626h)
- **Band-like**: le transizioni avvengono prevalentemente tra regimi adiacenti in LMP — il mercato transita gradualmente, non salta da off-peak a spike estremo
- **Asimmetrica nei regimi di spike**: R9 ha auto-transizione P[9→9] altissima — una volta entrato nello spike estremo, il mercato ci resta a lungo

### 6.7 Sensitività (un fattore alla volta)

Ogni test varia un singolo fattore rispetto alla configurazione baseline (MOMENT, arcsinh, MSTL, 512h/6h), mantenendo tutto il resto identico. La pipeline completa (DiffMaps → ToMATo → Tukey) è rieseguita per ogni variante.

| Test | Fattore variato | Baseline | Variante | η² baseline | η² variante | K base | K var | Conclusione |
|------|-----------------|----------|----------|-------------|-------------|--------|-------|-------------|
| S1 | Trasformazione prezzo | arcsinh | log_return | **0.182** | 0.023 | 9 | — | arcsinh vince 8×. Log-return rimuove il livello |
| S2 | Destagionalizzazione | MSTL(24+168+8760) | nessuna | **0.182** | 0.189 | **9** | 4 | MSTL essenziale: senza, K=4 (stagioni, banale) |
| S3 | ILR nell'input | senza | con (7D concat) | **0.267** | 0.245 | 9 | 9 | ILR non migliora, peggiora leggermente |
| S4 | Contesto finestra | 512h | 72h | **0.161** | 0.089 | 8 | 21 | Contesto lungo cattura dipendenze non-lineari |
| S5 | Foundation model | MOMENT | Chronos-2 | — | R² negativo | — | — | Chronos-2 opaco: embedding non cattura struttura |

---

## 7. PUNTI DI ATTENZIONE

### 7.1 Criticità metodologiche

| # | Punto | Dettaglio | Gravità | Mitigazione | Residuo |
|---|-------|-----------|---------|-------------|---------|
| 1 | **MSTL annuale per LMP** | ACF annuale LMP = 0.11 (debole). Inclusa per consistenza con ILR (ACF=0.40) | Bassa | Test sensitività: Δstd < 0.002, risultati identici | Dichiarare la scelta nel paper |
| 2 | **Silhouette negativa** (MOMENT, COMBO) | -0.05 e -0.18. Cluster ToMATo non convessi in DiffMaps 2D | Media | ToMATo opera sulla densità, non sulla geometria euclidea. η² e Tukey confermano la validità | Discutere: silhouette non è metrica appropriata per cluster non convessi |
| 3 | **bARI moderato** per FE+ToMATo | 0.28±0.01. I bordi tra cluster cambiano nel bootstrap | Media | K è stabile (sempre 10); solo l'assegnazione ai bordi cambia. std=0.01 indica alta riproducibilità della *struttura* | Aumentare N bootstrap a 100 per CI più stretti |
| 4 | **Ward K=3** su MOMENT/COMBO | DiffMaps seleziona 2D → Ward trova solo 3 bacini convessi | Bassa | Finding legittimo: Ward non è adatto a manifold 2D con struttura non-convessa | Documentare come evidenza che ToMATo > Ward per TDA |
| 5 | **Overlap 97%** tra finestre | Stride 6h su 512h = 97% overlap. Autocorrelazione spaziale nei punti | Media | Non invalida il clustering (i punti sono ridondanti, non distorti). Sojourn calcolato correttamente (in unità di stride) | Dichiarare; confrontare con stride 24h come robustness check |
| 6 | **Tukey merge dipende da LMP** | Il merge usa il prezzo come criterio. Se la definizione di "regime" non è legata al prezzo, il merge è inappropriato | Media | Per definizione, i regimi di mercato elettrico *sono* definiti dal prezzo. Il merge è coerente con la domanda di ricerca | Discutere: per regimi non-economici servirebbero criteri diversi |
| 7 | **MOMENT non fine-tunato** | Zero-shot su dati ISONE. Il gap con FE potrebbe ridursi con fine-tuning | Media | Test fine-tuning: η²=0.185 vs 0.182 zero-shot — nessun miglioramento. Ma il fine-tuning era minimale (pochi epoch) | Lavoro futuro: fine-tuning sistematico con più dati |

### 7.2 Limitazioni dichiarabili nel paper

| # | Limitazione | Implicazione | Mitigazione possibile |
|---|-------------|-------------|----------------------|
| 1 | **Un solo mercato** (ISONE) | I risultati potrebbero non generalizzare a PJM (carbone dominante), ERCOT (eolico), o mercati europei (prezzi negativi frequenti) | Replicare su PJM e/o Nord Pool |
| 2 | **Periodo 5 anni** | Copre la transizione energetica 2021-2025 ma non cicli economici completi (recessioni, booms) | Estendere a 10 anni se dati disponibili |
| 3 | **Stride 6h e overlap** | Autocorrelazione spaziale: finestre consecutive condividono 506/512 valori | Test di robustezza con stride 24h |
| 4 | **MOMENT zero-shot** | Il foundation model non è stato adattato al dominio energetico | Fine-tuning sistematico o confronto con modelli energy-specific |
| 5 | **ILR non nel clustering** | Scelta basata su esperimento negativo, ma weighting diverso potrebbe funzionare | Testare COMBO con pesi ILR aumentati |
| 6 | **No forecasting** | I regimi non sono stati testati per previsione | HMM o regime-switching model sui 10 stati |
| 7 | **No causalità** | I regimi descrivono correlazioni (prezzo-fuel-stagione), non causalità | Complementare con modelli strutturali del merit order |

---

## 8. SVILUPPI FUTURI

### 8.1 Priorità alta (per il paper)

| Sviluppo | Motivazione | Sforzo |
|----------|-------------|--------|
| **Bootstrap ARI con N=100** (ora N=20) | Intervalli di confidenza più stretti | 30 min GPU |
| **Fine-tuning MOMENT** su dati ISONE | Testare se il gap FE-MOMENT si chiude | 2-4 ore GPU |
| **Validazione su PJM** | Secondo mercato, diverso fuel mix | 1 giorno (replica pipeline) |

### 8.2 Priorità media (estensioni)

| Sviluppo | Motivazione | Sforzo |
|----------|-------------|--------|
| **Regime forecasting** | Predire il regime futuro con features laggati | 1-2 settimane |
| **Temporal regime-switching model** | Hidden Markov Model con i K=10 stati | 1 settimana |
| **Multi-nodo ISONE** | Regimi spazialmente eterogenei (8 zone) | 2 settimane |
| **Interpretabilità MOMENT** | Quali neuroni/layer codificano i regimi? | Ricerca aperta |

### 8.3 Priorità bassa (esplorativo)

| Sviluppo | Motivazione |
|----------|-------------|
| **Chronos/TimesFM** | Altri foundation models per confronto |
| **Wavelet scattering** | Rappresentazione alternativa non-appresa |
| **ILR come input con weighting** | Dare più peso alle coordinate ILR nel COMBO |

---

## 9. STRUTTURA FILE E RIPRODUCIBILITÀ

```
nepool-tda/
├── config.py                         # Configurazione centrale
├── run_pipeline.py                   # Launcher (--from-step, --skip-moment)
├── isone_dataset.parquet             # Dataset raw
├── pipeline/
│   ├── step01_preprocessing.py       # MSTL multiprocessing
│   ├── step02_representations.py     # MOMENT + FE + COMBO
│   ├── step03_pca.py                 # Diffusion Maps
│   ├── step04_tomato.py              # ToMATo + Tukey
│   ├── step04_ward.py                # Ward + Tukey
│   ├── step05_cluster_quality.py     # Metriche + 7 plot/config
│   └── post_analysis.py              # Caratterizzazione, ILR, transizioni
├── results/
│   ├── preprocessed.parquet          # 43.795 × 20 colonne
│   ├── exp_C/                        # MOMENT
│   ├── exp_FE/                       # Feature Engineering (vincitore)
│   ├── exp_COMBO/                    # Combinato
│   ├── bootstrap_ari.json            # Stabilità 6 config
│   └── post_analysis/                # 6 plot + 3 CSV finali
├── paper/
│   └── sintesi.tex                   # Sintesi LaTeX compilabile
└── RECAP_SUPERVISORE.md              # Questo file
```

**Per riprodurre tutto da zero**: `python run_pipeline.py` (tempo: ~30 min locale, ~10 min Ray).

---

## 10. TEMPI DI ESECUZIONE

| Step | Tempo (locale) | Bottleneck |
|------|----------------|-----------|
| step01 MSTL | 3 min | LOESS con periodo 8760h (multiprocessing) |
| step02 MOMENT | 65s | GPU (CUDA) |
| step02 FE | 7s | CPU (loop Python) |
| step02 COMBO | 2s | PCA + concat |
| step03 DiffMaps (×3) | 55s | eigsh sparse |
| step04 ToMATo (×3) | ~22 min | Tukey HSD iterativo (COMBO: 132 cluster) |
| step04 Ward (×3) | 10s | linkage |
| step05 Quality (×6) | 30s | Plot matplotlib |
| post_analysis | 3s | Plot |
| **Totale** | **~30 min** | |
