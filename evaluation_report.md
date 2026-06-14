# Robustness Testing and Evaluation Report (After Fixes)

This report presents the findings from running robustness tests on the updated fact-checking codebase:
1. **Fuzzy KB Lookup**
2. **LinearSVC Baseline (TF-IDF)**
3. **DistilBERT Sequence Classifier (Transformer)**


### 1. Control Group (General Health Myths/Facts)
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "Eating fruits and vegetables provides important vitamins and minerals." | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (51.0%) |
| "Vaccines cause autism." | FALSE | **FALSE** | **FALSE** | **FALSE** (77.6%) |
| "Antibiotics cure bacterial infections but do not work on viruses." | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | **TRUE** (99.6%) |
| "Drinking lemon water cures cancer." | FALSE | **FALSE** | **FALSE** | ~~TRUE~~ (FAIL) (97.7%) |
| "Smoking increases the risk of cancer and lung disease." | TRUE | **TRUE** | **TRUE** | **TRUE** (99.5%) |

### 2. Out of Dataset (OOD) Queries
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "Sleeping on your left side prevents acid reflux." | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (44.2%) |
| "Drinking bleach cures Covid-19." | FALSE | **FALSE** | **FALSE** | ~~TRUE~~ (FAIL) (99.7%) |
| "Consuming raw garlic eliminates all intestinal parasites." | FALSE | None | **FALSE** | **FALSE** (70.1%) |
| "Cancer is caused by positive thoughts and can be cured by meditation." | FALSE | **FALSE** | **FALSE** | ~~UNPROVEN~~ (FAIL) (60.0%) |

### 3. Misspelling Queries
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "vacines cause autism" | FALSE | **FALSE** | **FALSE** | ~~UNPROVEN~~ (FAIL) (69.8%) |
| "drnking lemon juice can treat cancr" | FALSE | **FALSE** | ~~UNPROVEN~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (47.6%) |
| "smokng causes lung desease" | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (61.9%) |
| "wash hands prevents infetcion" | TRUE | **TRUE** | ~~UNPROVEN~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (47.6%) |

### 4. Taglish (Filipino Elderly Dialect)
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "Ang vaccines ay nagdudulot ng autism sa mga bata." | FALSE | **FALSE** | **FALSE** | **FALSE** (77.3%) |
| "Uminom ng mainit na lemon water para gumaling sa cancer." | FALSE | **FALSE** | **FALSE** | ~~TRUE~~ (FAIL) (95.9%) |
| "Masama sa kidney ang uminom ng maraming softdrinks." | TRUE | None | ~~MIXTURE~~ (FAIL) | **TRUE** (99.5%) |
| "Washing hands bago kumain ay nakakaiwas sa sakit." | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (39.1%) |

### 5. Emoji Queries
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "Vaccines cause autism 💉❌" | FALSE | **FALSE** | **FALSE** | ~~UNPROVEN~~ (FAIL) (69.8%) |
| "Drinking lemon water cures cancer 🍋" | FALSE | **FALSE** | **FALSE** | ~~UNPROVEN~~ (FAIL) (60.8%) |
| "Washing your hands helps prevent infections 🧼🙌" | TRUE | **TRUE** | ~~MIXTURE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (51.5%) |

### 6. Numbers / Abbreviations
| Claim | Expected Label | Static KB | LinearSVC Baseline | DistilBERT Classifier (Confidence) |
| :--- | :---: | :---: | :---: | :---: |
| "wash hands b4 eating" | TRUE | **TRUE** | ~~FALSE~~ (FAIL) | ~~UNPROVEN~~ (FAIL) (33.2%) |
| "drinking lemon juice can cure cancer w/o chemo" | FALSE | **FALSE** | **FALSE** | ~~UNPROVEN~~ (FAIL) (44.7%) |
| "too much sugar increases obesity risk & tooth decay 2" | TRUE | **TRUE** | **TRUE** | ~~UNPROVEN~~ (FAIL) (38.4%) |
