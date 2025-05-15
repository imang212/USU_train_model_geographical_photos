# Trénovací model na geografická data
Rozhodl jsem se udělat trénovací model používající strojové učení na rozpoznávání geografických dat v rámci seminární práce na USU. 

### Dataset
Budu pracovat z datasetem ze stránek kaggle.com (planets_dataset), který obsahuje přibližně 40 000 testovacích, 40 000 trénovacích obrázků terénu amazonského pralesa a obsahuje soubory csv formátu s popisem testovacích a trénovacích obrázků. Datasetem geografických obrázku vyfocených v Amazonském pralese, který obsahuje 2 csv soubory s tabulkami, které popisují, co na obrázcích je pro strojové učení. Dataset můžu volně použít ke strojovému učení na základě license Database Contents License, která to povoluje. Odkaz k datasetu: https://www.kaggle.com/datasets/nikitarom/planets-dataset/data

Pojďme si zobrazit kořenovou strukturu datasetu.:

![image](https://github.com/user-attachments/assets/1fab1cc9-de39-4a88-a92f-6ae9f64b1def)

První csv tabulka s názvem *sample_submission.csv* obsahuje 61191 hodnot, obsahuje 2 sloupce:
image_name - název daného obrázku
tags - popis vlastností toho co je na obrázcích.

Druhá csv tabulka s názvem *train_classes.csv* obsahuje 40 479 hodnot obsahuje také 2 sloupce:
image_name - název daného obrázku
tags - popis vlastností toho co je na obrázcích např.: clear_primary, clear_cloudy_primary, atd...

Máme adresáře pro obrázky:

*test-jpg*, kde se nachází testovací obrázky, je jich přibližně 40 000. 

*train-jpg*, kde se nachází trénovací obrázky, je jich přibližně 40 000.

Máme ještě adresář *test-jpg-additional*, kde se nachází testovací ještě nachází přibližně 20 500 testovacích obrázků navíc. 

### Technologie

### Načtení dat pro trénovací model

```python
#import potřebných datasetů
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Nastavení cest k datům
DATA_DIR = './nikitarom/planets-dataset/versions/3/'
TRAIN_DIR = os.path.join(DATA_DIR, 'planet/planet/train-jpg')
TEST_DIR = os.path.join(DATA_DIR, 'planet/planet/test-jpg')
TRAIN_CLASSES = os.path.join(DATA_DIR, 'planet/planet/train_classes.csv')
SUBMISSION = os.path.join(DATA_DIR, 'planet/planet/sample_submission.csv')

# Načtení CSV souborů
train_df = pd.read_csv(TRAIN_CLASSES)
submission_df = pd.read_csv(SUBMISSION)
```
Naimportoval jsem potřebné datasety pro práci s daty. Určil jsem si cesty k adresářům s trénovací a testovacími obrázky a s popisem obrázků. Načetl jsem si csv soubory.

#### Zjišťovaní informací o tabulce
```python
print(f"Počet trénovacích záznamů: {len(train_df)}")
print(f"Počet testovacích záznamů: {len(submission_df)}")
# Zobrazení informací o tabulce
print("Hlavička tabulky: ", train_df.head(), '\n')
print("Informace o tabulce: ", train_df.info(), '\n')
print("Nulové hodnoty: ", train_df.isnull().sum(), '\n')

# Zobrazme si distribuci tagů ve sloupci 'tags'
all_tags = []
for tags in train_df['tags'].values:
    all_tags.extend(tags.split())
unique_tags = sorted(list(set(all_tags)))
print(f"\nPočet unikátních tagů: {len(unique_tags)}")
print(f"Unikátní tagy: {unique_tags}")
```
Vypsal jsem si informace o tabulce, abych věděl kolik s ní je záznamů, jak má nastavené sloupce, jestli tam například jsou povoleny nulové hodnoty a jakého typu jsou. Také jsem si vypsal informace o tom, jestli obsahuje nějaké nulové hodnoty a kolik je tagů ve sloupci *tags*, kde jsem zjistil, že se vyskytuje 17 různých tagů pro popis prostředí obrázků. 

Unikátní tagy: agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water 

