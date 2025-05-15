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

### Zjišťovaní informací o tabulkách
```python
print(f"Počet trénovacích záznamů: {len(train_df)}")
print(f"Počet testovacích záznamů: {len(submission_df)}")

# Zobrazení informací o tabulce
print('\nZobrazení informací o trénovací tabulce:')
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

Unikátní tagy:

agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water
 
### Trénování dat a vytvoření klasifikátorů

#### Vytvoření one-hot encodingu pro tagy
```python
# Vytvoření one-hot encodingu pro tagy
def get_tag_map(tags):
    labels = np.zeros(len(unique_tags))
    for tag in tags.split():
        if tag in unique_tags:
            labels[unique_tags.index(tag)] = 1
    return labels
# Přidání one-hot encodingu do dataframe
train_df['tag_vector'] = train_df['tags'].apply(get_tag_map)
```
#### Vytvoření dataset třídy pro PyTorch 
Abych mohl udělat trénovací model pro Resnet50, tak si musím vytvořit dataset třídu pro PyTorch.
```python
# Vytvoření vlastní Dataset třídy pro PyTorch
class PlanetDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        # Načtení obrázku
        image = Image.open(img_path).convert('RGB')
        # Aplikace transformací, pokud jsou definované
        if self.transform: image = self.transform(image)
        # Získání one-hot encodingu pro tagy
        tag_vector = torch.FloatTensor(self.dataframe.iloc[idx]['tag_vector'])
        return image, tag_vector
```
#### Definování transformace
```python
# Definování transformací pro obrázky
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
#### Rozdělení na trénovací data a jejich připracení pro trénování modelu
```python
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"\nPočet trénovacích vzorků: {len(train_data)}")
print(f"Počet validačních vzorků: {len(valid_data)}")
# Vytvoření datasetů
train_dataset = PlanetDataset(train_data, TRAIN_DIR, transform=transform)
valid_dataset = PlanetDataset(valid_data, TRAIN_DIR, transform=transform)
# Vytvoření dataloaderů
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# Ukázka načtení jedné dávky dat
images, labels = next(iter(train_loader))
print(f"\nTvar načtených obrázků: {images.shape}")
print(f"Tvar načtených tagů: {labels.shape}")
# Vizualizace několika obrázků z datasetu
def visualize_sample(dataset, num_samples=5):
    plt.figure(figsize=(15, 3*num_samples))
    for i in range(num_samples):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        # Denormalizace obrázku pro zobrazení
        image = std * image + mean
        image = np.clip(image, 0, 1)
        tags = [unique_tags[j] for j in range(len(unique_tags)) if label[j] == 1]
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(image)
        plt.title(f"Tagy: {', '.join(tags)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# Hodnoty pro denormalizaci obrázků
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Zakomentujte následující řádek, pokud nechcete zobrazit ukázkové obrázky
# visualize_sample(train_dataset)
print("\nData jsou připravena pro trénování modelu!")
```
#### Použití předtrénovaného modelu
```python
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Použití předtrénovaného modelu
model = models.resnet50(pretrained=True)
# Úprava poslední vrstvy pro multi-label klasifikaci
model.fc = nn.Linear(model.fc.in_features, len(unique_tags))

# Definice ztráty a optimizéru
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Přesun modelu na GPU, pokud je dostupná
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
