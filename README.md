# Clasificare Companii in Domeniul Asigurarilor

## Overview

Acest proiect ofera o solutie pentru clasificarea companiilor in domeniul asigurarilor. Se folosesc tehnici de procesare a textului si vectorizare TF-IDF pentru a asocia fiecare companie cu una sau mai multe etichete dintr-o taxonomie predefinita.

## Structura Proiectului

### main.py
- **Incarca datele**: Citește datele dintr-un fisier CSV (companii) si dintr-un fisier Excel (taxonomie).
- **Preproceseaza textul**: Aplica lowercase si elimina semnele de punctuatie.
- **Combineaza coloanele relevante**: Creeaza un text unic din campurile relevante.
- **Vectorizeaza textul**: Foloseste TF-IDF pentru a transforma textul in vectori.
- **Calculeaza similaritatea**: Se calculeaza similaritatea cosine intre vectorii companiilor si cei ai etichetelor din taxonomie.
- **Atribuie etichete**: Pe baza unui prag de similaritate, fiecare companie primeste una sau mai multe etichete.
- **Salveaza rezultatele**: Rezultatele sunt salvate intr-un fisier CSV.

### evaluate_classifier.py
- **Incarca rezultatele**: Citeste fisierul de rezultate al clasificarii.
- **Analizeaza distributia etichetelor**: Se verifica frecventa fiecarei etichete atribuite.
- **Evalueaza scorurile de similaritate**: (Daca sunt disponibile) Afiseaza statistici despre scorurile maxime.
- **Revizuire manuala**: Afiseaza un ecran de sample pentru validare manuala.
- **Evaluare optionala**: Suporta evaluarea cu ground truth, daca exista un astfel de fisier.

## Cerinte

- **Python 3.x**

### Librarii necesare:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---
