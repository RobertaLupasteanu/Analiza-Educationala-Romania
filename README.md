# Analiză Educațională România
 O aplicație Streamlit interactivă pentru explorarea, modelarea și vizualizarea datelor școlare pe județe în România (2019-2024).
Scopul este de a oferi o platformă completă de explorare, modelare și vizualizare a factorilor ce influențează performanța și abandonul școlar.

# Conținutul setului de date

- Elevi existenți: numărul de elevi înscriși anual (2019-2020 până în 2023-2024).
- Rate de abandon școlar calculate pe județe.
- Grupe de varsta asociate pe nivele de invatatura.
- Date demografice (populație, distribuție urban/rural).
- Date geografice: fișier Shapefile pentru harta României.
- Date categoriale: Tip invatamant, Tip finantare, Mod predare, Limba predare etc.

# Structura aplicației

- App.py: Pagina principală cu:
  - Navigare între module și rezumatul fiecăruia.
  - Previziualizare set de date.
- data_loader.py: funcții auxiliare care încarcă, curăță și pregătesc datele pentru întregul workflow.
- utils.py: Funcții auxiliare pentru normalizarea denumirilor de județe, codificări și alte transformări.
- pages/: Module separate pentru analiza fiecărei etape:
  - 1_analiza_exploratorie.py – vizualizări si analize descriptive, hărți și distribuții.
  - 2_analiza_predictiva.py – previzualizare pentru analizele predictive.
  - 3_analiza_cluster.py – clusterizare KMeans și interpretarea clustere.
  - 4_analiza_regresiei.py – regresie multiplă și inferență.
  - 5_analiza_de_clasificare.py – clasificare a elevilor în categorii de risc prin regresie logistică și arbore de clasificare.
