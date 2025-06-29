# Projet Python IA inspiré de W3Schools : Prédiction du prix des maisons

Ce projet est une implémentation simple de la régression linéaire pour prédire le prix des maisons en fonction de leur taille (superficie). Il utilise des concepts d'intelligence artificielle introduits sur W3Schools, notamment avec les bibliothèques `NumPy`, `SciPy` et `Matplotlib`. Une comparaison est également faite avec `scikit-learn` pour une approche plus standard en machine learning.

## Fonctionnalités
* Génération de données fictives sur la taille et le prix des maisons.
* Visualisation des données brutes.
* Application de la régression linéaire en utilisant `scipy.stats.linregress`.
* Application de la régression linéaire en utilisant `sklearn.linear_model.LinearRegression`.
* Visualisation des données avec les droites de régression des deux modèles.
* Exemples de prédictions pour de nouvelles tailles de maisons.

## Installation
1.  Clonez ce dépôt :
    ```bash
    git clone [https://github.com/Atae-AlNahlaouy/PREMIER-PROJET-PYTHON.git](https://github.com/Atae-AlNahlaouy/PREMIER-PROJET-PYTHON.git)
    cd PREMIER-PROJET-PYTHON
    ```
2.  Créez et activez un environnement virtuel :
    ```bash
    py -m venv venv
    # Sur Windows
    .\venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```
3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation
Exécutez le script principal :
```bash
py prediction_maison.py