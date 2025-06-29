import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression # Alternative plus standard pour ML

# --- 1. Génération de données (similaire à un dataset réel) ---
# Supposons que la taille est en pieds carrés (sqft) et le prix en milliers de dollars.
# Ces données sont fictives pour l'exemple.
taille_maison = np.array([500, 750, 1000, 1200, 1500, 1750, 2000, 2200, 2500, 2800])
prix_maison = np.array([150, 200, 250, 280, 320, 360, 400, 420, 480, 520])

# Ajouter un peu de bruit aux données pour qu'elles ne soient pas parfaitement linéaires
prix_maison = prix_maison + np.random.normal(0, 15, prix_maison.shape)

# --- 2. Visualisation des données brutes ---
plt.scatter(taille_maison, prix_maison)
plt.title("Prix des maisons en fonction de la taille (Données brutes)")
plt.xlabel("Taille de la maison (pieds carrés)")
plt.ylabel("Prix de la maison (milliers de $)")
plt.grid(True)
plt.show()

# --- 3. Régression Linéaire avec SciPy ---
# Calcul de la pente (slope) et de l'ordonnée à l'origine (intercept)
slope, intercept, r_value, p_value, std_err = stats.linregress(taille_maison, prix_maison)

# Fonction de prédiction basée sur le modèle linéaire de SciPy
def predict_scipy(taille):
    return slope * taille + intercept

# Générer des valeurs y prédites pour la droite de régression
modele_scipy_y = predict_scipy(taille_maison)

print(f"\n--- Modèle de Régression Linéaire (SciPy) ---")
print(f"Pente (slope): {slope:.2f}")
print(f"Ordonnée à l'origine (intercept): {intercept:.2f}")
print(f"Coefficient de corrélation (R): {r_value:.2f}") # Indique la force de la relation linéaire

# --- 4. Régression Linéaire avec Scikit-learn (approche plus robuste pour ML) ---
# Scikit-learn attend une entrée 2D pour les features
taille_maison_reshaped = taille_maison.reshape(-1, 1) # Convertir en colonne unique

model_sklearn = LinearRegression()
model_sklearn.fit(taille_maison_reshaped, prix_maison)

# Obtenir la pente et l'ordonnée à l'origine de Scikit-learn
slope_sklearn = model_sklearn.coef_[0]
intercept_sklearn = model_sklearn.intercept_

# Fonction de prédiction basée sur le modèle de Scikit-learn
def predict_sklearn(taille):
    # Scikit-learn's predict method expects a 2D array, even for single prediction
    return model_sklearn.predict(np.array([[taille]]))[0]


print(f"\n--- Modèle de Régression Linéaire (Scikit-learn) ---")
print(f"Pente (slope): {slope_sklearn:.2f}")
print(f"Ordonnée à l'origine (intercept): {intercept_sklearn:.2f}")
print(f"Score R^2 (Scikit-learn): {model_sklearn.score(taille_maison_reshaped, prix_maison):.2f}") # R-squared score

# Générer des valeurs y prédites pour la droite de régression de Scikit-learn
modele_sklearn_y = model_sklearn.predict(taille_maison_reshaped)

# --- 5. Visualisation des données avec les droites de régression ---
plt.scatter(taille_maison, prix_maison, label="Données réelles")
plt.plot(taille_maison, modele_scipy_y, color='red', label=f"Régression SciPy (R={r_value:.2f})")
plt.plot(taille_maison, modele_sklearn_y, color='green', linestyle='--', label=f"Régression Sklearn (R2={model_sklearn.score(taille_maison_reshaped, prix_maison):.2f})")
plt.title("Régression Linéaire : Prix des maisons vs Taille")
plt.xlabel("Taille de la maison (pieds carrés)")
plt.ylabel("Prix de la maison (milliers de $)")
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Exemple de prédiction ---
taille_nouvelle_maison = 1850 # en pieds carrés

prix_predit_scipy = predict_scipy(taille_nouvelle_maison)
print(f"\nPrédiction SciPy : Une maison de {taille_nouvelle_maison} sqft coûterait environ ${prix_predit_scipy:.2f} milliers.")

prix_predit_sklearn = predict_sklearn(taille_nouvelle_maison)
print(f"Prédiction Scikit-learn : Une maison de {taille_nouvelle_maison} sqft coûterait environ ${prix_predit_sklearn:.2f} milliers.")

# Exemple de prédiction hors de la plage des données d'entraînement (attention !)
taille_grande_maison = 3500
print(f"\nPrédiction pour une très grande maison ({taille_grande_maison} sqft) :")
print(f"  SciPy: ${predict_scipy(taille_grande_maison):.2f} milliers")
print(f"  Scikit-learn: ${predict_sklearn(taille_grande_maison):.2f} milliers")