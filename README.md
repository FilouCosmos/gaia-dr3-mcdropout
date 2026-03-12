<img width="700" height="371" alt="image" src="https://github.com/user-attachments/assets/fe550c00-5932-4a11-9717-02e9b0d11d18" />

# Prédiction Galactique par Deep Learning (MC Dropout)

Ce projet de Data Science / IA vise à prédire la **Vitesse Radiale** manquante de millions d'étoiles au sein de la Voie Lactée, en utilisant les données massives du catalogue **Gaia DR3** de l'Agence Spatiale Européenne (ESA). 

Face à l'enjeu de fiabilité des données spatiales, l'approche retenue est une **Régression Profonde Probabiliste** basée sur le **Monte Carlo Dropout** développée avec **PyTorch**.

## Enjeux et Valeur Métier

* **Complétion de données (Missing Value Imputation) :** Le spectromètre de Gaia n'a mesuré la vitesse radiale que pour environ 3% des étoiles les plus brillantes. L'IA permet ici d'estimer la vitesse d'éloignement ou de rapprochement (vitesse 3D) à partir des données 2D disponibles.
* **Ingénierie des incertitudes :** Contrairement à une IA classique qui fournit une valeur fixe, ce modèle "sait quand il ne sait pas". Il fournit une marge d'erreur ($\sigma$) pour chaque prédiction, une donnée cruciale pour la recherche scientifique et la fiabilité des modèles.
* **Feature Engineering Astrophysique :** Transformation des données brutes (parallaxe, magnitude apparente) en variables physiques réelles (distance en parsecs, magnitude absolue, mouvement propre total).
* **Full-Stack ML & Infra :** Mise en place d'un pipeline complet allant du requêtage **ADQL** sur les serveurs de l'ESA à l'entraînement accéléré sur GPU (**CUDA**) dans un environnement Lab local.

## Architecture de l'Intelligence Artificielle

Le modèle repose sur un réseau de neurones profond de type **Perceptron Multicouche (MLP)** conçu pour la régression complexe.
* **Architecture :** 4 couches denses (64, 128, 64 neurones) utilisant la fonction d'activation ReLU.
* **Monte Carlo Dropout :** Intégration de couches de Dropout ($p=0.2$) qui restent **actives lors de l'inférence**. 
* **Inférence Probabiliste :** En effectuant $T=100$ passages pour chaque étoile, le modèle génère une distribution de probabilité permettant de calculer la moyenne (la prédiction finale) et l'écart-type (l'indice de confiance).

## Fonction de Perte et Optimisation

L'entraînement du modèle minimise l'Erreur Quadratique Moyenne (MSE) :
$$L = \text{MSE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Optimiseur :** Adam avec un Learning Rate de $0.005$.
* **Accélération Matérielle :** Entraînement sur carte graphique (CUDA/GPU) pour une gestion efficace des tenseurs sur 100 000 échantillons.

## Résultats et Visualisation

Le modèle a été évalué sur un échantillon de test (20% des données). 
1.  **Compréhension Physique :** Les prédictions suivent fidèlement la réalité (vérité terrain), prouvant que l'IA a capturé les corrélations entre la couleur, la luminosité et la cinématique stellaire.
2.  **Analyse de l'Incertitude :** Le graphique final met en évidence les barres d'erreur ($\sigma$) issues du MC Dropout. Une barre large indique que la physique de l'étoile est atypique pour le modèle, prévenant ainsi les erreurs d'interprétation.
3.  **Régression vers la moyenne :** L'analyse montre que le modèle reste prudent sur les vitesses extrêmes (anomalies à hautes vitesses), un comportement de sécurité classique en apprentissage supervisé.

## Stack Technique

* **Langage :** Python 3
* **Deep Learning :** PyTorch (MCDropoutNet, CUDA/GPU)
* **Data Manipulation :** Pandas, Scikit-Learn (StandardScaler), Numpy
* **Collecte de données :** Astroquery (API Gaia de l'ESA)
* **Visualisation :** Matplotlib, Seaborn

## Reproductibilité (Installation Locale)

Pour exécuter ce projet sur votre machine avec prise en charge du GPU :

1. **Clonez ce dépôt :**
```bash
git clone [https://github.com/VOTRE_PSEUDO/gaia-radial-velocity-mcdropout.git](https://github.com/VOTRE_PSEUDO/gaia-radial-velocity-mcdropout.git)
cd gaia-radial-velocity-mcdropout
