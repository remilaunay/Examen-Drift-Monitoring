# Surveillance du service de vélos avec Evidently

Projet d’examen : entraîner un modèle de régression sur janvier 2011, simuler sa mise en production sur février, puis conserver les rapports et snapshots dans un workspace Evidently.

## Lancer le projet

### Avec Docker Compose

Depuis la racine du dépôt :

```bash
docker compose up --build
```

Le service `monitor-script` télécharge le jeu UCI à son premier lancement, calcule les rapports et enregistre les snapshots. Une fois ce service terminé, `monitor-ui` démarre. Ouvrir ensuite <http://localhost:8000> pour consulter le projet **Bike Rental Monitoring**.

Pour arrêter l’interface :

```bash
docker compose down
```

Le script monte le dépôt dans le conteneur afin que le workspace et le rapport HTML restent disponibles sur la machine hôte. L’interface partage le dossier `bike_sharing_monitoring/` avec le script.

### En local avec `uv`

Python 3.12 est utilisé par le conteneur et le venv local. À la racine du dépôt :

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/main.py
```

On peut aussi activer l’environnement avant de lancer le script :

```bash
source .venv/bin/activate
python src/main.py
```

Le premier lancement récupère les données avec `ucimlrepo` et crée `data/bike_sharing.csv`. Les lancements suivants réutilisent ce cache. Les rapports de production s’ajoutent au workspace existant.

## Structure du projet

```text
.
├── .dockerignore
├── .gitignore
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── .gitkeep
│   └── bike_sharing.csv              # créé au premier lancement, ignoré par Git
├── src/
│   └── main.py                       # logique complète du traitement
├── bike_sharing_monitoring/          # workspace et snapshots générés, ignorés par Git
└── diagnostic_report.html            # export généré, ignoré par Git
```

Les données téléchargées, le venv, le workspace Evidently et l’export HTML ne sont pas versionnés : ils sont régénérés par le script ou Compose. Le `.gitignore` garde le dépôt léger sans exclure le code ni les fichiers de configuration nécessaires.

## Démarche effectuée

1. Récupération du dataset UCI Bike Sharing (id 275) avec `fetch_ucirepo`, assemblage des variables et de la cible `cnt`, puis création d’un cache CSV local.
2. Filtrage des observations datées de janvier et février 2011. Janvier sert de référence et février de production simulée.
3. Découpage aléatoire reproductible de janvier en apprentissage (70 %) et validation (30 %). Les variables catégorielles codées sont encodées par `OneHotEncoder`; les variables météo continues restent numériques. Le modèle est un `RandomForestRegressor`.
4. Évaluation du test de janvier avec `RegressionPreset` d’Evidently et un test intégré exigeant `RMSE < 50`.
5. Découpage chronologique de février en trois fenêtres de taille comparable : W1 du 1 au 10, W2 du 11 au 19 et W3 du 20 au 28. Chaque lot reçoit des prédictions et un snapshot de performance.
6. Sélection de la fenêtre au RMSE le plus élevé, puis calcul dans un même rapport Evidently de la dérive de la cible (`ValueDrift` sur `cnt`) et de la dérive des variables météo (`DataDriftPreset`). Le snapshot est ajouté au workspace et exporté en `diagnostic_report.html`.

Le nombre de lignes varie selon les fenêtres, car certaines heures sont absentes du jeu UCI. Le script répartit les jours observés de février en trois groupes chronologiques équilibrés.

## Résultats et réponses aux questions

### Performance

La validation sur le test de janvier obtient un RMSE de **21,12 vélos**, sous le seuil de 50. Les résultats par fenêtre de février sont :

| Période | Jours | Lignes | RMSE | MAE |
|---|---:|---:|---:|---:|
| W1 | 1–10 février | 235 | 23,91 | 17,02 |
| W2 | 11–19 février | 211 | 43,57 | 34,03 |
| W3 | 20–28 février | 203 | **45,57** | 29,75 |

En prenant le RMSE comme mesure principale, cohérente avec le test de validation, **W3 est la période la plus dégradée**. Son RMSE est environ 2,16 fois celui du test de janvier. W2 est proche et a le MAE le plus élevé. Le seuil absolu `RMSE < 50` passe encore en W3 : seul, il ne détecte donc pas assez tôt cette dégradation relative.

### Dérive et météo

Le snapshot de W3 compare les observations à la référence de janvier. Evidently détecte une dérive sur **4 des 5 variables météo** examinées :

| Variable | p-value | Dérive détectée |
|---|---:|---|
| `weathersit` | 0,000111 | oui |
| `temp` | 3,04 × 10⁻³⁶ | oui |
| `atemp` | 1,75 × 10⁻³² | oui |
| `hum` | 0,00677 | oui |
| `windspeed` | 0,0622 | non |

Les températures normalisées moyennes montent de **0,197 à 0,301** (`temp`) et de **0,204 à 0,298** (`atemp`) entre janvier et W3. L’humidité moyenne passe de 0,574 à 0,612. La proportion d’heures classées `weathersit = 3` passe de 7,1 % en janvier à 15,3 % en W3. En moyenne, la demande réelle de W3 est de 75,72 vélos, contre 64,61 prédits : le modèle sous-estime la demande d’environ 11 vélos par heure.

La distribution de la cible `cnt` dérive aussi (K-S, p = 0,00112). La demande moyenne est de 75,72 vélos en W3, contre 55,51 en janvier. Le déplacement des températures et de la demande suggère un effet saisonnier que le modèle, entraîné uniquement sur janvier, représente mal. L’humidité et la fréquence plus élevée du niveau météo 3 peuvent aussi contribuer aux écarts. Les tests de dérive montrent une association entre ces changements de distribution et la fenêtre dégradée, mais ne prouvent pas à eux seuls une causalité.

### Remédiation proposée

En production, je déclencherais une **alerte de dérive et une réévaluation du modèle**, puis entraînerais un candidat avec plusieurs mois de données couvrant les saisons et les conditions météo. Le candidat ne remplacerait le modèle actif qu’après validation sur un jeu temporel récent et représentatif. Une alerte basée sur la hausse relative du RMSE, persistante sur plusieurs lots, compléterait le seuil absolu actuel. L’arrêt du service serait disproportionné pour une erreur de prévision de demande; la surveillance et le remplacement contrôlé du modèle sont plus adaptés.
