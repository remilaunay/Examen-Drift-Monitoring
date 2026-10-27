Examen : Cas d'étude sur le flux de travail de surveillance
Dans cet examen, vous travaillerez sur le célèbre ensemble de données UCI sur le vélo en libre-service, disponible ici (https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

Les systèmes de vélos en libre-service sont des versions modernisées des systèmes traditionnels de location de vélos. Ces systèmes automatisent l'ensemble du processus, permettant aux utilisateurs de louer facilement des vélos dans des lieux spécifiques et de les rendre ailleurs. Il existe plus de 500 programmes de partage de vélos dans le monde.

Contrairement à d'autres services de transport, les données relatives au vélo en libre-service enregistrent explicitement la durée du trajet, les positions de départ et d'arrivée, ce qui les rend précieuses pour la recherche et le suivi de la mobilité urbaine.

Voir la documentation pour une description plus détaillée de l'ensemble de données.

Le but de cet examen est d'utiliser evidently pour construire un espace d'observabilité persistant. Vous devrez valider techniquement un modèle de régression, puis simuler son monitoring en production sur les mois de janvier et février 2011 en utilisant les concepts de Workspace, de Snapshots et de Reports (incluant des tests intégrés).

1. Structure de projet attendue
Pour réussir cet examen, votre rendu doit suivre strictement cette structure :
```
    examen_evidently/
    ├── data/                     # Dossier pour les données téléchargées
    ├── src/
    │   └── main.py               # Script unique contenant toute la logique
    ├── bike_sharing_monitoring/  # Workspace (généré par le script)
    ├── Dockerfile                # Pour le service monitor-script
    ├── docker compose            # Orchestration Script + UI
    ├── requirements.txt          # Dépendances (evidently==0.7.2, scikit-Learn==1.8.0, etc.)
    └── README.md                 # Réponses aux questions d'analyse
```

2. Méthodologie détaillée
Phase 1 : Le script principal (main.py)
Votre script doit s'exécuter de bout en bout et réaliser les actions suivantes :

Ingestion : Télécharger et extraire le dataset UCI Bike Sharing.
```
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    bike_sharing = fetch_ucirepo(id=275) 
    
    # data (as pandas dataframes) 
    X = bike_sharing.data.features 
    y = bike_sharing.data.targets 
    
    # metadata 
    print(bike_sharing.metadata) 
    
    # variable information 
    print(bike_sharing.variables) 
```

Préparation : Créer deux DataFrames : reference (Janvier 2011) et production (Février 2011).
Entraînement & Validation :
Split 70/30 sur les données de Janvier.
Entraîner un RandomForestRegressor.
Exécuter un Report Evidently avec des tests de validation technique (Validation technique) sur le set de test de Janvier.
Initialisation Workspace : Créer le Workspace local et le projet "Bike Rental Monitoring".
Simulation de Production (Monitoring) :
Découper le mois de Février en 3 semaines (W1, W2, W3).
Pour chaque semaine, générer les prédictions et enregistrer un Snapshot du RegressionPreset dans le projet.
Diagnostic Final :
Générer un snapshot TargetDrift et DataDrift (numérique) pour la semaine la plus dégradée.
Exporter ce rapport en HTML (diagnostic_report.html).
Phase 2 : Conteneurisation
Vous devez encapsuler votre travail pour qu'il soit reproductible :

Dockerfile : Doit copier votre code, installer les requirements.txt et lancer main.py.
compose.yaml : Doit orchestrer deux services :
monitor-script : Le service qui exécute la logique de calcul.
monitor-ui : Le service qui lance evidently ui sur le port 8000.
Important : Utilisez un volume partagé pour que le dossier bike_sharing_monitoring généré par le script soit lisible par l'interface UI.
3. Aide technique : Templates de configuration
Exemple de docker-compose.yml (à compléter)
version: '3.8'
services:
  monitor-script:
    build: .
    volumes:
      - ./bike_sharing_monitoring:/app/bike_sharing_monitoring
      - ./diagnostic_report.html:/app/diagnostic_report.html

  monitor-ui:
    image: python:3.10-slim
    volumes:
      - ./bike_sharing_monitoring:/app/bike_sharing_monitoring
    ports:
      - "8000:8000"
    command: sh -c "pip install evidently==0.7.2 && evidently ui --workspace /app/bike_sharing_monitoring --port 8000"
Exemple de Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]
Exemple de configuration du Workspace et du Projet
# Initialisation du Workspace local
ws = Workspace.create("bike_sharing_monitoring")

# Création ou récupération du projet
project_name = "Bike Rental Service Monitoring"
projects = ws.search_project(project_name)
project = projects[0] if projects else ws.create_project(project_name)
Exemple de validation par Report et Tests
from evidently import Report
from evidently.metrics import RegressionQualityMetric
from evidently.tests import lt
from evidently.core.datasets import Dataset, DataDefinition

# Définition des données (Modernway)
definition = DataDefinition(numerical_columns=num_features + ['target', 'prediction'])
reference_dataset = Dataset.from_pandas(X_train, data_definition=definition)
current_dataset = Dataset.from_pandas(X_test, data_definition=definition)

# Création du Report avec test intégré
# Par exemple, on teste que la RMSE est inférieure à un seuil
regression_report = Report(metrics=[
    RegressionQualityMetric(tests=[lt(50)]) 
], include_tests=True)

regression_report.run(reference_data=reference_dataset, current_data=current_dataset)

# Vérification du verdict (Machine-readable)
result = regression_report.dict()
failed_tests = len([t for t in result.get('tests', []) if t['status'] != 'SUCCESS'])

if failed_tests == 0:
    print("✅ Modèle validé pour la production.")
Exemple d'enregistrement des Snapshots
# Après chaque .run() d'un Report
ws.add_run(project.id, regression_report)
Livrables attendus :
main.py : Le script complet.
Dockerfile : Pour environnementaliser l'exécution du script.
docker-compose.yml : Doit orchestrer deux services :
monitor-script : Lance main.py pour générer les snapshots.
monitor-ui : Lance evidently ui sur le port 8000 pour visualiser les résultats.
diagnostic_report.html et le dossier bike_sharing_monitoring/.
README.md :
Commande pour lancer l'environnement (docker compose up).
Analyse :
Une fois votre environnement lancé via docker compose up, explorez l'interface sur http://localhost:8000 et répondez aux questions suivantes dans votre README.md :

Analyse de Performance : Quelle semaine (W1, W2 ou W3) présente la chute de performance la plus significative ?
Analyse de Dérive : En observant le DataDrift et le TargetDrift, quel facteur semble expliquer cette baisse de performance ? (Analysez les variables météo).
Remédiation : Si ce système était en production réelle, quelle action automatique mettriez-vous en place (Réentraînement, alerte météo, arrêt du service) ? Justifiez.
Critères d'évaluation :
Conteneurisation : L'environnement se lance-t-il avec une seule commande ?
Utilisation d'Evidently : Workspace, Snapshots et Reports (avec tests) correctement implémentés.
Analyse : Capacité à interpréter les graphes de l'UI pour identifier la cause de la dérive.
Qualité du code : Code modulaire et respect de la PEP8.
Bon courage !