# Cahier des charges — Projet 9, variante A
## Petit LLM : fine-tuning QLoRA pour une sortie structurée (extraction d'informations en JSON)

> Le sujet ne fournit pas de cahier des charges détaillé pour le projet 9. Celui-ci est rédigé
> sur le modèle des cahiers 1 à 4 du sujet, à partir de la fiche du projet 9, de son « modèle de
> départ » et de la variante A.

### Contexte
Les grands modèles de langage savent extraire des informations d'un texte, mais leur sortie libre
est difficile à exploiter automatiquement. Les petits modèles ouverts (1 à 2 milliards de
paramètres) sont exécutables sur un GPU gratuit, mais suivent moins bien les consignes de format.
Le projet mesure si une adaptation légère (QLoRA) apporte un gain réel par rapport au même modèle
guidé uniquement par le prompt.

### Question de recherche
Sur des phrases jamais vues, un modèle Qwen2.5-1.5B-Instruct quantifié en 4 bits et adapté par
QLoRA (rang 16) produit-il des extractions JSON (personnes, organisations, lieux) **plus justes,
plus souvent valides et moins hallucinées** que le même modèle en zero-shot et en few-shot ?

### Objectifs pédagogiques
Fine-tuning efficace en paramètres (LoRA, quantification 4 bits), formatage en dialogue (chat
template) et masquage de la perte, évaluation rigoureuse d'une sortie générée, analyse des
hallucinations, gestion d'un budget de calcul.

### Données
* **Source** : CoNLL-2003 (dépêches Reuters en anglais, annotations d'entités nommées), copie
  Parquet `lhoestq/conll2003` figée à un commit précis.
* **Conversion** : chaque phrase devient un enregistrement
  `{"persons": [...], "organizations": [...], "locations": [...]}` ; la classe MISC est écartée
  (trop hétérogène) ; valeurs distinctes copiées mot pour mot du texte.
* **Découpage figé dès la semaine 1** : 1 500 exemples d'entraînement (split *train*),
  300 de validation (*validation*), 500 de test (*test* officiel). Phrases de moins de 4 mots
  écartées.
* **Anti-fuite** : doublons supprimés dans chaque ensemble ; toute phrase présente dans un
  ensemble prioritaire (test > validation > entraînement) est retirée des autres ; vérification
  par assertion et empreintes SHA-256 enregistrées dans `data/processed/manifest.json`.

### Travail demandé
1. **Exploration** : nombre d'entités par type et par ensemble, part des phrases sans entité
   (environ 21 %), longueurs, exemples.
2. **Baselines** (même modèle, même prompt système) :
   * zero-shot ;
   * few-shot avec 3 exemples tirés de l'entraînement uniquement ;
   * baseline triviale « réponse vide » comme point de repère.
3. **Fine-tuning QLoRA** : quantification NF4, LoRA de rang 16 sur les projections d'attention
   et du MLP, perte calculée **uniquement sur la réponse JSON**, sélection du meilleur point de
   contrôle par la perte de validation.
4. **Évaluation** sur la validation avec des métriques identiques pour tous les systèmes :
   * taux de JSON valide, de JSON strict (sortie = un objet JSON seul), de schéma valide ;
   * précision, rappel, F1 micro et par champ, avec intervalle de confiance bootstrap à 95 % ;
   * exact match par phrase ;
   * **taux d'hallucination** : valeurs prédites absentes du texte source ;
   * latence par exemple.
5. **Expériences de réglage** (au moins deux, sur validation uniquement) : quantité de données
   d'entraînement (100 / 500 / 1 500), rang LoRA (8 contre 16) ou taux d'apprentissage.
6. **Analyse d'erreurs** : taxonomie automatique (JSON invalide, hallucination, mauvais champ,
   frontière, faux positif, oubli) puis examen manuel d'au moins 20 erreurs et hypothèses.
7. **Test final, une seule fois** : meilleure baseline et modèle QLoRA retenu ; usage unique
   imposé par un verrou logiciel.

### Contraintes
* GPU T4 gratuit (Colab ou Kaggle), **4 h de GPU au maximum** au total.
  Ordre de grandeur : baselines 15 min, entraînement principal 20 à 30 min, ablations
  2 × 15 min, test 10 min.
* Pas de bfloat16 sur T4 : calcul en float16 ; longueur maximale de séquence 512 tokens.
* Graine fixée (42), décodage glouton, tous les hyperparamètres dans `configs/base.yaml`.

### Livrables
* Dépôt de code : paquet Python testé (`src/llmx`), configuration, tests unitaires.
* Notebook reproductible `notebooks/main.ipynb` exécuté avec ses sorties.
* Artefacts : découpages et manifeste, métriques et prédictions de chaque système, courbes.
* Soutenance de 10 minutes.

### Critères de réussite minimaux
* Baseline prompt seul (zero-shot et few-shot) et modèle QLoRA comparés avec les mêmes métriques.
* Au moins deux expériences de réglage.
* Analyse des erreurs et des hallucinations, avec exemples commentés.
* Test utilisé une seule fois, absence de fuite justifiée.
* Discussion des limites et des risques éthiques d'au moins une page.

### Pièges identifiés et parades
| Piège | Parade |
|---|---|
| Évaluer sur des exemples vus à l'entraînement | Découpage figé, déduplication inter-ensembles, verrou sur le test |
| Exemples few-shot issus de l'évaluation | Exemples tirés de l'entraînement uniquement |
| Comparaison injuste baseline / fine-tuning | Même modèle, même prompt système, même décodage |
| Réparer silencieusement les sorties invalides | Trois niveaux de validité rapportés ; sortie invalide = entités manquées |
| Mémoire GPU insuffisante | 4 bits, gradient checkpointing, optimiseur paginé 8 bits, batch 4 × accumulation 4 |
| Conclusions sur un seul chiffre | Intervalles de confiance bootstrap, analyse par champ et par type d'erreur |

### Calendrier
| Semaine | Jalon |
|---|---|
| 1 | Cadrage, données figées, baselines zero-shot et few-shot sur la validation |
| 2 | Fine-tuning QLoRA, courbes d'apprentissage, expériences de réglage |
| 3 | Analyse d'erreurs, test final unique, rédaction de l'analyse critique, soutenance |

### Extension (si le temps le permet)
Mini-RAG : récupérer les exemples d'entraînement les plus proches (modèle d'embeddings compact,
par exemple all-MiniLM-L6-v2) comme exemples few-shot dynamiques, et comparer au few-shot fixe.
