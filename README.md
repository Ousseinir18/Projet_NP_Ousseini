# Analyse de sentiment de tweets avec BERT

Projet de classification de sentiment : donner un texte (tweet) et prédire s'il est **négatif**, **neutre** ou **positif**.

## Données

- `train-3.csv` : dataset *Tweet Sentiment Extraction* (colonnes utilisées : `text` et `sentiment`).
- Encodage du fichier : `latin-1` (le CSV contient des caractères non-UTF-8).

## Modèle

- Base : `bert-base-cased` (Hugging Face).
- Architecture : BERT + une couche linéaire sur le token `[CLS]` pour classifier en 3 classes (`negative`, `neutral`, `positive`).
- Entraînement : 3 epochs, batch size 32, learning rate `2e-5`, 10% des données en validation.
- Suivi des métriques (loss/accuracy) via **wandb** (optionnel, désactivable).

## Fichiers du projet

| Fichier | Rôle |
|---|---|
| `train_colab_tweets.ipynb` | Notebook Google Colab pour entraîner le modèle sur `train-3.csv` et produire le checkpoint `model.pth`. |
| `checkpoints_bert/model.pth` | Poids du modèle entraîné (téléchargé depuis Colab). |
| `demo_tweets.py` | Petite interface **Gradio** en local pour tester le modèle : on tape un texte, il renvoie le sentiment prédit. |

## Comment relancer le projet

1. **Entraîner** (si besoin de refaire un modèle) : ouvrir `train_colab_tweets.ipynb` dans Google Colab, exécuter les cellules dans l'ordre, uploader `train-3.csv` quand demandé, puis télécharger `model.pth` à la fin.
2. **Tester en local** :
   - Placer `model.pth` dans `checkpoints_bert/` à côté de `demo_tweets.py`.
   - Vérifier que `CHECKPOINT_PATH` dans `demo_tweets.py` pointe bien vers ce fichier.
   - Lancer :
     ```
     python demo_tweets.py
     ```
   - Ouvrir `http://127.0.0.1:7860` dans le navigateur.

## Résultat

Le modèle prend un texte en entrée et renvoie une des 3 étiquettes : `negative`, `neutral`, `positive`, avec quelques exemples de tweets prêts à tester dans l'interface.
