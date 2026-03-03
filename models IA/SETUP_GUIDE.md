# 🚀 GUIDE D'INSTALLATION - MODELS IA

## 📦 Installation des Dépendances

### Option 1: Installation Simple (Recommandée)
Exécutez cette commande unique pour installer toutes les librairies:

```powershell
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Option 2: Installation Progressive (Si Option 1 échoue)
Installez une par une pour éviter les conflits:

```powershell
# 1. Core data science
pip install numpy
pip install pandas

# 2. Machine Learning
pip install scikit-learn

# 3. Gradient Boosting
pip install xgboost

# 4. Visualization
pip install matplotlib
pip install seaborn
```

### Option 3: Utiliser un Environnement Virtuel (Meilleure Pratique)

#### Créer un environnement virtuel
```powershell
# Naviguer vers le dossier du projet
cd "C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template"

# Créer l'environnement virtuel
python -m venv venv_agriculture

# Activer l'environnement (Windows PowerShell)
.\venv_agriculture\Scripts\Activate.ps1

# Si erreur de permission, exécuter d'abord:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Installer les packages dans l'environnement virtuel
```powershell
# Une fois l'environnement activé (vous verrez "(venv_agriculture)" au début de la ligne)
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

#### Utiliser l'environnement pour exécuter les scripts
```powershell
# Activer l'environnement
.\venv_agriculture\Scripts\Activate.ps1

# Exécuter le script
cd "models IA"
python Model_A_Plant_Health.py

# Désactiver l'environnement quand vous avez fini
deactivate
```

---

## 🔧 Vérification de l'Installation

Exécutez ce script pour vérifier que tout est installé correctement:

```powershell
python -c "import numpy; import pandas; import sklearn; import xgboost; import matplotlib; import seaborn; print('✓ Toutes les librairies sont installées!')"
```

---

## 📊 Versions Recommandées

- Python: 3.8+
- numpy: latest
- pandas: latest
- scikit-learn: 1.0+
- xgboost: 1.6+
- matplotlib: 3.3+
- seaborn: 0.11+

Pour vérifier vos versions:
```powershell
pip list
```

---

## 🏃 Exécution des Modèles

### Model A - Plant Health Classification
```powershell
cd "C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA"
python Model_A_Plant_Health.py
```

Les résultats seront sauvegardés dans:
- `results/` - Graphiques et métriques
- `models/` - Modèles entraînés

---

## ⚠️ Résolution des Problèmes Courants

### Problème: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:**
```powershell
pip install xgboost
```

### Problème: "Permission denied" lors de l'activation du venv
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problème: Conflit de versions
**Solution:** Créer un nouvel environnement virtuel et réinstaller
```powershell
# Supprimer l'ancien venv
Remove-Item -Recurse -Force venv_agriculture

# Créer un nouveau
python -m venv venv_agriculture
.\venv_agriculture\Scripts\Activate.ps1
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Problème: Erreur "Python not found"
**Solution:** Vérifier que Python est installé et dans le PATH
```powershell
python --version
# Si erreur, télécharger Python depuis https://www.python.org/downloads/
```

---

## 💡 Recommandation Finale

**Pour éviter tout conflit, utilisez l'Option 3 (environnement virtuel).**

C'est la meilleure pratique en data science car:
- ✅ Isole les dépendances du projet
- ✅ Évite les conflits avec d'autres projets Python
- ✅ Facilite le partage et la reproductibilité
- ✅ Permet de tester différentes versions de librairies

---

## 📝 Commandes Rapides

### Setup complet en une fois:
```powershell
cd "C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template"
python -m venv venv_agriculture
.\venv_agriculture\Scripts\Activate.ps1
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
cd "models IA"
python Model_A_Plant_Health.py
```

---

**Bon courage! 🚀**
