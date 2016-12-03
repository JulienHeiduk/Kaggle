# Which customers are happy customers?

From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.
Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.
In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

Link: https://www.kaggle.com/c/santander-customer-satisfaction

#Remarques
2 fichiers csv à disposition: Un pour l'apprentissage et un pour le test. Le but étant réaliser un score sur une population de la Bancassurance Santander permettant d'identifier les personnes susceptibles d'exprimer leur insatisfaction et donc être susceptible de quitter la banque.

Beaucoup de travail sur le fichier avec notamment la suppression des variables identiques, ajout de variables étant une combinaison lineaire des variables de base, Creation d'une cross validation pour le XGBOOST, Feature engineering (nombre de 0 par lignes etc...), imputation des valeurs manquantes.

Meilleurs résultats obtenus: Public:0.840723 Private:0.827639 (resultats non choisis pour le calcul final) Résultats soumis: Public:841854 Private:0.823895

Problème de surapprentissage sur les fichiers choisis pour le calcul final. Ce qui entraine une chute du score de 0.004... Ligne 477-512 du .py => overfitting sur l'echantillon train...

-Suppression variables identiques -Ajout de combianaison suivant la correlation avec la variables à predire -Calcul des corrélations deux à deux -CV pour XGB avec metric AUC (à nettoyer: une partie n'est pas à utiliser)
