Types of Naive Bayes in scikit-learn
Classifier	    Use Case
----------      --------
GaussianNB	    Continuous data, assumes Gaussian distribution
MultinomialNB	Discrete features (e.g. text, counts)
BernoulliNB	    Binary features

START
  |
  |-- Is your task classification (not regression)?
  |       |
  |       |-- NO ➜ ❌ Don't use Naive Bayes (use regression models)
  |       |
  |       |-- YES
  |
  |-- Is your data mostly text (e.g., emails, reviews)?
  |       |
  |       |-- YES ➜ ✅ Use Naive Bayes (MultinomialNB or BernoulliNB)
  |       |
  |       |-- NO
  |
  |-- Are the features mostly categorical or binary?
  |       |
  |       |-- YES ➜ ✅ Naive Bayes can work well
  |       |
  |       |-- NO (features are continuous)
  |              |
  |              |-- Are the continuous features roughly normally distributed?
  |              |       |
  |              |       |-- YES ➜ ✅ Try GaussianNB
  |              |       |-- NO ➜ ⚠️ Try other models (e.g., SVM, Random Forest)
  |
  |-- Are features strongly dependent on each other?
  |       |
  |       |-- YES ➜ ⚠️ Naive Bayes may perform poorly
  |       |-- NO  ➜ ✅ Suitable for Naive Bayes
  |
  |-- Do you need fast training and prediction?
  |       |
  |       |-- YES ➜ ✅ Use Naive Bayes (very fast)
  |       |-- NO ➜ You can try more complex models too

Need	                    Naive Bayes Fit?
----                        ----------------
Simple baseline model	        ✅ Yes
Interpretability	            ✅ Yes
Large number of features	    ✅ Yes
Fast training/prediction	    ✅ Yes
Continuous, nonlinear patterns	❌ No
Highly correlated input features❌ No