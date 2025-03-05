# sslmejor

This repo contains a linear semi-supervised learning algorithm, of which people say "sslmejor". 
Linear SSL is of interest because linear classification of good features is often enough to solve most problems.
However, it works best when abundant labeled data is available.
What happens if there is no that much labeled data? If we still have unlabeled data, we can do semi-supervised learning.

Graph-based semi-supervised learning is of interest, and according to https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf they work when the graph construction is correct.
These are based on the following minimization:

$$\arg\min_\theta \left( \beta \sum_{i=1}^L c(f_\theta(x_i), y_i) + \gamma \sum_{i=1}^{l+u} \sum_{j=1}^{l+u} W_{ij} c_u(f_\theta(x_i), f_\theta(x_j)) \right) $$

where $f_\theta$ is our linear model followed by a softmax (for instance), and $c,c_u$ are cost functions. It is of interest to set $\beta \to +\infty$ as fitting the available labels is more important than the smoothness on the graph defined by the adjacency matrix $W$. In what follows we will assume $\gamma=1$ unless otherwise specified.

## Constrained Optimization
Another way to see the objective is to take the fit to the available labels as a hard constraint and minimize the smoothness cost. This is

$$ \arg\min_\theta \sum_{i\neq j} W_{ij}c_u(f_\theta(x_i),f_\theta(x_j)) \quad \text{s.t.} \max_i c(f_\theta(x_i), y_i) < \epsilon, $$

which means that we ask the fit to any label to be at most $\epsilon$ wrong. It might happen that this is unfeasible, i.e., that the labels are not linearly separable. This could be solved by either expanding the feature space (e.g. by using random features) or by relaxing the constraint. If we tolerate samples to be $\xi_i$ wrong more than $\epsilon$, and we try to be the least tolerant as possible by penalizing $\|\xi\|_2^2$, we get 

$$ \arg\min_{\theta,\xi} \alpha \|\xi\|_2^2 +  \sum_{i\neq j} W_{ij}c_u(f_\theta(x_i),f_\theta(x_j)) \quad \text{s.t.} \max_i c(f_\theta(x_i), y_i) < \epsilon + \xi_i. $$

### Choosing $\epsilon$
Now, for $C$ class classification one usually goes with the cross-entropy loss as $c$. What is the maximum cross-entropy loss for which we are still sure that the sample is classified correctly? Or equivalently, what's the minimum cross-entropy loss for which a sample can still be incorrectly classified? A very low loss happens when the actual label $y_i$ is assigned high probability, but the classification is incorrect, this is, $y_i$ is assigned high probability but there is another class with higher probability. This happens when there are only two classes with high probability and they are equal at about $1/2$ (but with the incorrect one being marginally above). Therefore, we know that if the cross entropy is lower than $-\log(1/2)$ then the classification is correct. Therefore we set $\epsilon=-\log(1/2)$.

### Lagrangian
Having discussed how to set $\epsilon$, we turn our attention to the problem. As it is a constrained optimization problem, one can use the Lagrangian $\mathcal{L}$ to solve it, by running:

$$\max_\lambda \min_{\theta,\xi} \alpha \|\xi\|_2^2 + \sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon - \xi_i \right), $$

where $\lambda \in {\mathbb{R}^l}^+$. Once the solution of this problem has been found one can take the $\theta$ and use it for classification **UNJUSTIFIED CLAIM**. How do we solve this problem?

### Removing $\xi$

It is interesting to note that one can separate it into the part depending on $\xi$ and the part depending on $\theta$:

$$=\max_\lambda \min_{\theta} \left[\sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon  \right)\right] + \max_\lambda \min_\xi \left[\alpha \|\xi\|_2^2  + \sum_i \lambda_i (-\xi_i)\right] $$
$$=\max_\lambda \min_{\theta} \left[\sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon  \right)\right] + \max_\lambda  \sum_i \min_\xi \left[\alpha \xi^2 - \lambda_i \xi_i\right] $$
$$=\max_\lambda \min_{\theta} \left[\sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon  \right)\right] + \max_\lambda  \sum_i \frac{-\lambda_i^2}{4\alpha}  $$
$$=\max_\lambda \min_{\theta} \left[\sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon  \right)\right] + \max_\lambda   -\frac{\|\lambda\|_2^2}{4\alpha}  $$
$$=\max_\lambda \min_{\theta} \left[ -\frac{\|\lambda\|_2^2}{4\alpha} + \sum_{i\neq j} W_{ij} c_u(f_\theta(x_i),f_\theta(x_j)) + \sum_i \lambda_i \left( c(f_\theta(x_i), y_i) - \epsilon  \right)\right] , $$
and this final expression does not depend on $\xi$ anymore!

### Numerical Optimization

We can solve this via alternating optimization with a slowly varying $\lambda$. In other words, we compute this function (the lagrangian) and minimize it with respect to $\theta$ by taking one gradient minimization step, and then take one gradient maximization step with respect to $\lambda$. The parameter $\alpha$ relates to the weight decay, the learning rate of the optimizer of $\lambda$ is low, while the learning rate of the optimizer of $\theta$ is whatever is reasonable. Note that in fact there are two terms, one that depends on the adjacency matrix and $c_u$, and the other term is similar to cross-entropy minimization of supervised learning. The only parameter to consider here is the weight decay.

## Smoothness Term
There are two things to define in the smoothness term, the adjacency matrix $W$ and the cost function between two distributions $c_u$.

### Graph Construction 
Let us discuss now how to build the semi-supervised term. It has been said in https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf that the graph constuction step is the most important. We go for a default choice: use the exponential of the negated euclidean distances as similarities (with some temperature) and build a k-NN graph. In the survey they recommend $k$ to be low, and considering that DINOv2 code suggests using $20$ neighbors for best k-NN classification performance, that will be our upper bound. We set the temperature to be relatively high ($\tau=100$) as it is on the order of magintude of the distances. The distance function, the function that converts distances to similaritites, its temperature and the number of neighbors $k$ are already a lot of dimensions to optimize for the graph. 

### Distribution Distance $c_u$

There are many distances between distributions. For simplicity we take the Brier score, which is the mean squared error between the two distributions, i.e.,  $c_u(f_\theta(x_i), f_\theta(x_j))=\|f_\theta(x_i)-f_\theta(x_j)\|_2^2$.

## Comparison with Baselines

There are two baselines:
- Linear classification, which amounts to setting $\beta=1, \gamma=0$ in the very first equation.
- Direct optimization, which amounts to setting $\beta=1$ and leave $\gamma$ as a hyperparameter.

The proposed method should be better than linear classification because it accesses more information. The proposed method should also be better than direct optimization because weight decay does not change the result of the constrained formulation much, while changing $\gamma$ does. More importantly, the constrained formulation automatically adapts and weights each datapoint to ensure correct fit to the training set, while the original objective uses averages. The constrained formulation is more interpretable, and it is equivalent to using $\gamma=1$ and having a very good accuracy on the training set by using different $\beta_i$ for each training sample. 



