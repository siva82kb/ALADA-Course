## ALADA Animation

The folder `animations` in the root directory contains a set of interactive animations for the different concepts covered in the ALADA course. The animations are created using the `matplotlib` library in Python. All interaction with these animations is done through the keyboard. 


### How to get these to work?

You will need Python 3.9 or higher. The best thing to do is to install Anaconda and create a new environment. You can do this by running the following commands in your terminal:

```bash
conda env create -f alada.yml
```

This will create a new environment called `alada` with all the necessary packages. You can then activate the environment by running:

```bash
conda activate alada
```

You should now be able to run the different scripts to open the animations and interact with them.

### K-Mean Algorithm
**k-Means Demo**: [`kmeans_demo.py`](https://github.com/siva82kb/aladaanim/blob/main/animations/kmeans_demo.py)

<!-- ### Continuous Optimization Animations
The following animations are available for the continuous optimization section:
1. **Univariate Gradient Descent**: [`univar_graddesc_stepsize.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/univar_graddesc_stepsize.py)
Demonstration of the effect of step size using a quadrating function $f(x) = x^2$.

2. **Univariate Newton's Method:** [`univar_func1_newton.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/univar_func1_newton.py)
Demonstration of Newton's method using an arbitrary function $f(x)$.

3. **Univariate Secant Method:** [`univar_func1_secant.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/univar_func1_secant.py)
Demonstration of the Secant method using an arbitrary function $f(x)$.

4. **Univariate Newton's Method:** [`univar_func2_newton.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/univar_func2_newton.py)
Demonstration where Newton's method can fail. The function used is $f(x) = 1 - \exp\left(-\frac{x^2}{4}\right)$.

5. **Univariate Levenberg-Marquardt Method:** [`univar_func1_newton_lm.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/univar_func1_newton_lm.py)
Demonstration where Newton's method can fail, while the Levenberg-Marquart works for the appropriate choice of $\lambda$. The function used is $f(x) = 1 - \exp\left(-\frac{x^2}{4}\right)$.

6. **Multivariate function and its variation along a direction:** [`multivar_directional_plot.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/multivar_directional_plot.py)
Demonstration of a multivariate function and its variation along its gradient and another arbitrary direction.

7. **Multivariate Gradient Descent with Backtracking:** [`multivar_graddesc_backtracking.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/multivar_graddesc_backtracking.py)
Demonstration of the effect of backtracking line search on the convergence of the gradient descent algorithm.

8. **Multivariate Optimization with different Methods:** [`multivar_methods.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/multivar_methods.py)
Demonstration of the gradient descent (fixed step size), Newton's Method, and Levenberg-Marquardt method on a multivariate function.

9. **Multivariate Optimization with Steepest Descent:** [`multivar_steepdesc.py`](https://github.com/siva82kb/aladaanim/blob/main/optimization/multivar_steepdesc.py)
Demonstration of the steepest descent method on a multivariate function.
 -->
