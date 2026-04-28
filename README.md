## Applied Linear Algebra in Data Analysis

**Semester:** August - December, 2025

The course information document is here: [Course Information](info.pdf)

### Course TAs (Jan 2026) 
<p align="left">
  <img src="tas/monisha.jpeg" alt="Dr Monishja Yuvyaj" width="100"/><br>
    <b>Dr Monisha Yuvaraj</b>(monisha.yuvaraj@cmcvellore.ac.in)<br>
</p>


### Course Modules

**Linear Systems and Matrix Operations**  \
[Part 01](lecture_slides/linsysmatop-01.pdf) | [Part 02](lecture_slides/linsysmatop-02.pdf) | [Part 03](lecture_slides/linsysmatop-013.pdf) | [Part 04](lecture_slides/linsysmatop-04.pdf)  

**Orthogonality**  \
[Part 01](lecture_slides/orthogonality.pdf) 

**Matrix Inverses**  \
[Part 01](lecture_slides/matrixinverses.pdf) 

<!-- **Case Study**  \
[Part 01](lecture_slides/casestudy-01.pdf)  -->

**Least Squares**  \
[Part 01](lecture_slides/leastsq.pdf) 

**Eigenvalues and Eigenvectors**  \
[Part 01](lecture_slides/eigenvalvec.pdf) 

**Eigenvalues and Eigenvectors Applications**  \
[Part 01](lecture_slides/eigenvalvecappln.pdf) 

**Positive Definite Matrices & Matrix Norms**  \
[Part 01](lecture_slides/pdmatnorm.pdf) 

### Quizes
All the quizes that have been conducted so far can be found [here](quizes/quiz.pdf).

### Tutorial and Assignments
You can find all the tutorials and assignments [here](assignments/problems.pdf).\
All assignments will be due by 11:59 PM on the due date. You have 5 late days to use throughout the semester. Each late day extends the deadline by 24 hours. After the late days are used, late submissions will not be accepted.

#### Least Sqaures Assignment Files
We have provided an ipython file with boilerplate code for the least squares assignment. You can find it [here](assignments/leastsq/04-lls-01.ipynb), along with the following data files:
1. `polyfit.csv`: [Dataset](assignments/leastsq/polyfit.csv) for polynomial fitting.
2. `timeseries.csv`: [Dataset](assignments/leastsq/timeseries.csv) for time series fitting.
3. `trialtactpos.csv`: [Dataset](assignments/leastsq/trialtactpos.csv) of the true position of a moving object.
4. `trailctrlpos.csv`: [Dataset](assignments/leastsq/trailctrlpos.csv) containing the fixed locations from which distance of the moving object are measured.
5. `trialtdist.csv`: [Dataset](assignments/leastsq/trialtdist.csv) with the distance measurements of a moving object from a set of fixed position.
6. `polyfitweighted.csv`: [Dataset](assignments/leastsq/polyfitweighted.csv) for polynomial fitting using weighted least squares.

Download all the files onto your computer. Open the notebook file in Google Colab, and run the to upload the data files. Once, that is done you cna then work with th rest of the file to solve the assingment problems.

#### Additional problems
Here are some additional problems to test your understanding of the course materials.
1. [k-Means Clustering](case_studies/case_study_01.ipynb): A notebook on k-means clustering of doctors' notes.
2. [Co-occurance graph](case_studies/case_study_01b.ipynb): A notebook on building graphs to analyse co-occurance of words in doctors' notes.


### Submission Link
All assignments must be submitted as soft copies on the [submission portal](https://docs.google.com/forms/d/e/1FAIpQLSdl6o5p-J7d7HGFZTuXuo08clEXsg3rCBF0jfTFv5O_0IH1XA/viewform?usp=dialog).

---

### ALADA Animations
<!-- The folder `animations` in the root directory contains a set of interactive animations for the different concepts covered in the ALADA course. The animations are created using the `matplotlib` library in Python. All interaction with these animations is done through the keyboard. 

#### How to get these to work?

You will need Python 3.9 or higher. The best thing to do is to install Anaconda and create a new environment. You can do this by running the following commands in your terminal:

```bash
conda env create -f alada.yml
```

This will create a new environment called `alada` with all the necessary packages. You can then activate the environment by running:

```bash
conda activate alada
```

You should now be able to run the different scripts to open the animations and interact with them. -->

### K-Means Algorithm
1. **k-Means Demo**: [`kmeans_demo.py`](https://github.com/siva82kb/ALADA-Course/blob/main/animations/kmeans_demo.py)
2. **k-Nearest Neighbors Classifier Demo**: [`knn_class_demo.py`](https://github.com/siva82kb/ALADA-Course/blob/main/animations/knn_class_demo.py)
3. **k-Nearest Neighbors Regression Demo**: [`knn_reg_demo.py`](https://github.com/siva82kb/ALADA-Course/blob/main/animations/knn_reg_demo.py)

### Linear Dynamical Systems
1. **Discrete-time Linear Dynamical System Demo**: [DT LDS Demo](https://siva82kb.github.io/alada-anim/lds-zero-input.html)
2. **Continuous-time Linear Dynamical System Demo**: [CT LDS Demo](https://siva82kb.github.io/alada-anim/lds-continuous-zero-input.html)


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

<!-- ## ALADA Animations
The repository also has a set of interactive animations to demonstrate some of the concepts covered in the course. You can find details about these animations [here](aladaanim.md). -->
