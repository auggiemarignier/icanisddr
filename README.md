# Comparing Inner Core Anisotropy Models with the Savage-Dickey Density Ratio

[![tests](https://github.com/auggiemarignier/icanisddr/actions/workflows/tests.yml/badge.svg)](https://github.com/auggiemarignier/icanisddr/actions/workflows/tests.yml)

## Inner Core Anisotropy

Following [Brett et al., 2024](https://www.nature.com/articles/s41561-024-01539-6#Sec6).

We consider a model of transverse isotropy where the magnitude of anisotropy and the direction of the fast axis are independent variables.

The elastic tensor (in Voigt notation) for transverse isotropy is determined by 5 Love parameters

$$ \mathcal{C} = \begin{pmatrix}
A & A - 2N & F & 0 & 0 & 0 \\
A - 2N & A & F & 0 & 0 & 0 \\
F & F & C & 0 & 0 & 0 \\
0 & 0 & 0 & L & 0 & 0 \\
0 & 0 & 0 & 0 & L & 0 \\
0 & 0 & 0 & 0 & 0 & N \\
\end{pmatrix} $$

(excuse the confusing notation of the matrix $\mathcal{C}$ and the Love parameter $C$)

Rotating this using 2 Euler angles $\eta_1, \eta_2$ (the third is just a rotation about the fast axis but in transverse isotropy this is symmetrical) and the rotation matrix $R(r^{-1})$ where

$$ r^{-1} = \begin{pmatrix}
\cos\eta_1\cos\eta_2 & -\sin\eta_1 & \sin\eta_2\cos\eta_1 \\
\sin\eta_1\cos\eta_2 & \cos\eta_1 & \sin\eta_1\sin\eta_2 \\
-\sin\eta_2 & 0 & \cos\eta_2 \\
\end{pmatrix} $$

and

$$ R(r) = \begin{pmatrix}
r_{11}^2 & r_{12}^2 & r_{13}^2 & 2r_{12}r_{13} & 2r_{11}r_{13} & 2r_{11}r_{12} \\

r_{21}^2 & r_{22}^2 & r_{23}^2 & 2r_{22}r_{23} & 2r_{21}r_{23} & 2r_{21}r_{22} \\

r_{31}^2 & r_{32}^2 & r_{33}^2 & 2r_{32}r_{33} & 2r_{31}r_{33} & 2r_{31}r_{32} \\

r_{21}r_{31} & r_{22}r_{32} & r_{23}r_{33} & r_{22}r_{23} + r_{32}r_{23} & r_{23}r_{31} + r_{21}r_{33} & r_{21}r_{32} + r_{31}r_{22} \\

r_{11}r_{31} & r_{12}r_{32} & r_{13}r_{33} & r_{32}r_{13} + r_{12}r_{33} & r_{33}r_{11} + r_{13}r_{31} & r_{31}r_{12} + r_{11}r_{32} \\

r_{11}r_{21} & r_{12}r_{22} & r_{13}r_{23} & r_{12}r_{23} + r_{13}r_{22} & r_{13}r_{21} + r_{23}r_{11} & r_{11}r_{22} + r_{21}r_{12} \\
\end{pmatrix} $$

we get a general rotated transverse isotropic elastic tensor

$$ D = R(r^{-1})CR(r^{-1})^T $$

Then the relative travel time change is given by

$$ \frac{\delta t}{t_{\mathrm{PREM}}} = \sum_{i,j,k,l = 1}^3 n_i n_j n_k n_l D_{ijkl}(\eta_1, \eta_2, \delta A, \delta C, \delta F | N = N_{\mathrm{PREM}}, L = L_{PREM}) $$

This reduces to the standard fast axis parallel to Earth's rotation axis when $\eta_1 = \eta_2 = 0$.

$\mathcal{C}$ reduces to an isotropic elastic tensor when we have the following

\begin{align*}
A = C = \lambda + 2 \mu \\
A - 2N = F = \lambda \rightarrow N = \mu \\
L = N
\end{align*}

where $\lambda, \mu$ are the Lamé parameters (bulk and shear moduli).

PREM is isotropic in the inner core so $N_{\mathrm{PREM}} = N_{\mathrm{PREM}}$.  We only have P-wave measurements anyway so there is no sensitivity to $L$.

## Savage Dickey Density Ratio

The SDDR gives a fast way of estimating the Bayes Factor of nested models.

Consider a model $\mathcal{H}_2$ that has parameters given by $\left( \mathbf{\theta}, \mathbf{\nu} \right)$.  A model $\mathcal{H}_1$ is nested in $\mathcal{H}_2$ if it has the same parameters but where a subset are fixed to a specific value i.e. $\left( \mathbf{\theta}, \mathbf{\nu} = \mathbf{\nu}^* \right)$.  $\mathcal{H}_2$ is called the super-model.

The evidence for model $\mathcal{H}_i$ is given by

$$ z_i = p(\mathbf{d}|\mathcal{H}_i) \int p(\mathbf{d} | \mathbf{m}, \mathcal{H}_i)p(\mathbf{m} | \mathcal{H}_i)\;\mathrm{d}\mathbf{m} $$

where the parameter vector $\mathbf{m} = \left( \mathbf{\theta}, \mathbf{\nu} \right)$.  The Bayes Factor comparing models $\mathcal{H}_1$ and $\mathcal{H}_2$ is given by

$$B_{12} = \frac{z_1}{z_2}$$

If

1. the priors for the common parameters $\mathbf{\theta}$ are the same under both models i.e. $p(\mathbf{\theta}|\mathcal{H}_1) = p(\mathbf{\theta}|\mathcal{H}_2)$;
2. the joint prior of the common and nuisance parameters in the super model is separable i.e. $p(\mathbf{\theta}, \mathbf{\nu}) = p(\mathbf{\theta})p(\mathbf{\nu})$; and
3. at some value of the nuisance parameters $\mathbf{\nu} = \mathbf{\nu}^*$ the likelihood of the supermodel collapses to that of the nested model i.e. $p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu} = \mathbf{\nu}^*, \mathcal{H}_2) = p(\mathbf{d} | \mathbf{\theta}, \mathcal{H}_1)$

then the SDDR simplifies the Bayes Factor.

\begin{align*}
B_{12} &= \frac{p(\mathbf{d}|\mathcal{H}_1)}{p(\mathbf{d}|\mathcal{H}_2)} \\
       &= \frac{\int p(\mathbf{d} | \mathbf{\theta}, \mathcal{H}_1)p(\mathbf{\theta} | \mathcal{H}_1)\;\mathrm{d}\mathbf{\theta}}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\theta}, \mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}} \\
       &= \frac{\int p(\mathbf{d} | \mathbf{\theta}, \mathcal{H}_1)p(\mathbf{\theta} | \mathcal{H}_1)\;\mathrm{d}\mathbf{\theta}}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\theta} | \mathcal{H}_2)p(\mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}} \\
       &= \frac{\int p(\mathbf{d} | \mathbf{\theta}, \mathcal{H}_1)\;\mathrm{d}\mathbf{\theta}}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}} \\
       &= \frac{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu} = \mathbf{\nu}^*, \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}} \\
       &= \frac{p(\mathbf{\nu} = \mathbf{\nu}^* | \mathcal{H}_2)}{p(\mathbf{\nu} = \mathbf{\nu}^* | \mathcal{H}_2)}\frac{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu} = \mathbf{\nu}^*, \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}} \\
       &= \frac{1}{p(\mathbf{\nu} = \mathbf{\nu}^* | \mathcal{H}_2)}\frac{p(\mathbf{d} | \mathbf{\nu} = \mathbf{\nu}^*, \mathcal{H}_2)p(\mathbf{\nu} = \mathbf{\nu}^* | \mathcal{H}_2)}{\int p(\mathbf{d} | \mathbf{\theta}, \mathbf{\nu}, \mathcal{H}_2)p(\mathbf{\nu} | \mathcal{H}_2)\;\mathrm{d}\mathbf{\theta}\mathrm{d}\mathbf{\nu}}
\end{align*}

The second term on the right hand side is the normalised posterior probability that $\mathbf{\nu} = \mathbf{\nu}^*$ under $\mathcal{H}_2$. Thus the SDDR is

$$ B_{12} = \frac{p(\mathbf{\nu} = \mathbf{\nu}^* | \mathbf{d}, \mathcal{H}_2)}{p(\mathbf{\nu} = \mathbf{\nu}^* | \mathcal{H}_2)} $$

i.e. the ratio of the normalised posterior to the prior, both under the super model, evaluated at a particular value of the nuisance parameters.

In practice, this means one only needs to sample and fit a normalised distribution to the supermodel, and the Bayes Factor with the nested model follows immediately.

## SDDR for Inner Core Anisotropy

Our basic supermodel vector is

$$ \theta = (\eta_1, \eta_2, \delta A, \delta C, \delta F) $$

at each grid point.

The nested model that represents transverse isotropy with fast axis parallel to rotation axis is

$$ \theta = (\delta A, \delta C, \delta F, \eta_1 = 0, \eta_2 = 0) $$

The nested model that represents isotropy would be

$$ \theta = (\delta A, \delta C = \delta A, \delta F = \delta A - 2\delta N, \eta_1 = 0, \eta_2 = 0) $$

(DOUBLE CHECK FOR F.  We have F=A-2N for isotropy but we're dealing with perturbations here)

However, here were our constraints on some random variables are (linear) functions of other random variables.  The need to be constants for SDDR.  So instead we define our supermodel in terms of differences, and the nested model is when those differences equal 0.

The super model becomes

$$ \theta = (\delta A, \delta C_A, \delta F_A, \eta_1, \eta_2, ) $$

Where $C_A = C - A$ and $F_A = F - A + 2N$ (AGAIN CHECK WHAT THESE LOOK LIKE FOR PERTURBATIONS)

The isotropic nested model is then

$$ \theta = (\delta A, \delta C_A = 0, \delta F_A = 0, \eta_1 = 0, \eta_2 = 0) $$

The priors for each model parameter are given by

$$ \eta_1 \sim U(-180, 180) $$

$$ \eta_2 \sim U(0, 90) $$

$$ \delta A, \delta C, \delta F = \mathcal{N}(1.0, 0.2) $$

The priors for the Euler angles are geometrically motivated.  As for the priors for the perturbations to the Love parameters, this is what is said to be the prior in [Brett et al., 2024](https://www.nature.com/articles/s41561-024-01539-6#Sec6), although why it is not a zero-mean Gaussian is not clear.
