<!-- markdownlint-disable -->
# Automatic History Matching with a coupled Mixture of experts - Weighted adaptive Regularised Ensemble Kalman Inversion ($`\alpha`$REKI) scheme and a Physics Informed Neural Operator (PINO) based surrogate forward model on the Norne Field 
![alt text](Visuals/All1.png)


## An AI enabled Automatic History Matching Workflow with a PINO based forward solver:
Calibration of subsurface structures is an important step to forecast fluid dynamics and behaviours in a plethora of geoenvironments such as petroleum and geothermal reservoirs. History matching is an ill-posed inverse process to find reservoir model parameters honouring observations by integration of static (e.g., core, logging, and seismic) and dynamic data (e.g., oil and gas rate, water cut, bottomhole pressure, and subsidence/uplift) .In recent developments, the ensemble Kalman filter (EnKF) (Evensen 1994, Law et al 2012, Stuart 2010), ensemble smoother (ES) (van Leeuwen & Evensen 1996) ES with multiple data assimilation (ES-MDA) (Emerick & Reynolds 2013) and Iterative variants of ES (Chen & Oliver 2011, Iglesias 2013) have all been utilised for history matching/data assimilation problems. However, the ensemble-based data assimilation approaches have limited capability in preserving non-Gaussian distributions of model parameters such as facies (Aanonsen et al.,2009, Villegas et al, 2018, Lorentzen et al, 2013). In the ensemble-based data assimilation techniques, model parameters lose the non- Gaussianity of their original distributions (generated from a geostatistical software) that are initially constrained and constructed to available hard data and the distributions of the model parameters tend towards Gaussian ones (Evensen & Eikrem 2018, Kim et al 2018). Some methods utilised for this parametrisation technique in reservoir characterization literature is, the discrete cosine transform (DCT) (Jafarpour & McLaughlin, 2007)  (Liu & Jafarpour, 2013), level set (Moreno & Aanonsen 2007, Dorn & Villegas 2008, Villegas et al 2018, Chang et al 2010, Lorentzen et al., 2013) and sparse geologic dictionaries (Etienam et al, 2019, Kim et al, 2018, Khaninezhad et al, 2012). In particular, Fourier transform-based methods such as DCT are capable of capturing essential traits such as main shapes and patterns of a facies channel reservoir (Khaninezhad, et al., 2012) but reveal a deficiency in describing a crisp contrast among different facies because of data loss from inverse transformation (Kim et al 2018, Khaninezhad et al, 2012, Tarrahi & Afra,2016).

Reservoir model calibration is applicable and relevant for locating new hydrocarbon deposits and for CCUS stratigraphic trapping initiatives in many energy companies. Energy companies are aiming to accelerate their forward simulation codes for precise and detailed subsurface/dynamic/Petro-physical mappings of their assets/plays. A fascinating and easy to implement meshless approximation to solve reservoir simulation forward problems using physics constrained/informed deep neural networks show promising results.In this project, a physics informed neural operators (PINOs) is developed for surrogating a Two phase flow black oil model and a recently developed weighted adaptive regularised ensemble kalman inversion method is used for solving the inverse problem.


The aim of this project is to develop an integrated workflow, where the finite volume fully/adaptive implicit black oil reservoir simulator is replaced by a phyiscs informed neural operator. This developed PINO surrogate is now used in an inverse problem methodology. A newly developed adaptive regularised ensemble alman inversion method with covariance localisation together with a mixture of experts approach is implemented. This approach called _PINO-CCR_ is well suited for forward and inverse uncertainty quantifiication .


## Methods for the forward and inverse problem (In the weeds):


### Inverse problem 

**Weighted-$`\alpha`$REKI**

The aim and target of history matching can be formulated as;

```math
\begin{equation}
\min Φ(m;d^{obs})
\end{equation}
```
where the cost functional for the history matching inverse problem is posed as [24,29,30,31,35,36,19,18,14,15,6,1].
```math
\begin{equation}
Φ(m;d^{obs})≡ \frac{1}{2}\left\VertΓ^{\frac{-1}{2}}(d^{obs}-G(m))\right\Vert^2
\end{equation}
```
Where $`‖.‖`$ represents the $`M`$ dimensional Euclidean Norm.
                                                       
in Eqn.2 $`Φ`$ is the objective function of history matching and m is the state vector composed of reservoir variables (e.g., permeability/porosity and facies) in this case. The typical expression of $`Φ(m;d^{obs})`$for ensemble-based history matching problems is presented as (Emerick & Reynolds, 2013, Oliver et al 2008, Tarantola 2005);
```math
\begin{equation}
Φ(m;d^{obs}) = (m-m^b)^T B^{-1} (m-m^b ) + (d^{obs}-d)^T Γ^{-1} (d^{obs}-d)
\end{equation}
```
In Eqn.3, $`\,`$ $`m^b`$ denotes state vector before update and the superscript b denotes the background; $`B`$ denotes the covariance matrix of $`m^b`$; $`d^{obs}`$ denotes the observed responses; $`d = G(m)`$ is the dynamic vector composed of simulated responses constructed by running a reservoir simulator or the surrogate simulator $`G`$ for the state vector $`m`$; and $`Γ`$ denotes the covariance matrix of observation error. The right-hand side of Eqn.3 is the addition of background and observation error terms (Emerick & Reynolds, 2013).  $`m`$  in principle can contain any unknown variables such as facies indexes, permeability/porosity field, coefficients of discrete cosine functions or sparse coefficients depending.

$`\frac{∂J(m)} {∂m} = 0`$ can be used to derive the minimum of the cost function and update equation for $`m`$ as (Emerick and Reynolds, 2013) 
```math
\begin{equation}
m_i = m_i^{b} + C_{md} (C_{dd} +α_p C_d )^{-1} (d_i^{pert}-d_i )\quad  for \quad i = 1…N_{ens}
\end{equation}
```
                               

In Eqn.4 $`\,`$ the subscript $`i`$ denotes the $`i^{th}`$ ensemble member; $`C_{md}`$ denotes the cross-covariance matrix of $`m`$ and $`d`$; $`C_{dd}`$ denotes the autocovariance matrix of $`d`$; $`α_p`$ is the coefficient to inflate $`C_d`$, which denotes the covariance matrix of the observed data measurement error (Emerick and Reynolds, 2013); $`d^{pert}`$ denotes the observation data perturbed by the inflated observed data measurement error; and $`N_{ens}`$ is the ensemble size (i.e., number of reservoir models as columns of the matrix in the ensemble). Conventionally, ensemble-based history matching updates $`N_{ens}`$ reservoir models simultaneously. In Eqn.4, $`\frac {C_{md}}{(C_{dd}+α_pC_d )}`$ denotes the Kalman gain $`K`$, which is normally computed with regularization by SVD using 99.9% of the total energy in singular values (Emerick and Reynolds, 2013, Oliver et al 2008, Law et al 2012) 

The main difference between ES and ES-MDA is the update process of the state vector $`m`$. ES updates the state vector of each ensemble member using observation data measured at all time steps (Emerick and Reynolds, 2013). Compared to the single assimilation of ES, ES-MDA assimilates every state vector $`N_a`$ times using an inflated covariance matrix of measurement error (Emerick and Reynolds, 2013). In this case, $`N_a`$ is the number of assimilations in ES-MDA. 
We define $`C_{md}`$ and $`C_{dd}`$  as follows:
```math
\begin{equation}
C_{md} = \frac{1}{(N_{ens}-1)}  \sum_{n=1}^{N_{ens}} (m_i- \bar{m})(d_i-\bar{d})^T 
\end{equation}
```

```math
\begin{equation}
C_{dd} = \frac{1}{(N_{ens}-1)}  \sum_{n=1}^{N_{ens}} (d_i- \bar{d})(d_i-\bar{d})^T 
\end{equation}
```
```math
\textrm{where}\quad\bar{m}\quad\textrm{denotes the mean of the parameters and }\quad\bar{d}\quad\textrm{denotes the mean of predicted values}
```
```math
\textrm{In ES-MDA}, α_p\quad\textrm{is normally constrained to}\quad\sum_{n=1}^{N_{a}} \frac{1}{a_p} = 1  
```

```math
\begin{equation}
d_{i}^{pert} = d^{obs} + a_p^{0.5} C_d^{0.5} z_{d,i} \quad   for \quad i=1…N_{ens} 
\end{equation}
```


The second term on the right-hand side of Eqn.7 $`\,`$ is known as the perturbation term. It reflects the uncertainty associated with data measurement and processing. The stochastic characteristics of $`C_d`$ are reflected by $`z_d \sim \mathcal{N}(0,I_{N_d } )`$  $`z_d`$ is the random error matrix to observations, which is constructed with a mean of zero and a standard deviation of $`I_{N_d}`$, where $`N_d`$ is the number of time steps found in the observations (Emerick and Reynolds 2013).

The newly developed $`\alpha`$REKI constructs the dampening parameter $`α_p`$ via the discrepancy principle and jeffrey's divergence

```math
\begin{equation}
\rho = 
\begin{cases} 
-\frac{1}{4} \left(\frac{|z|}{c}\right)^5 + \frac{1}{2} \left(\frac{|z|}{c}\right)^4 + \frac{5}{8} \left(\frac{|z|}{c}\right)^3 - \frac{5}{3} \left(\frac{|z|}{c}\right)^2 + 1, & 0 \leq |z| \leq c \\
\frac{1}{12} \left(\frac{|z|}{c}\right)^5 - \frac{1}{2} \left(\frac{|z|}{c}\right)^4 + \frac{5}{8} \left(\frac{|z|}{c}\right)^3 + \frac{5}{3} \left(\frac{|z|}{c}\right)^2 - 5\left(\frac{|z|}{c}\right) + 4 - \frac{2}{3} \left(\frac{|z|}{c}\right), & c \leq |z| \leq 2c \\
0, & 2c \leq |z| 
\end{cases}
\end{equation}
```

```math
\begin{equation}
m_i = m_i^{b} + \rho \, o \, (C_{md} (C_{dd} + \alpha_p C_d )^{-1} (d_i^{pert}-d_i ))\quad  for \quad i = 1…N_{ens}
\end{equation}
```
The symbol $`\circ`$ is an element-by-element multiplication operator known as the Schur product.



### Forward problem
**Black oil model**;

 - Our simplified model for three-phase flow in porous media for reservoir simulation is given as [7].
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_w}}{{B_w}} u_w \right) - Q_w = -\frac{{\partial}}{{\partial t}} \left( \frac{{\varphi \rho_w}}{{B_w}} S_w \right) 
\end{equation}
```
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_o}}{{B_o}} u_o \right) - Q_o = -\frac{{\partial}}{{\partial t}} \left( \frac{{\varphi \rho_o}}{{B_o}} S_o \right)    
\end{equation}
```

```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g = -\frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right]
\end{equation}
```



$`φ(x)`$ stands for the porosity,subscript $`w`$ stands for water, subscript $`o`$ stands for oil and subscript $`g`$ stands for gas . $`T_{o},T_{w},T_{g},T`$ stands for the transmissibilities, which are known functions of the permeability $`K`$ the water saturation $`S_w`$ and gas saturation  $`S_g`$. $`u_{\gamma}`$ is the Darcy velocity of each phase.

```math
\begin{equation}
u_\gamma = -\frac{{k_{r\gamma}}}{{\mu_\gamma}} K (\nabla p_\gamma - \rho_\gamma \mathbf{g} \nabla z)
\end{equation}
```

Where:
- $`u_γ`$ is the Darcy velocity of phase γ.
- $`k_{rγ}`$ is the relative permeability of phase γ.
- $`μ_(γ)`$ is the viscosity of phase γ.
- $`B_(γ)`$ is the formation volume factor of phase γ.
- $`K`$ is the permeability.
- $`∇p_{γ}`$ is the gradient of pressure for phase γ.
- $`ρ_(γ)`$ is the density of phase γ.
- $`ĝ`$ is the gravitational vector.
- $`∇z`$ is the gradient of elevation.
- $`R_{so}`$ is the solution gas oil ratio.


```math
\begin{equation}
S_o + S_w + S_g = 1; P_{cwo} = p_o- p_w  ;P_{cog} = p_g- p_o  
\end{equation}
```


This gives six unknowns
```math
\begin{equation}
p_{o}, p_{w},p_{g}, S_{w}, S_{o}, S_{g}
\end{equation}
``` 
Gravity effects are considered by the terms 
```math
\begin{equation}
ρ_{w}gk,ρ_{o}gk\quad Ω \subset R^{n}(n = 2, 3) 
\end{equation}
``` 
The subsequent water, oil ,gas and overall transmissibilities is given by,
```math
\begin{equation} 
T_w = \frac{K(x) K_{rw}S_w}{μ_w}  ;T_o = \frac{K(x) K_{ro}S_o}{μ_o};T_g = \frac{K(x) K_{rg}S_g}{μ_g}   ;T= T_w + T_o + T_g           
\end{equation}
``` 
The relative permeabilities $`K_{rw} (S_w ),K_{ro} (S_w ),K_{rg} (S_o )`$ are available as tabulated functions, and $`μ_w,μ_o,μ_g`$ denote the viscosities of each phase. 

We define the oil flow, the water flow,gas flow and the total flow, respectively, which are measured at the well position as; 
```math
\begin{equation} 
Q = Q_{o}  + Q_{w } + Q_{g }
\end{equation}
``` 

The final pressure , water saturation and gas saturation equations for a three-phase oil-water flow is
```math
\begin{equation} 
- ∇ .[T∇p]=Q   
\end{equation}
```  
```math
\begin{equation}                                               
φ\frac{∂S_w}{∂t} - \nabla .[T_{w} (\nabla.p_{w}]= Q_{w} 
\end{equation}
``` 
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g = -\frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right]
\end{equation}
``` 
### Surrogate Forward modelling

**Fourier Neural operator based machine infused with physics constraint from black oil model ansatz**

An FNO model architecture, introduced in [54], is shown below.
![Nvidia-Energy](https://zongyi-li.github.io/assets/img/fourier_layer.png)

The goal is to replace the Finite volume simulator with an FNO surrogate.

For the PINO reservoir modelling [2], we are interested in predicting the pressure, saturation and fluxes given any input of the absolute permeability & porosity field for the pressure and saturation equation. We will introduce an additional (vector) variable, namely flux, F, which turns Eq. (3a) into a system of equations below. 

```math
\begin{equation} 
u=∇p ;    F = T∇p ;   - ∇ · F = Q
\end{equation}
``` 

Using a mixed residual loss formulation, the pressure equation loss ansatz is expressed as,
```math
\begin{equation} 
V(F,u;T)=\int_{Ω}[(F-T∇u)^2 + (- ∇ ·F-Q)^2 ]
\end{equation}
``` 
$`\Omega \subset \mathbb{R}^n \quad (n = 2, 3)`$. The discretised pressure, water saturation and gas saturation equation loss then become.

```math
\begin{equation} 
V(F,u;T)_{p} ≈ \frac{1}{n_{s}}  (‖F-T⨀∇u‖_{2}^{2} + ‖- ∇ ·F-Q‖_{2}^{2} )       
\end{equation}
``` 
```math
\begin{equation} 
V(u,S_w;t)_{S_{w}} = \frac{1}{n_s}  ‖(φ \frac{∂S_w}{∂t}- ∇ .[T_{w} (∇u)])-Q_w ‖_{2}^{2}     
\end{equation}
``` 

```math
\begin{equation} 
V(u,S_g,S_o;t)_{S_{g}} = \frac{1}{n_s}  ‖ \nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g + \frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right] ‖_{2}^{2}     
\end{equation}
``` 

```math
\begin{equation} 
Loss_{cfd} =V(F,u;T)_{p} + V(u,S_w;t)_{S_{w}} + V(u,S_g,S_o;t)_{S_{g}}      
\end{equation}
``` 

### Mixture of Experts
For a set of labelled data $` D=\{(x_i,y_i)\}_{i=1}^N`$ , where $`(x_i,y_i) \in \chi \times \gamma`$ is assumed to be the input and output of a model shown in Eqn 15.
```math
\begin{equation}
y_i \approx f(x_i)
\end{equation}
```
The output space is taken as  $`\gamma = \mathbb{R}`$ and $` \chi = \mathbb{R}^d`$

#### Cluster

In this stage, we seek to cluster the training input and output pairs.


$`\lambda : \chi \times \gamma \rightarrow L \stackrel{\text{def}}{=} \{ 1, \dots, L \} `$
where the label function minimizes,

```math
\begin{equation}
\Phi_{\text{clust}}(\lambda) = \sum_{l=1}^{L} \sum_{i \in S_l} l(x_i, y_i)
\end{equation}
```
```math
\begin{equation}
S_l = \{ (x_i, y_i); \lambda(x_i, y_i) = l \}
\end{equation}
```

```math
\begin{equation}
z_i = (x_i, y_i) , l_l = |z_i - \mu_l|^2
\end{equation}
```

 $`\mu_l = \frac{1}{\| S_l \|} \sum_{i \in S_l} z_i `$

 #### Classify

$` l_i = \lambda(x_i, y_i) `$
This is an expanded training set:

```math
\begin{equation}
\{ (x_i, y_i, l_i) \}_{i=1}^N 
\end{equation}
```

```math
\begin{equation}
f_c: \chi \rightarrow L 
\end{equation}
```

$`x \in \chi`$ provides an estimate $`f_c : x \mapsto f(x) \in L`$ such that $` f_c(x_i) = l_i `$ for most of the data. Crucial for the ultimate fidelity of the prediction. $`{ y_i }`$ is ignored at this phase. The classification function minimizes,
```math
\begin{equation}
\Phi_{\text{clust}}(f_c) = \sum_{i=1}^N \phi_c(l_i, f_c(x_i))
\end{equation}
```

$`\phi_c : L \times L \rightarrow R_+`$ is small if $` f_c(x_i) = l_i`$ for example we can choose $` f_c(x) = \arg\max_{l \in L} g_l(x)`$ where $` g_l(x) > 0, \sum_{l=1}^L g_l(x)`$ is a soft classifier and $` \phi_c(l_i, f_c(x_i)) = -\log(g_l(x))`$ is a cross-entropic loss.


#### Regress

```math
\begin{equation}
f_r : \chi \times L \rightarrow \gamma 
\end{equation}
```

For each $`(x,l)`$ in $`\chi \times L`$, we must provide an estimate 
$`f_r : (x,l) \mapsto f_r(x,l) \in \gamma `$
such that $`f_r(x, f_c(x)) \approx y`$ for both the training and test data. If successful, a good reconstruction for 

```math
\begin{equation}
f : \chi \rightarrow \gamma
\end{equation}
```

Where $`f(.) = f_r(., f_c(.))`$ . The regression function can be found by minimizing:

```math
\begin{equation}
\Phi_r(f_r) = \sum_{i=1}^N \phi_r(y_i, f_r(x_i, f_c(x_i))) 
\end{equation}
```

Where $`\phi_r : \gamma \times \gamma \rightarrow R_+`$ is minimized when $`f_r(x_i, f_c(x_i)) = y_i`$. In this case, it can be chosen as 

$` \phi_r(y, f_r(x, f_c(x))) = |y - f_r(x, f_c(x))|^2 `$. 
Data can be partitioned into 
$` C_l = \{ i; f_c(x_i) = l \}`$ 
for $`l = 1, \dots, L `$ and then perform $` L `$ separate regressions done in parallel.

```math
\begin{equation}
\Phi_r^l(f_r(.,l)) = \sum_{i \in C_l} \phi_r(y_i, f_r(x_i,l))
\end{equation}
```

#### Bayesian Formulation
A critique of this method is that it re-uses the data in each phase. A Bayesian postulation handles this limitation elegantly. Recall,

```math
\begin{equation}
D = \{ (x_i, y_i) \}_{i=1}^N 
\end{equation}
```


Assume parametric models for the classifier 
$`g_l( \cdot ; \theta_c) = g_l( \cdot ; \theta_c^l)`$ 
and the regressor 
$` f_r( \cdot , l; \theta_r^l )`$  
for $`l = 1, \dots, L`$  where 

```math
\begin{equation}
\theta_c = (\theta_c^1, \dots, \theta_c^L), \theta_r = (\theta_r^1, \dots, \theta_r^L) , \theta = (\theta_c, \theta_r), 
\end{equation}
```
the posterior density has the form,

```math
\begin{equation}
\pi( \theta , l | D ) \propto \prod_{i=1}^N \pi( y_i | x_i, \theta_r, l) \pi( l | x_i, \theta_c ) \pi( \theta_r ) \pi( \theta_c ) 
\end{equation}
```

```math
\begin{equation}
\pi( y_i | x_i, \theta_r, l ) \propto \exp\left(-\frac{1}{2} | y_i - f_r( x_i, l; \theta_r^l ) |^2\right)
\end{equation}
```

And

```math
\begin{equation}
\pi( l | x_i, \theta_c ) = g_l( x_i; \theta_c )
\end{equation}
```

```math
\begin{equation}
g_l( x_i; \theta_c ) = \frac{ \exp( h_l( x; \theta_c^l ) ) }{ \sum_{l=1}^L \exp( h_l( x; \theta_c^l ) ) } 
\end{equation}
```

```math
\begin{equation}
h_l( x; \theta_c^l )
\end{equation}
```
are some standard parametric regressors.


## Paper
- ArXiV paper :[link](https://arxiv.org/abs/2406.00889)

## Important Dependencies & Prerequisites:
- Nvidia's PhysicsNeMo symbolic :[link](https://github.com/NVIDIA/physicsnemo)
- CUDA 12.8 : [link](https://developer.nvidia.com/cuda-12-8-0-download-archive)
- CuPy : [link](https://github.com/cupy/cupy.git)
- Python 3.8 upwards

## Getting Started:
- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
- The code is developed in a Linux enviroment.

## Installation - Bare metal / Docker

- From terminal create a new conda enviroment named **MDLO** (check if conda is installed) .
```bash
conda create --name MDLO python=3.9
```

Clone this code base repository in a dedicated **work folder**.
```bash
cd **work folder**
conda activate MDLO
git lfs clone https://github.com/NVIDIA/physicsnemo.git
```
### Bare-metal
- From terminal do these sequence of operations to install PhysicsNeMo v24.09: [link](https://github.com/NVIDIA/physicsnemo.git)
```bash
pip install nvidia-physicsnemo          
```
- From terminal, install (missing) dependencies in 'requirements.txt' in the conda enviroment **MDLO**
- Follow instructions to install CuPy from : [link](https://github.com/cupy/cupy.git)



### PhysicsNeMo Virtual enviroment
```bash
chmod -R +x scripts
chmod -R +x src
./scripts/docker/install_modulus.sh
source modulus_env/bin/activate
```

### Docker (Recommended)
- You may need to temporarily deactivate any **vpn** for the docker installation & run
- Note, NVIDIA Container Toolkit must be installed first. This extension enables Docker daemon to treat GPUs properly.

- Please follow the installation instructions in [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

If you encounter a **Conflicting values set for option Signed** error when running apt update as shown below,

```bash
sudo apt-get update
E: Conflicting values set for option Signed-By regarding source https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/ /: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg !=
E: The list of sources could not be read.
```
Do the following
```bash
grep "nvidia.github.io" /etc/apt/sources.list.d/*
grep -l "nvidia.github.io" /etc/apt/sources.list.d/* | grep -vE "/nvidia-container-toolkit.list\$"
```
Delete the file(s) that will be shown from running the command above
```bash
sudo rm -f FILENAME
```
where FILENAME is the name of the file(s) shown above

More Troubleshooting can be found at [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/troubleshooting.html)

```bash
# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-build.sh
sudo chmod +x set_env.sh
# Build docker image
./scripts/docker/docker-build.sh

# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-run.sh
# Run docker container (also enables X server for docker)
./scripts/docker/docker-run.sh
```


### Run
**OPM Flow** is a fully CPU and GPU based Black oil reservoir simulator.
```bash
./scripts/docker/install_opm.sh
conda deactivate
```

[link](https://opm-project.org/?page_id=19 )

#### Forward problem


- Navigate to the code base root directory - **work folder** via terminal.

##### Bare Metal alone
```bash
cd **work folder**
```
- where **work folder** is the location you downloaded the code base to.

- Download the supplemental material.

- Run the Forward Problem surrogation with PINO  via the **src** folder.

##### Bare Metal
```bash
conda activate MDLO 
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/Modulus_reservoir_simulation_supplemental_material/versions/0.1/zip -O modulus_reservoir_simulation_supplemental_material_0.1.zip
unzip modulus_reservoir_simulation_supplemental_material_0.1.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
./Run_well_modelling.sh
cd ..
conda deactivate
```

##### Docker
```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/Modulus_reservoir_simulation_supplemental_material/versions/0.1/zip -O modulus_reservoir_simulation_supplemental_material_0.1.zip
unzip modulus_reservoir_simulation_supplemental_material_0.1.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
./Run_well_modelling.sh
cd ..
```


- Forward problem solution results are found in the root directory folder **outputs**

- Compare the surrogate solution from **PINO** with the finite volume reservoir simulator (**NVRS**) from the **src** folder.

- Results for the comparison are found in the root directory folder **COMPARE_RESULTS**

#### Inverse problem

- Inverse problem solution results are found in the root directory folder **HM_RESULTS**

## Pretrained models

- Pre-trained models and all necessary files are provided in the script for rapid prototyping & reproduction

- The Inverse_problem.py/Compare_FVM_Surrogate.py scripts can be ran without necessary running the PINO forward problem steps.



## Results
### Summary of Numerical Model
The result for the PINO surrogate is shown in Fig.2(a), 100 training samples was used were we compute the data loss and physics loss. The water flows from the injectors (downwards facing arrows) towards the producers (upwards facing arrows). The size of the reservoir computational voxel is nx, ny, nz = 46,112,22. Three phases are considered (oil,gas and water) and the wells (9 water injectors and 4 gas injectors and 22 producers) shown here [link](https://www.equinor.com/energy/norne) . The 22 producers well have measurable quantities of oil rate, gas rate and  water rate are controlled by bottom-hole-pressure. The 9 water injector wells have measurable quantity of bottom hole pressure (BHP), controlled by injection rates. 3,298 days of simulation are simulated. The left column of Fig.2(a) are the responses from the PINO surrogate, the middle column are the responses from the Flow finite volume solver  and the right column is the difference between each response. For all panels in Fig. 2(a), the first row is for the pressure, the second row is for the water saturation, the third row is for oil saturation and the fourth row is for the gas saturation

- The results from the Forward Problem using the PINO implementation
![alt text](Numerical_experiment/COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION/Evolution.gif)***Figure 2**: Numerical implementation of Reservoir forward simulation shwoing the pressure (first row), water saturation (second row), oil saturation (third row) and gas saturation (fourth row) evolution. PINO based reservoir forwarding (left column), OPM-Flow based reservoir forwarding (first principle), (middle-column) and the difference in magnitudes from both approaches (last-column) with the well locations*



| PINO - CCR | PINO - FNO |
| -----------|------------ |
| ![Image 1][img1] | ![Image 2][img2] |
| **(a) - Speed-up using the PINO-CCR machine** | **(b) - Speed-up using the PINO-FNO machine** |

***Figure 3**: Forwarding of the Norne Field. (a) Speed-up using the PINO-CCR machine, and (b) Speed-up using the PINO-FNO machine*


[img1]: Numerical_experiment/COMPARE_RESULTS/PINO/PEACEMANN_CCR/HARD_PREDICTION/Compare_time.png "Permeability Field ( 33 by 33 by 1)"
[img2]: Numerical_experiment/COMPARE_RESULTS/PINO/PEACEMANN_FNO/Compare_time.png "Permeability Field ( 40 by 40 by 3)"

| Qo (STB/DAY) | Qw (STB/DAY) |
|---------------------|---------------------|
| <img src="Numerical_experiment/COMPARE_RESULTS/Oil.png" alt="Image 1" width="600"> | <img src="Numerical_experiment/COMPARE_RESULTS/Water.png" alt="Image 2" width="600"> |
| **(a) - Outputs from peacemann machine for oil production rate** | **(b) - Outputs from peacemann machine for water production rate** |

| Qg (SCF/DAY) | RMSE FORWARD PROBLEM |
|---------------------|---------------------|
| <img src="Numerical_experiment/COMPARE_RESULTS/Gas.png" alt="Image 3" width="600"> | <img src="Numerical_experiment/COMPARE_RESULTS/Histogram.png" alt="Image 4" width="600"> |
| **(c) - Outputs from peacemann machine for gas production rate** | **(d) - RMSE values between the PINO-CCR & PINO-FNO machine** |


***Figure 4**: Forwarding of the Norne Field.Comparison between using the PINO -FNO machine and the PINO – CCR machine and the reference true data(a) Outputs from peacemann machine for oil production rate ,Qo from the 22 producers
They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. (b) Outputs from peacemann machine for water production rate ,Qw from the 22 producers. (c) Outputs from peacemann machine for gas production rate,Qg from the 22 producers and, (d) RMSE to a reference true data gotten from the Flow numerical solver, The PINO -CCR forward model workflow gave a lower RMSE value than the PINO -FNO forward model workflow*





![alt text](Numerical_experiment/HM_RESULTS/Localisation_matrix.png)***Figure 5**: Covariance localisation matrix generated from the $`5^{th}`$ order compact support Gaspari-Cohn correlation matrix*



- The results from the Inverse Problem using the **PINO -CCR** surrogate for forwarding

|Qo (STB/DAY)|Qw (STB/DAY)|Qg (SCF/DAY)|
|-------------------|-------------------|-------------------|
| ![Oil Production][img1b] | ![Water Production][img2b] | ![Gas Production][img3a] |
| **(a) - Oil Production Rate from Peacemann Machine** | **(b) - Water Production Rate from Peacemann Machine** | **(c) - Gas Production Rate from Peacemann Machine** |

|Qo (STB/DAY)|Qw (STB/DAY)|Qg (SCF/DAY)|
|-------------------|-------------------|-------------------|
| ![Oil Production][img4a] | ![Water Production][img5a] | ![Gas Production][img6a] |
| **(d) - Oil Production Rate from Peacemann Machine (Final)** | **(e) - Water Production Rate from Peacemann Machine** | **(f) - Gas Production Rate from Peacemann Machine** |

***Figure 6**: Inverse modeling with covariance localization and (aREKI)  oil production rate (Qo), water production rate (Qw), and gas production rate (Qg). Sub-figures (a-c) show the prior ensemble results, while subfigures (d-f) represent the posterior ensemble results.*

[img1b]: Numerical_experiment/HM_RESULTS/Oil_Initial.png.png 
[img2b]: Numerical_experiment/HM_RESULTS/Water_Initial.png.png  
[img3a]: Numerical_experiment/HM_RESULTS/Gas_Initial.png.png 
[img4a]: Numerical_experiment/HM_RESULTS/Oil_Final_cummulative_best.png.png
[img5a]: Numerical_experiment/HM_RESULTS/Water_Final_cummulative_best.png.png
[img6a]: Numerical_experiment/HM_RESULTS/Gas_Final_cummulative_best.png.png



| permeability fields |Qo(STB/DAY) |
|---------------------------------------|---------------------------------------|
| ![Image 1][img1c]                      | ![Image 2][img2c]                      |
| **(a) - Posterior permeability fields** | **(b) - Outputs from peacemann machine for oil production rate** |

|Qw (STB/DAY)| Qg (SCF/DAY) |
|---------------------------------------|-----------------------|
| ![Image 3][img3c]                      | ![Image 4][img4c]     |
| **(c) - Outputs from peacemann machine for water production rate** | **(d) - Outputs from peacemann machine for gas production rate** |

***Figure 7**: Inverse modelling of the Norne Field with covariance localisation and aREKI. (a) Permeability field recovered after the inverse modelling showing the **prior/posterior P10/posterior P50/ posterior P90/ posterior cumulative P10/posterior cumulative P50 reservoir models**. (b-d) Qo , Qw and gas production rate,Qg from the 22 producers for the **prior/posterior P10/posterior P50/posterior P90/posterior cumulative P10/posterior cumulative P50 reservoir models** to the true model*

[img1c]: Numerical_experiment/HM_RESULTS/PERCENTILE/Reservoir_models.png 
[img2c]: Numerical_experiment/HM_RESULTS/PERCENTILE/Oil.png  
[img3c]: Numerical_experiment/HM_RESULTS/PERCENTILE/Water.png 
[img4c]: Numerical_experiment/HM_RESULTS/PERCENTILE/Gas.png 


| MAP Reservoir model |
|---------------------------------------|
| ![Image 1][img1e]                      |
| **(a) - MAP Reservoir model showing the pressure (top left), water saturation (top right), oil saturation (bottom left) and gas saturation (bottom right)** |

|MLE Reservoir model | 
|---------------------------------------|
| ![Image 2][img2e]                      |
| **(c) - MLE Reservoir model showing the pressure (top left), water saturation (top right), oil saturation (bottom left) and gas saturation (bottom right)** |

***Figure 8**: Inverse modelling of the Norne Field with covariance localisation and aREKI. (a) Permeability field recovered after the inverse modelling showing the **prior/posterior P10/posterior P50/ posterior P90/ posterior cumulative P10/posterior cumulative P50 reservoir models**. (b-d) Qo , Qw and gas production rate,Qg from the 22 producers for the **prior/posterior P10/posterior P50/posterior P90/posterior cumulative P10/posterior cumulative P50 reservoir models** to the true model*

[img1e]: Numerical_experiment/HM_RESULTS/ADAPT_REKI/Evolution.gif 
[img2e]: Numerical_experiment/HM_RESULTS/MEAN_RESERVOIR_MODEL/Evolution.gif 


![alt text](Numerical_experiment/HM_RESULTS/PERCENTILE/Histogram.png)***Figure 9**: Inverse modelling of the Norne Field with covariance localisation and aREKI. RMSE value showing the performance of the 6 reservoir model accuracy to the True data to be matched*



## Release Notes

**23.01**
* First release 

## End User License Agreement (EULA)
Refer to the included Energy SDK License Agreement in **Energy_SDK_License_Agreement.pdf** for guidance.

## Author:
- Clement Etienam- DevTech-Energy @Nvidia  Email: cetienam@nvidia.com

## Contributors:
- Juntao Yang- Nvidia
- Issam Said- Nvidia
- Oleg Ovcharenko- Nvidia
- Nick Luiken- Nvidia
- Kaustubh Tangsali- Nvidia



## References:
[1] Aanonsen, S., Oliver, D., Reynolds, A. & Valles, B., 2009. The Ensemble Kalman Filter in Reservoir Engineering--a Review. SPE Journal, 14(3), pp. 393-412.

[2] Aharon, M., Elad, M. & Bruckstein, A., 2006. K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation. IEEE Transactions on Signal Processing, 54(11), pp. 4311-4322.

[3] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[4] Candes, E. J., Romberg, J. & Tao, T., 2006. Robust uncertainty principles: Exact signal reconstruction from high incomplete frequency information. Information Theory, IEEE Transactions on, 52(2), pp. 489-509.

[5] Chang, H., Zhang, D. & Lu, Z., 2010. History matching of facies distribution with the EnKF and level set parameterization. Journal of Computational Physics, 229(1), pp. 8011-8030.

[6] Chen, Y. and Oliver,D., .2011 Ensemble randomized maximum likelihood method as an iterative ensemble smoother, Mathematical Geosciences, Online First.


[7] Dorn, O. & Villegas, R., 2008. History matching of petroleum reservoirs using a level set technique. Inverse problems, p. 24.


[8] Elsheikh, A., Wheeler, M. & Hoteit, I., 2013. Sparse calibration of subsurface flow models using nonlinear orthogonal matching pursuit and an iterative stochastic ensemble method. Advances in Water Resources, 56(1), pp. 14-26.

[9] Emerick, A. A. and Reynolds A. C., 2013 Ensemble smoother with multiple data assimilation, Computers & Geosciences.

[10] Etienam, C., Mahmood, I. & Villegas, R., 2017. History matching of Reservoirs by Updating Fault Properties Using 4D Seismic Results and Ensemble Kalman Filter. Paris, SPE.

[11] Etienam, C., Villegas, R., Babaei, M & Dorn,O 2018. An Improved Reservoir Model Calibration through Sparsity Promoting ES-MDA. ECMI conference proceeding. 20th European Conference on Mathematics for Industry (ECMI) 18-22 June 2018, Budapest, Hungary 

[12] Etienam C. 4D Seismic History Matching Incorporating Unsupervised Learning, Society of Petroleum Engineers B (2019 June), 10.2118/195500-MS

[13] Etienam C., Law, J.H L., Wade, S., Ultra-fast Deep Mixtures of Gaussian Process Experts. https://arxiv.org/abs/2006.13309.

[14] Evensen, G., 2003. The Ensemble Kalman Filter: Theoretical formulation and practical implementation. Ocean Dynamics, 53(4), pp. 343-367.

[15] Evensen, G.,2009. The ensemble Kalman filter for combined state and parameter estimation, IEEE Control Systems Magazine, pp. 83-104.

[16] Evensen, G., Eikrem, K, S 2018. Conditioning reservoir models on rate data using ensemble smoothers.Computational Geosciences,22(5), pp.1251-1270.

[17] Haibin, C. & Zhang, D., 2015. Jointly updating the mean size and spatial distribution of facies in reservoir history matching. Computational Geosciences.

[18] Hanke, M.,1997. A regularizing Levenberg-Marquardt scheme, with applications to inverse groundwater filtration problems. Inverse problems 13(1), pp. 79–95 

[19] Iglesias, M.A., Dawson, C.2013. The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs. Computational Geosciences 17, pp.1033–1053 

[20] Jafarpour, B., 2011. Wavelet reconstruction of geologic facies from nonlinear dynamic flow measurements. Geosciences and Remote sensing, IEEE Transactions, 49(5), pp. 1520-1535.

[21] Jafarpour, B. & McLaughlin, D. B., 2007. History matching with an ensemble Kalman filter and discrete cosine parametrization. Anaheim, California, SPE.

[22] Khaninezhad, M. M., Jafarpour, B. & Li, L., 2012. Sparse geologic dictionaries for subsurface flow model calibration: Part 1, Inversion formulation. Advances in Water Resources, 39(1), pp. 106-121.

[23] Kim S, Min B, Lee K,& Jeong H, 2018 Integration of an Iterative Update of Sparse Geologic Dictionaries with ES-MDA for History Matching of Channelized Reservoirs. Geofluids, Volume 2018, Article ID 1532868, 21 pages.

[24] Law K J H & Stuart A M, 2012 Evaluating Data Assimilation Algorithms Mon.  Weather  Rev 140 37-57

[25] Liu, E. & Jafarpour, B., 2013. Learning sparse geologic dictionaries from low-rank representations of facies connectivity for flow model calibration. Water resources, Volume 49, pp. 7088-7101.


[26] Lorentzen, R. J., Flornes, M. K. & Naevdal, G., 2012. History matching Channelized Reservoirs Using the Ensemble Kalman Filter. Society of Petroleum Engineers, 17(1).


[27] Luo X, Bhakta T, Jakobsen M, Naevdal G, 2018. Efficient big data assimilation through sparse representation: A 3D benchmark case study in petroleum engineering. PLoS ONE 13(7): e0198586. https://doi.org/10.1371/journal.pone.0198586


[28] Moreno, D. L., 2011. Continuous Facies Updating Using the Ensemble Kalman Filter and the Level set method. s.l., Mathematical Geosciences.

[29] Nocedal, J. and Wright, S.J, 1999.Numerical Optimization,  Springer, New York.

[30] Oliver, D. S. & Chen, Y., 2010. Recent progress on reservoir history matching: a review. s.l.: Computational Geoscience - Springer Science.

[31] Oliver, D. S., Reynolds, A. C. & Liu, N., 2008. Inverse Theory for Petroleum Reservoir Characterization and History Matching. s.l.:Cambridge University Press.


[32] Sana, F, Katterbauer, K , Al-Naffouri, T.Y and Hoteit, I.,2016 Orthogonal matching pursuit for enhanced recovery of sparse geological structures with the ensemble Kalman filter, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 9, no. 4, pp. 1710–1724

[33] Sarma, P., L. J. Durlofsky, and K. Aziz,.2008. Kernel principal component analysis for efficient differentiable parameterization of multipoint geostatistics, Mathematical Geosciences, 40, 3-32, 2008.


[34] Sherman, J., Morrison, W.J.1950. Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. The Annals of Mathematical Statistics 21, 124–127.


[35] Stuart A M ,2010 Inverse problems:  A Bayesian perspective Acta Numerica 19 451-559.

[36] Tarantola, 2005. Inverse problem theory and methods for model parameter estimation. 1 ed. Philadelphia: SIAM.

[37] Tarrahi, M. & Afra, S., 2016. Improved Geological model calibration through Sparsity-promoting Ensemble Kalman Filter. Offshore Technology Conference.

[38] Tropp, J. A. & Gilbert, C. A., 2007. Signal recovery from random measurements via orthogonal matching pursuit. Information Theory, IEEE Transactions on, 53(12), pp. 4655-4666.


[39] Villegas, R., Etienam, C., Dorn, O., & Babaei, M.,2018. The shape and Distributed Parameter Estimation for History Matching using a Modified Ensemble Kalman Filter and Level Sets. ;  Inverse problems Science and Engineering.


[40] William P, Mitchell J,1993 JPEG: Still Image Data Compression Standard, Van Nostrand Reinhold, 

[41] Wu, J., Boucher, A. & Journel, G. A., 2006. A 3D code for mp simulation of continuous and categorical variables: FILTERSIM. SPE.

[42] Zhou, H,   Li, L and. Gómez-Hernández, J.,2012, “Characterizing curvilinear features using the localized normal-score ensemble Kalman filter,” Abstract and Applied Analysis, vol. 2012, Article ID 805707, 18 pages.

[43] L. Yang, D. Zhang, G. E. Karniadakis, Physics-informed generative adversarial networks for stochastic differential equations, arXiv preprint arXiv:1811.02033

[43] J. Adler, O. Oktem, solving ill-posed inverse problems using iterative ¨ deep neural networks, Inverse Problems 33 (12) (2017) 124007. URL http://stacks.iop.org/0266-5611/33/i=12/a=124007

[44] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, E. Shechtman, Toward multimodal image-to-image translation, in Advances in Neural Information Processing Systems, 2017, pp. 465–476.

[45] S. Rojas, J. Koplik, Nonlinear flow in porous media, Phys. Rev. E 58 (1998) 4776–4782.doi:10.1103/PhysRevE.58.4776.URL,https://link.aps.org/doi/10.1103/PhysRevE.58.4776

[46] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, K. Q. Weinberger (Eds.), Advances in Neural Information Processing Systems 27, Curran Associates, Inc., 2014, pp. 2672–2680. URL http://papers.nips.cc/paper/5423-generative-adversarial-nets. pdf

[47] M. Raissi, P. Perdikaris, G. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686 – 707. doi:https://doi.org/10.1016/j.jcp.2018.10.045.URL,http://www.sciencedirect.com/science/article/pii/ S0021999118307125

[48] M. Raissi, Forward-backward stochastic neural networks: Deep learning of high-dimensional partial differential equations, arXiv preprint arXiv:1804.07010

[49] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics Informed Deep Learning (Part I): Data-driven solutions of nonlinear partial differential equations, arXiv preprint arXiv:1711.10561

[50] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[51] Dorn O & Villegas R, 2008 History matching of petroleum reservoirs using a level set technique, Inverse problems Volume: 24   Issue: 3 Article Number: 035015 

[52] Hansen, T.M., Vu. L.T., and Bach, T. 2016. MPSLIB: A C++ class for sequential simulation of multiple-point statistical models, in Softwar X, doi:10.1016/j.softx.2016.07.001. [pdf,www].

[53] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 1(2639-8001-2019-4-491):491, 2019.

[54] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.08895

[55] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf

<!-- markdownlint-enable -->