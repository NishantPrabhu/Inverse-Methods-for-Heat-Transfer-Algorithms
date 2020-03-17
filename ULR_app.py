
import numpy as np
import pandas as pd
import streamlit as st

st.title("Univariate Linear Regression")
st.write("### Parameter estimation by solving linear system of equations")
st.write("We can perform linear regression to find parameters of systems which can be modeled with linear equations. In this notebook, we will use temperature data obtained from 1D steady-state heat flow across an infinitely long plate to find the equation governing spatial variation of temperature along the plate's width. In our case, we have only one feature - the distance from one end of the plate - which means we'll have two parameters to estimate.")

with st.echo():
    # Dependencies
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

st.write("## Generating data from the forward model")
st.write("Let's create some data which follows the model. This will be used to test the regressor we program later. The forward model for 1D conduction is given by the expression on the left hand side. For steady-state conduction with no heat generation (we will relax this assumption later) and constant thermal conductivity $k$, the equation reduces to one on the right.")

r'''
$$ 
\frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right)+q_{v}=\rho C_{p}\frac{\partial T}{\partial t}\quad\Rightarrow\quad\frac{\partial^{2}T}{\partial x^{2}}=0\quad\Rightarrow\quad \boxed{T(x)=\alpha_{0}+\alpha_{1}x}
$$
'''
st.write('')
r'''The true values of $\alpha_{0}$ and $\alpha_{1}$ are dictated by the system's boundary conditions. $\alpha_{1}$ is proportional to the heat flux through the system and $\alpha_{0}$ is the temperature at $x=0$. For our case, assume we're using a steel plate ($k \approx 50\; W/mK$) with a heat flux $q = 1000\;W/m^{2}$ flowing from one edge of it. Let the location from where heat flows ($x=0$) be at a steady state temperature of $300\;K$ (this depends on how heat leaves the system, say by natural convection, radiation, or its variants).'''

r'''This gives us the expected values of $\boxed{\alpha_{0}=300\;K}$ and $\boxed{\alpha_{1}=-20\;K/m}$. We'll generate 100 data points for a plate which is 1 m wide, such that temperature is recorded every 1 cm. Also, we'll add normally distributed noise with mean = 0 and standard deviation = 1 to the data, to loosely account for measurement errors. While training, we will ignore the readings for $x=0.0\;m$ and $x=1.0\;m$. We'll find these using the line fit by our model and compare them with expected values.'''

r'''Play around with your own settings of $\alpha_{0}$ and $\alpha_{1}$ below!'''
st.write('')

# Main code
alpha_0 = st.number_input("Alpha 0", 300)
alpha_1 = st.number_input("Alpha 1", -20)
n_samples = st.number_input("Number of samples", 100)

st.write("Change the above values and hit Enter to see the plots change accordingly. Here is some code that will generate our data.")

with st.echo():
    @st.cache
    # Data generation function
    def generate_data(alpha_0, alpha_1, n_samlpes):
        x = np.linspace(0, 1.0, n_samples+1)    
        T_real = alpha_0 + alpha_1*x                                

        # Adding noise
        noise = np.random.normal(loc=0, scale=1, size=len(T_real))   
        T_noisy = T_real + noise 
        return x, T_real, T_noisy                                      

    # Generate data
    x, T_real, T_noisy = generate_data(alpha_0, alpha_1, n_samples)

st.write("Let's plot the data and verify that it indeed is linear.")
    
with st.echo():   
    # Display the data on plots
    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(121)
    ax1.scatter(x, T_real, marker='.', color='blue', alpha=0.7)
    ax1.set_title('Expected temperature vs. x')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Temperature (K)')
    ax1.grid()

    ax2 = fig.add_subplot(122)
    ax2.scatter(x, T_noisy, marker='.', color='red', alpha=0.7)
    ax2.set_title('Measured temperature vs. x')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('Temperature (K)')
    ax2.grid()

    plt.tight_layout()
    plt.show()

# Show data on streamlit
st.pyplot(fig)

st.write("On the left you have the expected behaviour of temperature with $x$. On the right, we have the same data after adding noise.")

r'''
### **Loss Function and Optimization**

If our fit line approximates temperature (given $x$) well, the error between its prediction and the actual value of temperature should be minimum. Esentially, we would like to solve an optimization problem to obtain those values of parameters which minimize the error function (also called loss function). There are various ways to aggregate the errors from all predictions; the **sum of squared errors** is most widely used, since it always provides a unique solution. It has two important properties:
(1) Positive and negative errors are treated equally, and
(2) Larger errors are penalized more than smaller errors

The functional form of SSE is as follows. This optimization problem is therefore called **Least Squares Regression** (LSR in short). 

$$
\mathcal{L}(\alpha)=\frac{1}{2m}\sum_{i=1}^{m}\left(T^{(i)}-T_{fit}^{(i)}\right)^{2}=\frac{1}{2m}\sum_{i=1}^{m}\left(T^{(i)}-\alpha_{0}-\alpha_{1}x^{(i)}\right)^{2}
$$

We take an average over all examples to keep the numbers small. We also divide by 2 for computational convenience, which will be evident soon. For the chosen values of parameters, let's see what our loss function looks like.
'''

with st.echo():
    def get_sse(alpha_0, alpha_1, x, y):
        return (0.5/len(x))*np.sum((y - alpha_0 - alpha_1*x)**2)

    # Create a meshgrid of alpha space and get SSE for each pair of parameters
    sse_vals = []
    a0, a1 = np.meshgrid(np.arange(200, 400, 1), np.arange(-120, 80, 1))
    for pair in np.c_[a0.ravel(), a1.ravel()]:
        sse_vals.append(get_sse(pair[0], pair[1], x, T_real))
    y_grid = np.array(sse_vals).reshape(a0.shape)

    # Plot the surface
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(a0, a1, y_grid, rstride=1, cstride=1, cmap='jet')
    ax.set_xlabel('alpha_0')
    ax.set_ylabel('alpha_1')
    ax.set_zlabel('MSE')
    plt.title('Loss function surface')
    plt.show()

st.pyplot(fig)

r'''
Not the best view for a paraboloid, but it forms one with a minima at the set values of $\alpha_{0}$ and $\alpha_{1}$. In fact, this function has a global minimum at these values. Where this function attains its minimum, we expect its derivative to be zero with respect to both $\alpha_{0}$ and $\alpha_{1}$. We get two equations from that:

$$
\frac{\partial\mathcal{L}}{\partial\alpha_{0}}=-\frac{1}{m}\sum_{i=1}^{m}\left(T^{(i)}-\alpha_{0}-\alpha_{1}x^{(i)}\right)=0
$$
$$
\frac{\partial\mathcal{L}}{\partial\alpha_{1}}=-\frac{1}{m}\sum_{i=1}^{m}\left(T^{(i)}-\alpha_{0}-\alpha_{1}x^{(i)}\right)\cdot x^{(i)}=0
$$

This can be compactly (and conveniently) transformed into this matrix equation ($m$ is the number of measurements).

$$
\left[\begin{array}{cc}
m & \sum x^{(i)}\\
\sum x^{(i)} & \sum\left(x^{(i)}\right)^{2}
\end{array}\right]\cdot\left[\begin{array}{c}
\alpha_{0}\\
\alpha_{1}
\end{array}\right]=\left[\begin{array}{c}
\sum T^{(i)}\\
\sum T^{(i)}x^{(i)}
\end{array}\right]
$$

In the next block of code, we implement this. 
'''

with st.echo():
    # Perform the matrix operations to get alpha_0 and alpha_1
    A = np.array([[len(x), sum(x)],
                [sum(x), sum(x**2)]])
    b = np.array([[sum(T_noisy)],
                [sum(T_noisy * x)]])

    fit_params = np.dot(np.linalg.inv(A), b)

st.write("The estimated values of our parameters are stored in `fit_params` now. Let's use those to see what our fit line looks like.")

with st.echo():
    # Visualizing the fit line
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, T_noisy, color='red', marker='o')
    x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    fit_vals = fit_params[0] + fit_params[1]*x_vals
    ax.scatter(x_vals, fit_vals, color='black', marker='.')
    plt.title('Fit line')
    plt.xlabel('x (m)')
    plt.ylabel('Temperature (K)')
    plt.grid()
    plt.show()

st.pyplot(fig)
st.write("Estimated value of Alpha 0 $\quad$ = $\quad$ **{:.4f}**".format(fit_params[0][0]))
st.write("Estimated value of Alpha 1 $\quad$ = $\quad$ **{:.4f}**\n".format(fit_params[1][0]))

st.write("It seems like we have estimated the parameters really well. Below is a parity plot showing how our predictions look against actual values of temperature. The black line is indicates the ideal distribution of measured and predicted values (equal), and has slope = 1. All points above the line were underestimated and those below were overestimated by our model.")

with st.echo():
    # Get temperature predictions for x with estimated parameters
    predicted_T = fit_params[0] + fit_params[1]*x

    # Parity plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(predicted_T, T_noisy, c='red', alpha=0.6)
    lim = np.linspace(*ax.get_xlim())
    ax.plot(lim, lim, color='black')
    plt.xlabel('Predictions')
    plt.ylabel('Target')
    plt.title('Parity plot')
    plt.grid()
    plt.show()

st.pyplot(fig)

r'''
## **Measuring goodness of fit**

We use certain metrics to quantify the goodness of our model's fit to the data. Three of them are **coefficient of determination** ($R^{2}$), **correlation coefficient** ($R$) and **standard error** ($SE$). They are calculated using the expressions shown below. $\overline{T}$ is the mean of measured temperatures.

$$
S_{t}=\sum_{i=1}^{m}\left(T^{(i)}-\overline{T}\right)^{2} \quad ; \quad S_{r}=\sum_{i=1}^{m}\left(T^{(i)}-T_{fit}^{(i)}\right)^{2}
$$

$$
R^{2}=\frac{S_{t}-S_{r}}{S_{t}} \quad ; \quad R=\sqrt{\frac{S_{t}-S_{r}}{S_{t}}} \quad ; \quad SE=\sqrt{\frac{S_{r}}{m-2}}
$$
'''

with st.echo():
    @st.cache
    def get_residuals(x, T_noisy, predicted_T):

        # Make a dataframe with x, T_noisy and predicted_T
        data = np.vstack((x.reshape((1, -1)), T_noisy.reshape((1, -1)), predicted_T.reshape((1, -1)))).T
        df = pd.DataFrame(data, columns=['x', 'T_measured', 'T_fit'])

        df['Mean_deviation'] = (df['T_measured'] - df['T_measured'].mean())**2
        df['Fit_deviation'] = (df['T_measured'] - df['T_fit'])**2
        return df

    resid_df = get_residuals(x, T_noisy, predicted_T)

st.write("Here's the table with residuals calculated.\n")
st.write(resid_df)

st.write("Let's have a look at the goodness of fit metric values as well.\n")

with st.echo():
    def goodness_of_fit(x, resid_df):
        # Calculate goodness of fit parameters
        S_t, S_r = resid_df['Mean_deviation'].sum(), resid_df['Fit_deviation'].sum()
        R_2 = (S_t - S_r)/S_t
        SE = np.sqrt(S_r/(len(x) - 2))
        return R_2, SE

    R_2, SE = goodness_of_fit(x, resid_df)

st.write("Coefficient of determination $\quad$ : $\quad$ **{:.4f}**".format(R_2))
st.write("Correlation coefficient $\quad$ : $\quad$ **{:.4f}**".format(R_2**0.5))
st.write("Standard error for fit $\quad$ $\quad$ : $\quad$ **{:.4f}**\n".format(SE))

st.write("The correlation coefficient is very close to 1.0, indicating that our fit is pretty good. Thus, we have successfully fit a linear curve to our data.")