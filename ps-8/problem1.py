import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import scipy.optimize as optimize

# Data: Age and recognition responses (0 = no, 1 = yes)
data = np.loadtxt("ps-8/survey.csv", skiprows=1, delimiter=",")
ages = data[:, 0]  # Age data
reco = data[:, 1]  # Responses (0 or 1)

# Logistic function for probability p(x) based on parameters
def p(x, params):
    beta0, beta1 = params
    return 1 / (1 + jnp.exp(-(beta0 + beta1 * x)))

# Negative log-likelihood function
def negloglike(params, x, responses):
    probabilities = p(x, params)
    # Avoid log(0) by adding a tiny epsilon inside log function
    epsilon = 1e-10
    log_likelihood = responses * jnp.log(probabilities + epsilon) + (1 - responses) * jnp.log(1 - probabilities + epsilon)
    return -jnp.sum(log_likelihood)

# Gradient and Hessian of the negative log-likelihood
negloglike_grad = jax.grad(negloglike)
negloglike_hessian = jax.jacfwd(jax.grad(negloglike))

# Initial guess for parameters beta0 and beta1
initial_params = np.array([-5, 0.8])

# Minimize the negative log-likelihood
result = optimize.minimize(
    fun=negloglike,
    x0=initial_params,
    args=(ages, reco),  # Pass ages and reco as additional arguments
    jac=negloglike_grad,
    method="Newton-CG",
    tol=1e-6
)
print(f"result {result}")
# Extract optimal parameters, covariance, and standard errors
optimal_params = result.x
hessian_matrix = negloglike_hessian(optimal_params, ages, reco)
covariance_matrix = np.linalg.inv(hessian_matrix)
standard_errors = np.sqrt(np.diag(covariance_matrix))

# Print results
print("Optimal parameters (beta0, beta1):", optimal_params)
print("Covariance matrix:\n", covariance_matrix)
print("Standard errors:", standard_errors)

# Get the indices that would sort the ages array
sorted_indices = np.argsort(ages)

# Use the sorted indices to sort both arrays
sorted_ages = ages[sorted_indices]
sorted_reco = reco[sorted_indices]

# Summing the responses w/ respect to sorted ages to get a pdf I can match
response_pdf = []
i = 0
sum = 0
for _ in sorted_ages:
    sum += sorted_reco[i]
    i+=1
    response_pdf.append(sum)
response_pdf = np.array(response_pdf)/np.sum(reco)

# Calculate predicted probabilities using the optimal parameters
predicted_probs = 1 / (1 + np.exp(-(optimal_params[0] + optimal_params[1] * sorted_ages)))

# Plot data points and logistic model
plt.figure(figsize=(8, 6))
plt.scatter(sorted_ages, sorted_reco, color="red", label="Survey Responses (0 or 1)", s= 10)
plt.plot(sorted_ages, predicted_probs, color="blue", label="Logistic Model (Predicted Probability)")
plt.xlabel("Age")
plt.ylabel("Probability of Recognizing 'Be Kind, Rewind'")
plt.title("Logistic Regression Model of Recognition vs. Age")
plt.legend()
plt.grid(True)
plt.savefig("ps-8/plots/ages")