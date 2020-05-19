import numpy as np
import distributions.Gamma as Gamma
import distributions.Gaussian as Gaussian


def main():
    print("Get samples from generative process")
    mu = 10
    sigma = 50
    nb_samples = 1000
    samples = np.random.normal(mu, sigma, nb_samples)

    print("Create generative model and variational distribution")
    prior_a = 0
    prior_b = 1000
    PY = Gaussian.Gaussian(prior_a, prior_b)
    prior_c = 0
    prior_d = 1000
    PZ = Gamma.Gamma(prior_c, prior_d)
    PX = Gaussian.Gaussian(-1, -1)

    posterior_a = prior_a
    posterior_b = prior_b
    QY = Gaussian.Gaussian(posterior_a, posterior_b)
    posterior_c = prior_c
    posterior_d = prior_d
    QZ = Gamma.Gamma(posterior_c, posterior_d)

    print("Perform variational message passing")
    for s in np.nditer(samples):
        for i in range(0, 10):
            # Update mean posterior
            QY.set_params(PY.params() + PX.mean_message(PX.observed_expectations(s), QZ.hidden_expectations()))
            # Update precision posterior
            QZ.set_params(PZ.params() + PX.precision_message(PX.observed_expectations(s), QY.hidden_expectations()))
        # Posterior as new prior
        PY.set_params(QY.params())
        PZ.set_params(QZ.params())

    print("Parameters of Gaussian over mean: (" + str(PY.mean) + "," + PY.precision + ")")
    print("Parameters of Gamma over precision: (" + str(PZ.shape) + "," + PZ.scale + ")")

    print("Done.")


if __name__ == "__main__":
    main()
