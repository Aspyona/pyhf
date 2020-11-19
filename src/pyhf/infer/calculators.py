"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit, fit
from ..optimize.opt_minuit import minuit_optimizer
from .. import get_backend, set_backend
from .test_statistics import qmu, qmu_tilde, tmu_tilde, tmu
import tqdm
from scipy.optimize import minimize
from numpy import argmin, linspace
import numpy as np


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds, fixed_params):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Example:

        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> pyhf.infer.calculators.generate_asimov_data(mu_test, data, model, None, None, None)
        array([ 60.61229858,  56.52802479, 270.06832542,  48.31545488])

    Args:
        asimov_mu (:obj:`float`): The value for the parameter of interest to be used.
        data (:obj:`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
        par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
        fixed_params (:obj:`tensor`): Parameters to be held constant in the fit.

    Returns:
        Tensor: The Asimov dataset.

    """
    bestfit_nuisance_asimov = fixed_poi_fit(
        asimov_mu, data, pdf, init_pars, par_bounds, fixed_params
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


class AsymptoticTestStatDistribution(object):
    r"""
    The distribution the test statistic in the asymptotic case.

    Note: These distributions are in :math:`-\hat{\mu}/\sigma` space.
    In the ROOT implementation the same sigma is assumed for both hypotheses
    and :math:`p`-values etc are computed in that space.
    This assumption is necessarily valid, but we keep this for compatibility reasons.

    In the :math:`-\hat{\mu}/\sigma` space, the test statistic (i.e. :math:`\hat{\mu}/\sigma`) is
    normally distributed with unit variance and its mean at
    the :math:`-\mu'`, where :math:`\mu'` is the true poi value of the hypothesis.
    """

    def __init__(self, shift):
        """
        Asymptotic test statistic distribution.

        Args:
            shift (:obj:`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.sqrtqmuA_v = None

    def cdf(self, value):
        """
        Compute the value of the cumulative distribution function for a given value of the test statistic.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> bkg_dist.pvalue(0)
            0.5

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a test statistic less than or equal to the observed ``value``.

        """
        tensorlib, _ = get_backend()
        return tensorlib.normal_cdf((value - self.shift))

    def pvalue(self, value):
        r"""
        The :math:`p`-value for a given value of the test statistic corresponding
        to signal strength :math:`\mu` and Asimov strength :math:`\mu'` as
        defined in Equations (59) and (57) of :xref:`arXiv:1007.1727`

        .. math::

            p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(\sqrt{q_{\mu}} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

        with Equation (29)

        .. math::

            \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

        given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        # computing cdf(-x) instead of 1-cdf(x) for right-tail p-value for improved numerical stability
        return tensorlib.normal_cdf(-(value - self.shift))

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> n_sigma = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> bkg_dist.expected_value(n_sigma)
            array([-2., -1.,  0.,  1.,  2.])

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        return self.shift + nsigma


class AsymptoticCalculator(object):
    """The Asymptotic Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        qtilde=True,
    ):
        """
        Asymptotic Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
            par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
            qtilde (:obj:`bool`): When ``True`` use :func:`~pyhf.infer.test_statistics.qmu_tilde`
             as the test statistic.
             When ``False`` use :func:`~pyhf.infer.test_statistics.qmu`.
            qtilde (:obj:`bool`): When ``True`` perform the calculation using the alternative test statistic,
             :math:`\\tilde{q}`, as defined in Equation (62) of :xref:`arXiv:1007.1727`
             (:func:`~pyhf.infer.test_statistics.qmu_tilde`).
             When ``False`` use :func:`~pyhf.infer.test_statistics.qmu`.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()

        self.qtilde = qtilde
        self.sqrtqmuA_v = None

    def distributions(self, poi_test, b_dist=True, sb_dist=True):
        """
        Probability distributions of the test statistic, as defined in
        :math:`\S` 3 of :xref:`arXiv:1007.1727` under the Wald approximation,
        under the signal + background and background-only hypotheses.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, qtilde=True)
            >>> _ = asymptotic_calculator.teststatistic(mu_test)
            >>> qmu_sig, qmu_bkg = asymptotic_calculator.distributions(mu_test)
            >>> qmu_sig.pvalue(mu_test), qmu_bkg.pvalue(mu_test)
            (0.002192624107163899, 0.15865525393145707)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Tuple (~pyhf.infer.calculators.AsymptoticTestStatDistribution): The distributions under the hypotheses.

        """
        if self.sqrtqmuA_v is None:
            raise RuntimeError('need to call .teststatistic(poi_test) first')
        if sb_dist:
            sb_dist = AsymptoticTestStatDistribution(-self.sqrtqmuA_v)
            if not b_dist:
                return sb_dist
        if b_dist:
            b_dist = AsymptoticTestStatDistribution(0.0)
            if not sb_dist:
                return b_dist
        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, qtilde=True)
            >>> asymptotic_calculator.teststatistic(mu_test)
            0.14043184405388176

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Float: The value of the test statistic.

        """
        tensorlib, _ = get_backend()

        teststat_func = qmu_tilde if self.qtilde else qmu

        qmu_v = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 0.0  # ?!!
        asimov_data = generate_asimov_data(
            asimov_mu,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        qmuA_v = teststat_func(
            poi_test,
            asimov_data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        self.sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

        if not self.qtilde:  # qmu
            teststat = sqrtqmu_v - self.sqrtqmuA_v
        else:  # qtilde

            def _true_case():
                teststat = sqrtqmu_v - self.sqrtqmuA_v
                return teststat

            def _false_case():  # correct for bkg only case, bc mu_prime = 0, for sig+bkg case it turns into (qmu + qmu_A) / (2 * sqrtqmuA_v) where qmu_A is calculated with mu_prime = 0
                qmu = tensorlib.power(sqrtqmu_v, 2)
                qmu_A = tensorlib.power(self.sqrtqmuA_v, 2)
                teststat = (qmu - qmu_A) / (2 * self.sqrtqmuA_v)
                return teststat

            teststat = tensorlib.conditional(
                (sqrtqmu_v < self.sqrtqmuA_v), _true_case, _false_case
            )
        return teststat


class EmpiricalDistribution(object):
    """
    The empirical distribution of the test statistic.

    Unlike :py:class:`~pyhf.infer.calculators.AsymptoticTestStatDistribution` where the
    distribution for the test statistic is normally distributed, the
    :math:`p`-values etc are computed from the sampled distribution.
    """

    def __init__(self, samples, expected=None):
        """
        Empirical distribution.

        Args:
            samples (:obj:`tensor`): The test statistics sampled from the distribution.

        Returns:
            ~pyhf.infer.calculators.EmpiricalDistribution: The empirical distribution of the test statistic.

        """
        tensorlib, _ = get_backend()
        self.samples = tensorlib.ravel(samples)
        self.expected = expected

    def pvalue(self, value):
        """
        Compute the :math:`p`-value for a given value of the test statistic.

        Examples:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> mean = pyhf.tensorlib.astensor([5])
            >>> std = pyhf.tensorlib.astensor([1])
            >>> normal = pyhf.probability.Normal(mean, std)
            >>> samples = normal.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(samples)
            >>> dist.pvalue(7)
            0.02

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> init_pars = model.config.suggested_init()
            >>> par_bounds = model.config.suggested_bounds()
            >>> fixed_params = model.config.suggested_fixed()
            >>> mu_test = 1.0
            >>> pdf = model.make_pdf(pyhf.tensorlib.astensor(init_pars))
            >>> samples = pdf.sample((100,))
            >>> test_stat_dist = pyhf.infer.calculators.EmpiricalDistribution(
            ...     pyhf.tensorlib.astensor(
            ...         [pyhf.infer.test_statistics.qmu_tilde(mu_test, sample, model, init_pars, par_bounds, fixed_params) for sample in samples]
            ...     )
            ... )
            >>> test_stat_dist.pvalue(test_stat_dist.samples[9])
            0.3

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return (
            tensorlib.sum(tensorlib.where(self.samples >= value, 1, 0))
            / tensorlib.shape(self.samples)[0]
        )

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Examples:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> mean = pyhf.tensorlib.astensor([5])
            >>> std = pyhf.tensorlib.astensor([1])
            >>> normal = pyhf.probability.Normal(mean, std)
            >>> samples = normal.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(samples)
            >>> dist.expected_value(nsigma=1)
            6.15094381209505

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> init_pars = model.config.suggested_init()
            >>> par_bounds = model.config.suggested_bounds()
            >>> fixed_params = model.config.suggested_fixed()
            >>> mu_test = 1.0
            >>> pdf = model.make_pdf(pyhf.tensorlib.astensor(init_pars))
            >>> samples = pdf.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(
            ...     pyhf.tensorlib.astensor(
            ...         [
            ...             pyhf.infer.test_statistics.qmu_tilde(
            ...                 mu_test, sample, model, init_pars, par_bounds, fixed_params
            ...             )
            ...             for sample in samples
            ...         ]
            ...     )
            ... )
            >>> n_sigma = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> dist.expected_value(n_sigma)
            array([0.00000000e+00, 0.00000000e+00, 5.53671231e-04, 8.29987137e-01,
                   2.99592664e+00])

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """

        if self.expected is not None:
            # for tmu 'nsigma' are switched on purpose
            # -2 <-> 2
            # -1 <-> 1
            # bc for qmu nsigma of cdf is switched wrt nsigma of mu_hat
            # works only for nsigma in [-2, -1, 0, 1, 2]
            # no interpolation
            return(self.expected[2 - nsigma])

        tensorlib, _ = get_backend()
        import numpy as np

        # TODO: tensorlib.percentile function
        # c.f. https://github.com/scikit-hep/pyhf/pull/817
        return np.percentile(
            self.samples, tensorlib.normal_cdf(nsigma) * 100, interpolation="linear"
        )

# import logging
# log = logging.getLogger(__name__)


class ToyCalculator(object):
    """The Toy-based Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        qtilde=False,
        ntoys=2000,
        track_progress=True,
        bootstrap=False,
        return_fitted_pars=False,
        reuse_bkg_sample=False,
        fix_auxdata=True,
        return_dist=False,
        test_statistic='tmu',
        tilde=True,
    ):
        """
        Toy-based Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
            par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
            qtilde (:obj:`bool`): When ``True`` perform the calculation using the alternative test statistic, :math:`\\tilde{q}`, as defined in Equation (62) of :xref:`arXiv:1007.1727`.
            ntoys (:obj:`int`): Number of toys to use (how many times to sample the underlying distributions)
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)
            ttilde (:obj:`bool`): When ``True`` perform the calculation using the test statistic, :math:`\\tilde{t}`, as defined in Equation (40) of :xref:`arXiv:1007.1727`.
            bootstrap (:obj:`bool`): When ``True`` perform the calculation using the parametric bootstrap method.

        Returns:
            ~pyhf.infer.calculators.ToyCalculator: The calculator for toy-based quantities.

        """

        self.ntoys = ntoys
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()
        self.test_statistic = test_statistic
        self.tilde = tilde

        if qtilde:
            self.test_statistic = 'qmu'
            self.tilde = True

        if self.test_statistic == 'tmu':
            self.teststat_func = tmu_tilde if tilde else tmu
        elif self.test_statistic == 'qmu':
            self.teststat_func = qmu_tilde if tilde else qmu

        self.bootstrap = bootstrap
        self.reuse_bkg_sample = reuse_bkg_sample
        self.return_fitted_pars = return_fitted_pars or reuse_bkg_sample
        self.return_dist = return_dist
        self.fix_auxdata = fix_auxdata

        self.bkg_sample = None
        self.lhood_vals = None
        self.bkg_pars_reused = None
        self.bkg_pars = []
        self.bkg_pars_fixed_poi = []
        self.bkg_teststat_dist = []
        self.sig_bkg_pars = []
        self.sig_bkg_pars_fixed_poi = []
        self.sig_bkg_teststat_dist = []
        self.poi_list = []

        self.tqdm_options = dict(
            total=ntoys,
            leave=False,
            disable=not (
                track_progress if track_progress is not None else track_progress
            ),
            unit='toy',
        )

    def distributions(self, poi_test, b_dist=True, sb_dist=True, track_progress=None):
        """
        Probability Distributions of the test statistic value under the signal + background and background-only hypothesis.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> qmu_sig, qmu_bkg = toy_calculator.distributions(mu_test)
            >>> qmu_sig.pvalue(mu_test), qmu_bkg.pvalue(mu_test)
            (0.14, 0.76)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            Tuple (~pyhf.infer.calculators.EmpiricalDistribution): The distributions under the hypotheses.

        """
        tensorlib, _ = get_backend()
        sample_shape = (self.ntoys,)

        if self.bootstrap:
            fixed = [True] * self.pdf.config.npars
            fixed[self.pdf.config.poi_index] = False
            # params = fit(self.data, self.pdf, self.init_pars, self.par_bounds, self.fixed_params).tolist()
            # signal_pars = params.copy()
            # bkg_pars = params.copy()
            # or likelihood-ratio profile bootstrap:
            signal_pars = fixed_poi_fit(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds, self.fixed_params).tolist()
            bkg_pars = fixed_poi_fit(0.0, self.data, self.pdf, self.init_pars, self.par_bounds, self.fixed_params).tolist()
        else:
            fixed = self.fixed_params
            signal_pars = self.init_pars
            bkg_pars = self.init_pars

        signal_pars[self.pdf.config.poi_index] = poi_test
        signal_pdf = self.pdf.make_pdf(tensorlib.astensor(signal_pars))
        signal_sample = signal_pdf.sample(sample_shape)
        if self.fix_auxdata:
            signal_sample = tensorlib.concatenate([signal_sample[:, :-self.pdf.config.nauxdata], tensorlib.astensor([self.pdf.config.auxdata] * self.ntoys)], axis=1)

        bkg_pars[self.pdf.config.poi_index] = 0.0
        bkg_pdf = self.pdf.make_pdf(tensorlib.astensor(bkg_pars))
        self.bkg_sample = self.bkg_sample if (self.bkg_sample is not None and self.reuse_bkg_sample) else bkg_pdf.sample(sample_shape)
        if self.fix_auxdata:
            self.bkg_sample = tensorlib.concatenate([self.bkg_sample[:, :-self.pdf.config.nauxdata], tensorlib.astensor([self.pdf.config.auxdata] * self.ntoys)], axis=1)

        if self.return_fitted_pars or self.return_dist:
            self.poi_list.append(poi_test)

        if sb_dist:
            s_plus_b = self.sig_bkg_dist_calc(poi_test, signal_pars, fixed, signal_sample)
            if not b_dist:
                return s_plus_b

        if b_dist:
            b_only = self.bkg_dist_calc(poi_test, bkg_pars, fixed)
            if not sb_dist:
                return b_only

        return s_plus_b, b_only

    def sig_bkg_dist_calc(self, poi_test, signal_pars, fixed, signal_sample):
        tensorlib, _ = get_backend()
        signal_teststat = []
        signal_mubhathat = []
        signal_muhatbhat = []
        for sample in tqdm.tqdm(signal_sample, **self.tqdm_options, desc='Signal-like'):
            teststat_out = self.teststat_func(
                poi_test,
                sample,
                self.pdf,
                signal_pars,
                self.par_bounds,
                fixed,
                return_fitted_pars=self.return_fitted_pars,
                bootstrap=self.bootstrap
            )
            if self.return_fitted_pars:
                teststat, (mubhathat, muhatbhat), _ = teststat_out
                if not self.bootstrap:
                    signal_mubhathat.append(mubhathat)
                signal_muhatbhat.append(muhatbhat)
                signal_teststat.append(teststat)
            else:
                signal_muhatbhat.append(muhatbhat)
                signal_teststat.append(teststat_out)
        if self.return_fitted_pars:
            self.sig_bkg_pars.append(tensorlib.astensor(signal_muhatbhat))
            if not self.bootstrap:
                self.sig_bkg_pars_fixed_poi.append(tensorlib.astensor(signal_mubhathat))
        if self.return_dist:
            self.sig_bkg_teststat_dist.append(tensorlib.astensor(signal_teststat))

        # calculate expected
        expected = None
        if self.test_statistic == 'tmu':
            muhats = tensorlib.astensor(signal_muhatbhat)[:, self.pdf.config.poi_index]
            arg_percentiles = [self.arg_percentile(muhats, tensorlib.normal_cdf(nsigma) * 100) for nsigma in np.arange(-2, 3)]
            expected = [tensorlib.astensor(signal_teststat).flatten()[idx] for idx in arg_percentiles]
        return EmpiricalDistribution(tensorlib.astensor(signal_teststat), expected)

    def bkg_dist_calc(self, poi_test, bkg_pars, fixed):
        tensorlib, optimizer = get_backend()
        bkg_teststat = []
        bkg_mubhathat = []
        bkg_muhatbhat = []
        lhood_vals = []

        if self.lhood_vals is None:
            zipper = zip(self.bkg_sample)
        else:
            zipper = zip(self.bkg_sample, self.bkg_pars_reused, self.lhood_vals)

        return_fitted_pars = self.return_fitted_pars or self.reuse_bkg_sample
        first_run = self.reuse_bkg_sample and self.lhood_vals is None

        for sample in tqdm.tqdm(zipper, **self.tqdm_options, desc='Background-like'):
            if self.lhood_vals is None:
                muhatbhat = None
                unconstrained_fit_lhood_val = None
            else:
                sample, muhatbhat, unconstrained_fit_lhood_val = sample

            teststat_out = self.teststat_func(
                poi_test,
                sample,
                self.pdf,
                bkg_pars,
                self.par_bounds,
                fixed,
                return_fitted_pars=return_fitted_pars,
                muhatbhat=muhatbhat,
                unconstrained_fit_lhood_val=unconstrained_fit_lhood_val,
                bootstrap=self.bootstrap,
            )

            if return_fitted_pars:
                teststat, (mubhathat, muhatbhat), lhood_val = teststat_out
                bkg_teststat.append(teststat)
                if first_run:
                    lhood_vals.append(lhood_val)
                    bkg_muhatbhat.append(muhatbhat)
                if self.return_fitted_pars and not self.bootstrap:
                    bkg_mubhathat.append(mubhathat)
                if not self.reuse_bkg_sample:
                    bkg_muhatbhat.append(muhatbhat)
            else:
                bkg_muhatbhat.append(muhatbhat)
                bkg_teststat.append(teststat_out)

        if first_run:
            self.lhood_vals = tensorlib.astensor(lhood_vals)  # independent of poi_test if bkg_sample is reused
            self.bkg_pars_reused = tensorlib.astensor(bkg_muhatbhat)  # independent of poi_test if bkg_sample is reused
        if self.return_fitted_pars:
            if not self.reuse_bkg_sample:
                self.bkg_pars.append(tensorlib.astensor(bkg_muhatbhat))
            if not self.bootstrap:
                self.bkg_pars_fixed_poi.append(tensorlib.astensor(bkg_mubhathat))
        if self.return_dist:
            self.bkg_teststat_dist.append(tensorlib.astensor(bkg_teststat))

        # calculate expected
        expected = None
        if self.test_statistic == 'tmu':
            if self.reuse_bkg_sample:
                muhats = self.bkg_pars_reused[:, self.pdf.config.poi_index]
            else:
                muhats = tensorlib.astensor(bkg_muhatbhat)[:, self.pdf.config.poi_index]
            arg_percentiles = [self.arg_percentile(muhats, tensorlib.normal_cdf(nsigma) * 100) for nsigma in np.arange(-2, 3)]
            expected = [tensorlib.astensor(bkg_teststat).flatten()[idx] for idx in arg_percentiles]
        return EmpiricalDistribution(tensorlib.astensor(bkg_teststat), expected)

    def arg_percentile(self, a, q):
        idx = q / 100 * (len(a) - 1)
        idx = int(idx + 0.5)
        return np.argpartition(a, idx)[idx]

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> toy_calculator.teststatistic(mu_test)
            array(3.93824492)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Float: The value of the test statistic.

        """
        teststat = self.teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        return teststat


class AsymptoticTestStatDistributionCDF(object):
    r"""
    The distribution the test statistic in the asymptotic case.
    Using the CDF from the paper.
    """

    def __init__(self, data, model, mu, mu_prime, sigma=None, test_statistic='tmu', asimov_val=None, tilde=True, use_asimov=True):
        self.data = data
        self.model = model
        self.mu = mu
        self.mu_prime = mu_prime
        self.sigma = sigma
        self.asimov_val = asimov_val
        self.use_asimov = use_asimov
        self.tilde = tilde
        self.test_statistic = test_statistic
        if test_statistic == 'tmu':
            self.teststat_cdf = tmu_cdf
        elif test_statistic == 'qmu':
            self.teststat_cdf = qmu_cdf

    def cdf(self, value):
        return(self.teststat_cdf(value,
                                 self.mu,
                                 self.mu_prime,
                                 self.data,
                                 self.model,
                                 self.sigma,
                                 asimov_val=self.asimov_val,
                                 tilde=self.tilde,
                                 use_asimov=self.use_asimov))

    def ppf(self, quantile, method='Nelder-Mead', tol=1e-6, start_val=None):
        if quantile == 0.0:
            return(0.0)

        if self.tilde and (self.test_statistic == 'qmu') and (self.mu == 0):
            return(0.0)

        tensorlib, _ = get_backend()

        def function(value):
            return self.cdf(value)

        def diff(value, quantile):
            y = function(value)
            return (y - quantile)**2

        if not start_val:
            if function(1.0) >= quantile:
                test_values = linspace(0, 1, 1000)
            elif function(10.0) >= quantile:
                test_values = linspace(1, 10, 1000)
            elif function(50.0) >= quantile:
                test_values = linspace(10, 50, 10000)
            else:
                test_values = linspace(50, 200, 10000)
            arg = argmin(tensorlib.abs(function(test_values) - quantile))
            start_val = test_values[arg]

        if tensorlib.abs(function(start_val) - quantile) < 0.002:
            return(start_val)

        res = minimize(diff, start_val, args=(quantile), method=method, tol=tol)
        value = res.x[0]
        return(value)

    def pvalue(self, value):
        return(1 - self.cdf(value))

    def expected_value(self, nsigma):
        tensorlib, _ = get_backend()
        return self.ppf(tensorlib.normal_cdf(0 + nsigma))


class AsymptoticCalculatorCDF(object):
    """The Asymptotic Calculator using the CDF."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        qtilde=False,
        use_asimov=True,
        tilde=True,
        test_statistic='tmu',
    ):
        """
        Asymptotic Calculator for tmu tuilde and qmu tilde using the CDF.
        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()
        self.asimov_val = None
        self.tilde = tilde
        self.use_asimov = use_asimov
        self.test_statistic = test_statistic
        self.sigma_poi_null = None

        if qtilde:
            self.test_statistic = 'qmu'
            self.tilde = True

        if self.test_statistic == 'tmu':
            self.teststat_func = tmu_tilde if self.tilde else tmu
        elif self.test_statistic == 'qmu':
            self.teststat_func = qmu_tilde if self.tilde else qmu

    def distributions(self, poi_test, b_dist=True, sb_dist=True):
        if self.use_asimov and (self.tilde or b_dist):
            self.calc_asimov(poi_test)
            sigma_poi_test = None
        else:
            if sb_dist:
                sigma_poi_test = calc_sigma_fit(self.data, self.pdf, poi_test)
            if b_dist:
                self.sigma_poi_null = self.sigma_poi_null if self.sigma_poi_null else calc_sigma_fit(self.data, self.pdf, 0)

        if sb_dist:
            sb_dist = AsymptoticTestStatDistributionCDF(data=self.data,
                                                        model=self.pdf,
                                                        mu=poi_test,
                                                        mu_prime=poi_test,
                                                        sigma=sigma_poi_test,
                                                        test_statistic=self.test_statistic,
                                                        asimov_val=self.asimov_val,
                                                        tilde=self.tilde,
                                                        use_asimov=self.use_asimov,
                                                        )
            if not b_dist:
                return sb_dist
        if b_dist:
            b_dist = AsymptoticTestStatDistributionCDF(data=self.data,
                                                       model=self.pdf,
                                                       mu=poi_test,
                                                       mu_prime=0,
                                                       sigma=self.sigma_poi_null,
                                                       test_statistic=self.test_statistic,
                                                       asimov_val=self.asimov_val,
                                                       tilde=self.tilde,
                                                       use_asimov=self.use_asimov,
                                                       )
            if not sb_dist:
                return b_dist

        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        tensorlib, _ = get_backend()
        qmu_v = self.teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        return qmu_v

    def calc_asimov(self, poi_test):

        asimov_mu = 0.0
        asimov_data = generate_asimov_data(
            asimov_mu,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        self.asimov_val = self.teststat_func(
            poi_test,
            asimov_data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )


def tmu_cdf(tmu_val, mu, mu_prime, data, model, sigma=None, asimov_val=None, use_asimov=False, tilde=True):
    tensorlib, _ = get_backend()
    phi = tensorlib.normal_cdf
    sqrt = tensorlib.sqrt
    tmu_val = tensorlib.astensor(tmu_val)

    above = 0
    if use_asimov:
        below_thr = tmu_val <= asimov_val if tilde else 1.0
        if mu == mu_prime:
            common = phi(sqrt(tmu_val))
            below = (below_thr) * (phi(sqrt(tmu_val)) - 1)
            if tilde:
                if asimov_val == 0:
                    above = 0.0
                else:
                    above = (~below_thr) * (phi((tmu_val + asimov_val) / (2 * sqrt(asimov_val))) - 1)
        else:
            common = phi(sqrt(tmu_val) + sqrt(asimov_val))
            below = (below_thr) * (phi(sqrt(tmu_val) - sqrt(asimov_val)) - 1)
            if tilde:
                if asimov_val == 0:
                    above = 0.0
                else:
                    above = (~below_thr) * (phi((tmu_val - asimov_val) / (2 * sqrt(asimov_val))) - 1)
    else:
        below_thr = tmu_val <= (mu**2 / sigma**2) if tilde else 1.0
        sigma = sigma if sigma else calc_sigma_fit(data, model, mu_prime)
        common = phi(sqrt(tmu_val) + (mu - mu_prime) / sigma)
        below = (below_thr) * (phi(sqrt(tmu_val) - (mu - mu_prime) / sigma) - 1)
        if tilde:
            if mu == 0:
                above = 0.0
            else:
                above = (~below_thr) * (phi((tmu_val - (mu**2 - 2 * mu * mu_prime) / sigma**2) / (2 * mu / sigma)) - 1)

    total = common + below + above
    return(total)


def qmu_cdf(qmu_val, mu, mu_prime, data, model, sigma=None, asimov_val=None, use_asimov=False, tilde=True):
    tensorlib, _ = get_backend()
    sqrt = tensorlib.sqrt
    phi = tensorlib.normal_cdf
    qmu_val = tensorlib.astensor(qmu_val)

    above = 0
    if use_asimov:
        below_thr = qmu_val <= asimov_val if tilde else 1.0
        iszero = qmu_val == 0
        if mu == mu_prime:
            below = below_thr * (~iszero) * phi(sqrt(qmu_val))
            if tilde:
                if asimov_val == 0:
                    above = (~below_thr) * 1.0
                else:
                    above = (~below_thr) * phi((qmu_val + asimov_val) / (2 * sqrt(asimov_val)))
        else:
            below = below_thr * (~iszero) * phi(sqrt(qmu_val) - sqrt(asimov_val))
            if tilde:
                if asimov_val == 0:
                    above = (~below_thr) * 1.0
                else:
                    above = (~below_thr) * phi((qmu_val - asimov_val) / (2 * sqrt(asimov_val)))
    else:
        sigma = sigma if sigma else calc_sigma_fit(data, model, mu_prime)
        below_thr = qmu_val <= (mu**2 / sigma**2) if tilde else 1.0
        iszero = qmu_val == 0

        below = below_thr * (~iszero) * phi(sqrt(qmu_val) - (mu - mu_prime) / sigma)
        if tilde:
            if mu == 0:
                above = (~below_thr) * 1.0
            else:
                above = (~below_thr) * phi((qmu_val - (mu**2 - 2 * mu * mu_prime) / sigma**2) / (2 * mu / sigma))

    total = below + above
    return(total)


def calc_sigma_fit(data, model, mu_prime):
    backend = get_backend()

    init_pars = model.config.suggested_init()
    pars_pdf = model.config.suggested_init()
    pars_pdf[model.config.poi_index] = mu_prime
    unbounded_bounds = model.config.suggested_bounds()
    unbounded_bounds[model.config.poi_index] = (-25, 25)
    bounded_bounds = model.config.suggested_bounds()
    bounded_bounds[model.config.poi_index] = (0, 25)
    fixed_params = model.config.suggested_fixed()

    bestfit_nuisance_asimov = fixed_poi_fit(mu_prime, data, model, init_pars, bounded_bounds, fixed_params)
    asimov_data = model.expected_data(bestfit_nuisance_asimov)

    set_backend("numpy", minuit_optimizer(verbose=False))
    sigma = fit(asimov_data, model, pars_pdf, unbounded_bounds, fixed_params, return_uncertainties=True)[model.config.poi_index, 1]

    set_backend(*backend)
    return sigma
