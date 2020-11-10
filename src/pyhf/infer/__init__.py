"""Inference for Statistical Models."""

from . import utils
from .. import get_backend
# from numpy import histogram


def hypotest(
    poi_test,
    data,
    pdf,
    init_pars=None,
    par_bounds=None,
    fixed_params=None,
    qtilde=True,
    calctype="asymptotics",
    return_tail_probs=False,
    return_CLsb=False,
    return_expected=False,
    return_expected_set=False,
    return_dist=False,
    **kwargs,
):
    r"""
    Compute :math:`p`-values and test statistics for a single value of the parameter of interest.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional keyword arguments to be specified.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> mu_test = 1.0
        >>> CLs_obs, CLs_exp_band = pyhf.infer.hypotest(
        ...     mu_test, data, model, qtilde=True, return_expected_set=True
        ... )
        >>> CLs_obs
        array(0.05251497)
        >>> CLs_exp_band
        [array(0.00260626), array(0.01382005), array(0.06445321), array(0.23525644), array(0.57303621)]

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The data considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``
        init_pars (:obj:`tensor`): The initial parameter values to be used for minimization
        par_bounds (:obj:`tensor`): The parameter value bounds to be used for minimization
        fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
        qtilde (:obj:`bool`): When ``True`` perform the calculation using the alternative
         test statistic, :math:`\tilde{q}_{\mu}`, as defined under the Wald
         approximation in Equation (62) of :xref:`arXiv:1007.1727`.
        calctype (:obj:`str`): The calculator to create. Choose either 'asymptotics' (default) or 'toybased'.
        return_tail_probs (:obj:`bool`): Bool for returning :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
        return_expected (:obj:`bool`): Bool for returning :math:`\mathrm{CL}_{\mathrm{exp}}`
        return_expected_set (:obj:`bool`): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\mathrm{CL}_{\mathrm{exp}}` --- the "Brazil band"
        return_dist (:obj:`bool`): Return the sampled distributions (only if calculator is toybased)

    Returns:
        Tuple of Floats and lists of Floats:

            - :math:`\mathrm{CL}_{s}`: The modified :math:`p`-value compared to
              the given threshold :math:`\alpha`, typically taken to be :math:`0.05`,
              defined in :xref:`arXiv:1007.1727` as

            .. math::

                \mathrm{CL}_{s} = \frac{\mathrm{CL}_{s+b}}{\mathrm{CL}_{b}} = \frac{p_{s+b}}{1-p_{b}}

            to protect against excluding signal models in which there is little
            sensitivity. In the case that :math:`\mathrm{CL}_{s} \leq \alpha`
            the given signal model is excluded.

            - :math:`\left[\mathrm{CL}_{s+b}, \mathrm{CL}_{b}\right]`: The
              signal + background model hypothesis :math:`p`-value

            .. math::

                \mathrm{CL}_{s+b} = p_{s+b}
                = p\left(q \geq q_{\mathrm{obs}}\middle|s+b\right)
                = \int\limits_{q_{\mathrm{obs}}}^{\infty} f\left(q\,\middle|s+b\right)\,dq
                = 1 - F\left(q_{\mathrm{obs}}(\mu)\,\middle|\mu'\right)

            and 1 minus the background only model hypothesis :math:`p`-value

            .. math::

                \mathrm{CL}_{b} = 1- p_{b}
                = p\left(q \geq q_{\mathrm{obs}}\middle|b\right)
                = 1 - \int\limits_{-\infty}^{q_{\mathrm{obs}}} f\left(q\,\middle|b\right)\,dq
                = 1 - F\left(q_{\mathrm{obs}}(\mu)\,\middle|0\right)

            for signal strength :math:`\mu` and model hypothesis signal strength
            :math:`\mu'`, where the cumulative density functions
            :math:`F\left(q(\mu)\,\middle|\mu'\right)` are given by Equations (57)
            and (65) of :xref:`arXiv:1007.1727` for upper-limit-like test
            statistic :math:`q \in \{q_{\mu}, \tilde{q}_{\mu}\}`.
            Only returned when ``return_tail_probs`` is ``True``.

            .. note::

                The definitions of the :math:`\mathrm{CL}_{s+b}` and
                :math:`\mathrm{CL}_{b}` used are based on profile likelihood
                ratio test statistics.
                This procedure is common in the LHC-era, but differs from
                procedures used in the LEP and Tevatron eras, as briefly
                discussed in :math:`\S` 3.8 of :xref:`arXiv:1007.1727`.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}`: The expected :math:`\mathrm{CL}_{s}`
              value corresponding to the test statistic under the background
              only hypothesis :math:`\left(\mu=0\right)`.
              Only returned when ``return_expected`` is ``True``.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}` band: The set of expected
              :math:`\mathrm{CL}_{s}` values corresponding to the median
              significance of variations of the signal strength from the
              background only hypothesis :math:`\left(\mu=0\right)` at
              :math:`(-2,-1,0,1,2)\sigma`.
              That is, the :math:`p`-values that satisfy Equation (89) of
              :xref:`arXiv:1007.1727`

            .. math::

                \mathrm{band}_{N\sigma} = \mu' + \sigma\,\Phi^{-1}\left(1-\alpha\right) \pm N\sigma

            for :math:`\mu'=0` and :math:`N \in \left\{-2, -1, 0, 1, 2\right\}`.
            These values define the boundaries of an uncertainty band sometimes
            referred to as the "Brazil band".
            Only returned when ``return_expected_set`` is ``True``.

    """
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    fixed_params = fixed_params or pdf.config.suggested_fixed()

    calc = utils.create_calculator(
        calctype,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_params,
        qtilde=qtilde,
        **kwargs,
    )

    teststat = calc.teststatistic(poi_test)
    dists = calc.distributions(poi_test)
    if len(dists) == 5:
        sig_plus_bkg_distribution, b_only_distribution, muhats, bkg_sample, lhood_vals = dists
        return_fitted = True
        return_sample = True
    elif len(dists) == 3:
        sig_plus_bkg_distribution, b_only_distribution, muhats = dists
        return_fitted = True
        return_sample = False
    elif len(dists) == 2:
        sig_plus_bkg_distribution, b_only_distribution = dists
        return_fitted = False
        return_sample = False

    CLsb = sig_plus_bkg_distribution.pvalue(teststat)
    CLb = b_only_distribution.pvalue(teststat)
    CLs = CLsb / CLb

    tensorlib, _ = get_backend()
    # Ensure that all CL values are 0-d tensors
    CLsb, CLb, CLs = (
        tensorlib.astensor(CLsb),
        tensorlib.astensor(CLb),
        tensorlib.astensor(CLs),
    )

    _returns = [CLs]
    if return_tail_probs or return_CLsb:
        _returns.append([CLsb, CLb])
    if return_expected_set:
        CLs_exp = []
        CLsb_exp = []
        for n_sigma in [2, 1, 0, -1, -2]:

            expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

            CLs = sig_plus_bkg_distribution.pvalue(
                expected_bonly_teststat
            ) / b_only_distribution.pvalue(expected_bonly_teststat)
            CLs_exp.append(tensorlib.astensor(CLs))

            CLsb = sig_plus_bkg_distribution.pvalue(expected_bonly_teststat)
            CLsb_exp.append(tensorlib.astensor(CLsb))
        if return_expected:
            _returns.append(CLs_exp[2])
            if return_CLsb:
                _returns.append([CLsb_exp[2]])
        _returns.append(CLs_exp)
        if return_CLsb:
            _returns.append([CLsb_exp])
    elif return_expected:
        n_sigma = 0
        expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

        CLs = sig_plus_bkg_distribution.pvalue(
            expected_bonly_teststat
        ) / b_only_distribution.pvalue(expected_bonly_teststat)
        CLsb = sig_plus_bkg_distribution.pvalue(expected_bonly_teststat)
        _returns.append(tensorlib.astensor(CLs))
        if return_CLsb:
            _returns.append([CLsb])
    if calctype == 'toybased' and return_dist:
        # sig_max = tensorlib.max(sig_plus_bkg_distribution.samples)
        # bkg_max = tensorlib.max(b_only_distribution.samples)
        # hist_range = (0, bkg_max) if bkg_max > sig_max else (0, sig_max)
        # sig_hist = histogram(sig_plus_bkg_distribution.samples, bins=50, range=hist_range)
        # bkg_hist = histogram(b_only_distribution.samples, bins=50, range=hist_range)
        # _returns.append([sig_hist, bkg_hist])
        _returns.append([sig_plus_bkg_distribution.samples, b_only_distribution.samples])
    if return_fitted:
        _returns.append(muhats)
    if return_sample:
        return tuple(_returns), bkg_sample, lhood_vals
    else:
        # Enforce a consistent return type of the obseved CLs
        return tuple(_returns) if len(_returns) > 1 else _returns[0]


from . import intervals

# TODO: Can remove intervals when switch to flake8 (Issue #863)
__all__ = ["hypotest", "intervals"]
