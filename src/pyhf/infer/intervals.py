"""Interval estimation"""
from . import hypotest
from . import utils
from .. import get_backend
import numpy as np
from scipy.interpolate import interp1d


def _interp(x, xp, fp):
    tb, _ = get_backend()

    if np.isnan(fp).all():
        print('all nan in scan')
        return None

    points = np.asarray([fp, np.nan_to_num(xp)]).T

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    method = 'quadratic'  # 'cubic' 'slinear' 'quadratic'
    alpha = np.linspace(0, 1, 10000)
    interpolator = interp1d(distance, points, kind=method, axis=0)
    fp_dense, xp_dense = interpolator(alpha).T

    greater_than_x = xp_dense > x
    if greater_than_x.all():
        return None
    if (~greater_than_x).all():
        return 0.0

    lower_bound = fp_dense[np.where(greater_than_x)[0][-1]]
    upper_bound = fp_dense[np.where(greater_than_x)[0][0]]

    # return tb.astensor(np.interp(x, xp, fp))
    return upper_bound


def upperlimit(data, model, scan, level=0.05, return_results=False, return_calculators=False, return_CLsb=False, results=None, sb_calc_kw=None, b_calc_kw=None, calctype='asymptotics', **kwargs):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using a fixed scan through POI-space.

    Example:
        >>> import numpy as np
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> scan = np.linspace(0, 5, 21)
        >>> obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upperlimit(
        ...     data, model, scan, return_results=True
        ... )
        >>> obs_limit
        array(1.01764089)
        >>> exp_limits
        [array(0.59576921), array(0.76169166), array(1.08504773), array(1.50170482), array(2.06654952)]

    Args:
        data (:obj:`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        scan (:obj:`iterable`): Iterable of POI values.
        level (:obj:`float`): The threshold value to evaluate the interpolated results at.
        return_results (:obj:`bool`): Whether to return the per-point results.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limits on the POI.
            - Tuple of Tensors: The given ``scan`` along with the
              :class:`~pyhf.infer.hypotest` results at each test POI.
              Only returned when ``return_results`` is ``True``.
    """
    tb, _ = get_backend()

    if not results:
        if sb_calc_kw or b_calc_kw:
            assert(b_calc_kw is not None)
            assert(sb_calc_kw is not None)

            sb_kw = sb_calc_kw.copy()
            b_kw = b_calc_kw.copy()
            sb_kw['par_bounds'] = sb_kw.get('par_bounds') or kwargs.get('par_bounds')
            b_kw['par_bounds'] = b_kw.get('par_bounds') or kwargs.get('par_bounds')

            sb_calc = utils.create_calculator(
                data=data,
                pdf=model,
                **sb_kw,
            )
            b_calc = utils.create_calculator(
                data=data,
                pdf=model,
                **b_kw,
            )
            calc = None
        else:
            calc = utils.create_calculator(
                calctype,
                data,
                model,
                **kwargs,
            )
            sb_calc = None
            b_calc = None

        results = []

        for mu in scan:
            results.append(
                hypotest(mu, data, model, return_expected_set=True, return_CLsb=True, sb_dist_calc=sb_calc, b_dist_calc=b_calc, calc=calc,)
            )
        calculators = None
        if calc:
            if calctype == 'toybased':
                if calc.return_fitted_pars or calc.return_dist:
                    calculators = calc_to_dict(calc)
        else:
            sb_calc_return = None
            b_calc_return = None
            if sb_calc_kw['calctype'] == 'toybased':
                if sb_calc.return_fitted_pars or sb_calc.return_dist:
                    sb_calc_return = calc_to_dict(sb_calc)
            if b_calc_kw['calctype'] == 'toybased':
                if b_calc.return_fitted_pars or b_calc.return_dist:
                    b_calc_return = calc_to_dict(b_calc)
            calculators = [sb_calc_return, b_calc_return]
    else:
        if type(results) == tuple:
            (results, calculators) = results
        else:
            calculators = None

    obs = tb.astensor([[r[0]] for r in results])
    exp = tb.astensor([[r[2][idx] for idx in range(5)] for r in results])
    result_arrary = tb.concatenate([obs, exp], axis=1).T

    # observed limit and the (0, +-1, +-2)sigma expected limits
    limits = [_interp(level, result_arrary[idx].tolist()[::-1], scan.tolist()[::-1]) for idx in range(6)]
    obs_limit, exp_limits = limits[0], limits[1:]

    if return_CLsb:
        # same for CLsb
        obs = tb.astensor([[r[1][0]] for r in results])
        exp = tb.astensor([[r[3][0][idx] for idx in range(5)] for r in results])
        result_arrary = tb.concatenate([obs, exp], axis=1).T
        limits = [_interp(level, result_arrary[idx].tolist()[::-1], scan.tolist()[::-1]) for idx in range(6)]
        obs_limit_CLsb, exp_limits_CLsb = limits[0], limits[1:]

        obs_limit = (obs_limit, obs_limit_CLsb)
        exp_limits = (exp_limits, exp_limits_CLsb)

        if return_results or return_calculators:
            if return_calculators:
                results = (results, calculators)
            return obs_limit, exp_limits, (scan, results)
        return obs_limit, exp_limits
    else:
        if return_results or return_calculators:
            if return_calculators:
                results = (results, calculators)
            return obs_limit, exp_limits, (scan, results)
        return obs_limit, exp_limits


def calc_to_dict(calc):
    calculator_dict = {}
    calculator_dict['bkg_pars_reused'] = calc.bkg_pars_reused
    calculator_dict['bkg_pars'] = calc.bkg_pars
    calculator_dict['bkg_pars_fixed_poi'] = calc.bkg_pars_fixed_poi
    calculator_dict['bkg_teststat_dist'] = calc.bkg_teststat_dist
    calculator_dict['sig_bkg_pars'] = calc.sig_bkg_pars
    calculator_dict['sig_bkg_pars_fixed_poi'] = calc.sig_bkg_pars_fixed_poi
    calculator_dict['sig_bkg_teststat_dist'] = calc.sig_bkg_teststat_dist
    calculator_dict['poi_list'] = calc.poi_list
    calculator_dict['ntoys'] = calc.ntoys
    calculator_dict['init_pars'] = calc.init_pars
    calculator_dict['par_bounds'] = calc.par_bounds
    calculator_dict['fixed_params'] = calc.fixed_params
    calculator_dict['test_statistic'] = calc.test_statistic
    calculator_dict['tilde'] = calc.tilde
    calculator_dict['bootstra'] = calc.bootstrap
    calculator_dict['reuse_bkg_sample'] = calc.reuse_bkg_sample
    calculator_dict['return_fitted_pars'] = calc.return_fitted_pars
    calculator_dict['return_dist'] = calc.return_dist
    calculator_dict['fix_auxdata'] = calc.fix_auxdata
    return calculator_dict
