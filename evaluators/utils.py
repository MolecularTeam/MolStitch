# from pymoo.factory import get_performance_indicator
import numpy as np
import math
from copy import deepcopy
from pymoo.indicators.hv import HV
from pygmo import hypervolume
import subprocess


def get_all_metrics(solutions, eval_metrics, **kwargs):
    """
    This method assumes the solutions are already filtered to the pareto front
    """

    metrics = {}
    if "hypervolume" in eval_metrics and "hv_ref" in kwargs.keys():
        hv_indicator = HV(ref_point=kwargs["hv_ref"])
        # `-` cause pymoo assumes minimization
        metrics["hypervolume"] = hv_indicator.do(-solutions)

    if "r2" in eval_metrics and "r2_prefs" in kwargs.keys() and "num_obj" in kwargs.keys():
        metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))

    # if "hsri" in eval_metrics and "num_obj" in kwargs.keys():
    #     # class assumes minimization so transformer to negative problem
    #     hsr_class = HSR_Calculator(lower_bound=-np.ones(kwargs["num_obj"]) - 0.1,
    #                                upper_bound=np.zeros(kwargs["num_obj"]) + 0.1)
    #     # try except cause hsri can run into divide by zero errors
    #     try:
    #         metrics["hsri"], x = hsr_class.calculate_hsr(-solutions)
    #     except:
    #         metrics["hsri"] = 0.
    #     try:
    #         metrics["hsri"] = metrics["hsri"] if type(metrics["hsri"]) is float else metrics["hsri"][0]
    #     except:
    #         metrics["hsri"] = 0.
    return metrics


def get_all_metrics_pygmo(solutions, eval_metrics, **kwargs):
    """
    This method assumes the solutions are already filtered to the pareto front
    """

    metrics = {}
    if "hypervolume" in eval_metrics and "hv_ref" in kwargs.keys():
        hv = hypervolume(-solutions)
        ref_point = [0] * solutions.shape[-1]
        metrics["hypervolume"] = hv.compute(ref_point)

    if "r2" in eval_metrics and "r2_prefs" in kwargs.keys() and "num_obj" in kwargs.keys():
        metrics["r2"] = r2_indicator_set(kwargs["r2_prefs"], solutions, np.ones(kwargs["num_obj"]))

    return metrics


def r2_indicator_set(reference_points, solutions, utopian_point):
    """Computer R2 indicator value of a set of solutions (*solutions*) given a set of
    reference points (*reference_points) and a utopian_point (*utopian_point).
        :param reference_points: An array of reference points from a uniform distribution.
        :param solutions: the multi-objective solutions (fitness values).
        :param utopian_point: utopian point that represents best possible solution
        :returns: r2 value (float).
        """
    min_list = []
    for v in reference_points:
        max_list = []
        for a in solutions:
            max_list.append(np.max(v * np.abs(utopian_point - a)))
        min_list.append(np.min(max_list))

    v_norm = np.linalg.norm(reference_points)
    r2 = np.sum(min_list) / v_norm

    return r2

