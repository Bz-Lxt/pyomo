# -*- coding: utf-8 -*-
# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Optimality (KKT condition) verification for LP and convex QP models.

This module verifies the first-order optimality conditions of a linear or
convex quadratic optimization model using the primal variable values,
constraint duals, and variable reduced costs that have previously been loaded
onto the model following the Pyomo suffix conventions (for example, through
``SolverFactory.solve``, ``load_duals`` / ``load_rc``, or the
``pyomo.contrib.solver`` interfaces).

The check is strictly read-only: no component, value, suffix, or activation
state on the model is modified.

Sign convention
---------------
Following the documented Pyomo dual sign convention, a constraint is
represented in ``(lower, body, upper)`` form and the Lagrangian for a
minimization problem is written as

.. math::

    L(x) = f(x) - \\sum_i \\lambda_i c_i(x)
                 - \\sum_i \\nu_i g_i(x)
                 - \\sum_i \\delta_i h_i(x),

where ``c(x) = 0`` are equalities, ``g(x) = body - upper <= 0`` are the
upper-bound sides and ``h(x) = body - lower >= 0`` are the lower-bound sides.
The dual signs stored in the ``dual`` suffix and the reduced costs stored in
the ``rc`` suffix then satisfy:

==========================  =================  =================
component                   minimize           maximize
==========================  =================  =================
lower-bound side (``>=``)   dual / rc ``>= 0``  dual / rc ``<= 0``
upper-bound side (``<=``)   dual / rc ``<= 0``  dual / rc ``>= 0``
equality                    unrestricted sign   unrestricted sign
==========================  =================  =================

For a ranged (two-sided) constraint a single dual is reported: if a side is
active, the dual corresponds to that side (and carries its sign); if neither
side is active the dual must be zero.

For every free variable the reduced gradient implied by the objective and
constraint duals is

.. math::

    r_j = \\frac{\\partial f}{\\partial x_j}
          - \\sum_i dual_i \\frac{\\partial body_i}{\\partial x_j},

and the stationarity condition is ``r_j == rc_j``.

The four residual categories reported are: primal feasibility (constraint
bodies and variable bounds), dual feasibility (dual and reduced-cost signs),
stationarity (reduced gradient versus reduced cost), and complementarity
(multiplier times slack).
"""

import logging

from pyomo.common.collections import ComponentMap
from pyomo.core import Constraint, Objective, Suffix, Var, value
from pyomo.repn.standard_repn import generate_standard_repn

logger = logging.getLogger(__name__)

# Residual category identifiers (also used as report section labels)
PRIMAL = 'primal'
DUAL = 'dual'
STATIONARITY = 'stationarity'
COMPLEMENTARITY = 'complementarity'


class ConstraintResidual:
    """KKT residual information for a single :class:`ConstraintData`.

    Attributes
    ----------
    component: ConstraintData
        The constraint that was checked.
    name: str
        Fully qualified component name.
    equality: bool
        Whether the constraint is an equality constraint.
    ranged: bool
        Whether the constraint has both a finite lower and upper bound.
    body_value: float or None
        Numeric value of the constraint body (``None`` if it could not be
        evaluated).
    lower, upper: float or None
        Numeric constraint bounds.
    dual: float or None
        Dual value read from the ``dual`` suffix.
    primal_lb_residual, primal_ub_residual: float
        Lower- and upper-side primal feasibility violations (always
        non-negative).
    dual_sign_residual: float
        Violation of the dual sign condition (always non-negative).
    complementarity_lb_residual, complementarity_ub_residual: float
        Lower- and upper-side complementarity residuals ``|dual| * slack``
        (always non-negative).
    errors: list[str]
        Human-readable messages describing missing or un-evaluable data.
    """

    __slots__ = (
        'component',
        'name',
        'equality',
        'ranged',
        'body_value',
        'lower',
        'upper',
        'dual',
        'primal_lb_residual',
        'primal_ub_residual',
        'dual_sign_residual',
        'complementarity_lb_residual',
        'complementarity_ub_residual',
        'errors',
    )

    def __init__(self, component):
        self.component = component
        self.name = component.name
        self.equality = False
        self.ranged = False
        self.body_value = None
        self.lower = None
        self.upper = None
        self.dual = None
        self.primal_lb_residual = 0.0
        self.primal_ub_residual = 0.0
        self.dual_sign_residual = 0.0
        self.complementarity_lb_residual = 0.0
        self.complementarity_ub_residual = 0.0
        self.errors = []

    @property
    def primal_residual(self):
        return max(self.primal_lb_residual, self.primal_ub_residual)

    @property
    def complementarity_residual(self):
        return max(self.complementarity_lb_residual, self.complementarity_ub_residual)


class VariableResidual:
    """KKT residual information for a single :class:`VarData`.

    Attributes
    ----------
    component: VarData
        The variable that was checked.
    name: str
        Fully qualified component name.
    fixed: bool
        Whether the variable is fixed. Fixed variables are treated as data:
        only their primal bound feasibility is checked.
    value: float or None
        Loaded variable value.
    lower, upper: float or None
        Numeric variable bounds.
    reduced_cost: float or None
        Reduced cost read from the ``rc`` suffix (``None`` if the suffix or
        the individual value was unavailable).
    reduced_gradient: float or None
        Reduced gradient ``r_j`` implied by the objective gradient and the
        loaded constraint duals.
    primal_lb_residual, primal_ub_residual: float
        Lower- and upper-side primal feasibility violations (always
        non-negative).
    stationarity_residual: float
        ``|r_j - rc_j|`` when reduced costs are available; otherwise the
        residual of the implied bound-multiplier conditions.
    dual_sign_residual: float
        Violation of the reduced-cost sign condition (always non-negative).
    complementarity_lb_residual, complementarity_ub_residual: float
        Bound complementarity residuals ``|rc| * slack`` (always
        non-negative).
    interior_multiplier_residual: float
        Non-zero reduced cost for a variable strictly inside its bounds
        (always non-negative).
    errors: list[str]
        Human-readable messages describing missing or un-evaluable data.
    """

    __slots__ = (
        'component',
        'name',
        'fixed',
        'value',
        'lower',
        'upper',
        'reduced_cost',
        'reduced_gradient',
        'primal_lb_residual',
        'primal_ub_residual',
        'stationarity_residual',
        'dual_sign_residual',
        'complementarity_lb_residual',
        'complementarity_ub_residual',
        'interior_multiplier_residual',
        'errors',
    )

    def __init__(self, component):
        self.component = component
        self.name = component.name
        self.fixed = component.fixed
        self.value = component.value
        self.lower = None
        self.upper = None
        self.reduced_cost = None
        self.reduced_gradient = None
        self.primal_lb_residual = 0.0
        self.primal_ub_residual = 0.0
        self.stationarity_residual = 0.0
        self.dual_sign_residual = 0.0
        self.complementarity_lb_residual = 0.0
        self.complementarity_ub_residual = 0.0
        self.interior_multiplier_residual = 0.0
        self.errors = []

    @property
    def primal_residual(self):
        return max(self.primal_lb_residual, self.primal_ub_residual)

    @property
    def complementarity_residual(self):
        return max(
            self.complementarity_lb_residual,
            self.complementarity_ub_residual,
            self.interior_multiplier_residual,
        )


class OptimalityCheckResults:
    """Container for the output of :func:`check_optimality`.

    The per-component records are stored in deterministic model-traversal
    order on ``constraint_residuals`` and ``variable_residuals``. Aggregate
    worst-case residuals for each KKT category are provided as well as the
    overall optimality verdict.
    """

    __slots__ = (
        'model',
        'objective_sense',
        'tolerances',
        'convexity',
        'reduced_costs_available',
        'constraint_residuals',
        'variable_residuals',
        'errors',
        'max_primal_residual',
        'max_dual_residual',
        'max_stationarity_residual',
        'max_complementarity_residual',
        'is_optimal',
    )

    def __init__(self, model, objective_sense, tolerances, convexity):
        self.model = model
        self.objective_sense = objective_sense
        self.tolerances = tolerances
        self.convexity = convexity
        self.reduced_costs_available = False
        self.constraint_residuals = []
        self.variable_residuals = []
        self.errors = []
        self.max_primal_residual = 0.0
        self.max_dual_residual = 0.0
        self.max_stationarity_residual = 0.0
        self.max_complementarity_residual = 0.0
        self.is_optimal = False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def violations(self, categories=(PRIMAL, DUAL, STATIONARITY, COMPLEMENTARITY)):
        """Return the deterministic list of residual violations.

        Parameters
        ----------
        categories: tuple[str, ...]
            Restrict the output to the given residual categories.

        Returns
        -------
        list[tuple[str, ComponentData, float, str]]
            Tuples of ``(category, component, residual, detail)`` ordered by
            constraints first, then variables, in model-traversal order.
        """
        wanted = set(categories)
        ans = []
        tol = self.tolerances
        for rec in self.constraint_residuals:
            if PRIMAL in wanted and rec.primal_residual > tol[PRIMAL]:
                ans.append(
                    (
                        PRIMAL,
                        rec.component,
                        rec.primal_residual,
                        f'body {rec.body_value} violates bounds '
                        f'[{rec.lower}, {rec.upper}]',
                    )
                )
            if DUAL in wanted and rec.dual_sign_residual > tol[DUAL]:
                ans.append(
                    (
                        DUAL,
                        rec.component,
                        rec.dual_sign_residual,
                        f'dual {rec.dual} has the wrong sign for its active '
                        f'bound and objective sense',
                    )
                )
            if (
                COMPLEMENTARITY in wanted
                and rec.complementarity_residual > tol[COMPLEMENTARITY]
            ):
                ans.append(
                    (
                        COMPLEMENTARITY,
                        rec.component,
                        rec.complementarity_residual,
                        f'non-zero dual {rec.dual} with non-zero slack',
                    )
                )
        for rec in self.variable_residuals:
            if PRIMAL in wanted and rec.primal_residual > tol[PRIMAL]:
                ans.append(
                    (
                        PRIMAL,
                        rec.component,
                        rec.primal_residual,
                        f'value {rec.value} violates bounds '
                        f'[{rec.lower}, {rec.upper}]',
                    )
                )
            if STATIONARITY in wanted and rec.stationarity_residual > tol[STATIONARITY]:
                detail = (
                    f'reduced gradient {rec.reduced_gradient} differs from '
                    f'reduced cost {rec.reduced_cost}'
                    if self.reduced_costs_available and rec.reduced_cost is not None
                    else f'implied bound multipliers cannot realize reduced '
                    f'gradient {rec.reduced_gradient}'
                )
                ans.append(
                    (STATIONARITY, rec.component, rec.stationarity_residual, detail)
                )
            if DUAL in wanted and rec.dual_sign_residual > tol[DUAL]:
                ans.append(
                    (
                        DUAL,
                        rec.component,
                        rec.dual_sign_residual,
                        f'reduced cost {rec.reduced_cost} has the wrong sign '
                        f'for its active bound and objective sense',
                    )
                )
            if (
                COMPLEMENTARITY in wanted
                and rec.complementarity_residual > tol[COMPLEMENTARITY]
            ):
                ans.append(
                    (
                        COMPLEMENTARITY,
                        rec.component,
                        rec.complementarity_residual,
                        'non-zero reduced cost with non-zero bound slack',
                    )
                )
        return ans

    def write_report(self, ostream=None, violations_only=True):
        """Write a human-readable optimality report.

        Parameters
        ----------
        ostream: io.TextIOBase or None
            Stream to write to; defaults to ``sys.stdout``.
        violations_only: bool
            If ``True`` (the default), only components with a residual
            exceeding its tolerance and data errors are listed. The summary
            block is always written.
        """
        if ostream is None:
            import sys

            ostream = sys.stdout
        sense_name = 'minimize' if self.objective_sense == 1 else 'maximize'
        tol = self.tolerances
        lines = [
            'Optimality check (' + sense_name + ')',
            f'  tolerances: primal={tol[PRIMAL]:g}, dual={tol[DUAL]:g}, '
            f'stationarity={tol[STATIONARITY]:g}, '
            f'complementarity={tol[COMPLEMENTARITY]:g}',
            f'  convexity: {self.convexity}',
            f'  max primal residual:          {self.max_primal_residual:.3e}',
            f'  max dual residual:            {self.max_dual_residual:.3e}',
            f'  max stationarity residual:    {self.max_stationarity_residual:.3e}',
            f'  max complementarity residual: {self.max_complementarity_residual:.3e}',
            f'  verdict: {"OPTIMAL" if self.is_optimal else "NOT OPTIMAL"}',
        ]
        if self.errors:
            lines.append('  data errors:')
            lines.extend(f'    - {msg}' for msg in self.errors)
        lines.append('  constraint residuals:')
        for rec in self.constraint_residuals:
            if violations_only and not (
                rec.errors
                or rec.primal_residual > tol[PRIMAL]
                or rec.dual_sign_residual > tol[DUAL]
                or rec.complementarity_residual > tol[COMPLEMENTARITY]
            ):
                continue
            lines.append(
                f'    {rec.name}: body={rec.body_value}, bounds=[{rec.lower}, '
                f'{rec.upper}], dual={rec.dual}, primal(lb='
                f'{rec.primal_lb_residual:.3e}, ub={rec.primal_ub_residual:.3e}), '
                f'dual_sign={rec.dual_sign_residual:.3e}, '
                f'complementarity(lb={rec.complementarity_lb_residual:.3e}, '
                f'ub={rec.complementarity_ub_residual:.3e})'
            )
            lines.extend(f'      error: {msg}' for msg in rec.errors)
        lines.append('  variable residuals:')
        for rec in self.variable_residuals:
            if violations_only and not (
                rec.errors
                or rec.primal_residual > tol[PRIMAL]
                or rec.stationarity_residual > tol[STATIONARITY]
                or rec.dual_sign_residual > tol[DUAL]
                or rec.complementarity_residual > tol[COMPLEMENTARITY]
            ):
                continue
            lines.append(
                f'    {rec.name}: value={rec.value}, bounds=[{rec.lower}, '
                f'{rec.upper}], rc={rec.reduced_cost}, r={rec.reduced_gradient}, '
                f'primal(lb={rec.primal_lb_residual:.3e}, '
                f'ub={rec.primal_ub_residual:.3e}), '
                f'stationarity={rec.stationarity_residual:.3e}, '
                f'dual_sign={rec.dual_sign_residual:.3e}, '
                f'complementarity(lb={rec.complementarity_lb_residual:.3e}, '
                f'ub={rec.complementarity_ub_residual:.3e}, '
                f'interior={rec.interior_multiplier_residual:.3e})'
            )
            lines.extend(f'      error: {msg}' for msg in rec.errors)
        ostream.write('\n'.join(lines) + '\n')

    def __repr__(self):
        return (
            f'<OptimalityCheckResults: optimal={self.is_optimal}, '
            f'max residuals primal={self.max_primal_residual:.3e}, '
            f'dual={self.max_dual_residual:.3e}, '
            f'stationarity={self.max_stationarity_residual:.3e}, '
            f'complementarity={self.max_complementarity_residual:.3e}>'
        )


def check_optimality(
    model,
    tol=1e-6,
    *,
    primal_tol=None,
    dual_tol=None,
    stationarity_tol=None,
    complementarity_tol=None,
    check_convexity=True,
):
    """Check the KKT optimality conditions of an LP or convex QP model.

    The function uses the variable values, ``dual`` suffix and (optionally)
    ``rc`` suffix currently loaded on the model and does not modify the model
    in any way.

    Parameters
    ----------
    model: Block
        The concrete Pyomo model (or sub-block) to verify.
    tol: float
        Default absolute tolerance applied to every residual category.
    primal_tol, dual_tol, stationarity_tol, complementarity_tol: float or None
        Optional category-specific overrides of ``tol``. The primal
        tolerance is also used to decide whether a constraint side or
        variable bound is active.
    check_convexity: bool
        If ``True`` (the default), the Hessian of a quadratic objective is
        tested for positive semi-definiteness (minimization) or negative
        semi-definiteness (maximization) using NumPy, when NumPy is
        available. A non-convex objective fails the optimality verdict.
        Linear objectives and models without NumPy skip the eigenvalue test.

    Returns
    -------
    OptimalityCheckResults
        Per-component residual records, aggregate residuals and the overall
        optimality verdict.

    Raises
    ------
    ValueError
        If the model does not have exactly one active objective, if the
        ``dual`` suffix is missing, or if an objective or constraint body is
        not at most quadratic (the scope of this checker).
    """
    tolerances = {
        PRIMAL: tol if primal_tol is None else primal_tol,
        DUAL: tol if dual_tol is None else dual_tol,
        STATIONARITY: (tol if stationarity_tol is None else stationarity_tol),
        COMPLEMENTARITY: (tol if complementarity_tol is None else complementarity_tol),
    }
    primal_tol = tolerances[PRIMAL]

    active_objectives = list(
        model.component_data_objects(Objective, active=True, descend_into=True)
    )
    if len(active_objectives) != 1:
        raise ValueError(
            'check_optimality expects exactly one active objective; found '
            f'{len(active_objectives)} on model {model.name!r}.'
        )
    objective = active_objectives[0]
    sense = int(objective.sense)  # minimize = 1, maximize = -1

    dual_suffix = _find_suffix(model, 'dual')
    if dual_suffix is None:
        raise ValueError(
            "The model has no import 'dual' suffix. Solve with duals "
            'enabled (e.g. model.dual = Suffix(direction=Suffix.IMPORT)) '
            'and load a solution before calling check_optimality().'
        )
    rc_suffix = _find_suffix(model, 'rc')

    # Objective gradient and LP/QP scope validation
    obj_repn = _quadratic_repn(objective.expr, f'objective {objective.name}')
    reduced_gradient = _numeric_gradient(obj_repn, f'objective {objective.name}')

    convexity = _check_objective_convexity(obj_repn, sense, check_convexity)

    # Up-front scope validation and repn extraction for every active
    # constraint, so out-of-scope models fail regardless of suffix data.
    constraint_repns = ComponentMap()
    for con in model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True
    ):
        constraint_repns[con] = _quadratic_repn(
            con.body, f'constraint {con.name}', allow_quadratic=False
        )

    results = OptimalityCheckResults(model, sense, tolerances, convexity)
    results.reduced_costs_available = rc_suffix is not None

    # ------------------------------------------------------------------
    # Constraints: primal feasibility, dual signs, complementarity, and
    # accumulation of the reduced gradient r = grad f - sum dual * grad body
    # ------------------------------------------------------------------
    for con in model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True
    ):
        rec = ConstraintResidual(con)
        results.constraint_residuals.append(rec)
        rec.equality = con.equality
        rec.ranged = con.has_lb() and con.has_ub() and not con.equality

        body_value = value(con.body, exception=False)
        rec.body_value = body_value
        if body_value is None:
            rec.errors.append(
                'constraint body could not be evaluated (missing variable '
                'value or evaluation error)'
            )
            results.errors.append(f'{con.name}: unevaluable constraint body')
            continue

        if con.has_lb():
            rec.lower = value(con.lower, exception=False)
        if con.has_ub():
            rec.upper = value(con.upper, exception=False)
        if con.has_lb() and rec.lower is None:
            rec.errors.append('lower bound could not be evaluated')
            results.errors.append(f'{con.name}: unevaluable lower bound')
        if con.has_ub() and rec.upper is None:
            rec.errors.append('upper bound could not be evaluated')
            results.errors.append(f'{con.name}: unevaluable upper bound')

        # Primal feasibility
        if rec.equality:
            residual = abs(body_value - rec.upper)
            rec.primal_lb_residual = residual
            rec.primal_ub_residual = residual
        else:
            if rec.lower is not None:
                rec.primal_lb_residual = max(0.0, rec.lower - body_value)
            if rec.upper is not None:
                rec.primal_ub_residual = max(0.0, body_value - rec.upper)

        # Dual lookup
        if con in dual_suffix:
            rec.dual = dual_suffix[con]
        if rec.dual is None:
            rec.errors.append('missing dual value')
            results.errors.append(f'{con.name}: missing dual value')
            continue

        # Reduced-gradient contribution (read-only gradient extraction)
        try:
            con_grad = _numeric_gradient(
                constraint_repns[con], f'constraint {con.name}'
            )
        except _UnevaluableGradient as err:
            rec.errors.append(str(err))
            results.errors.append(f'{con.name}: {err}')
            continue
        for var, partial in con_grad.items():
            if partial is None:
                rec.errors.append('gradient depends on a variable without a value')
                results.errors.append(
                    f'{con.name}: gradient requires unassigned variable value'
                )
                continue
            reduced_gradient[var] = reduced_gradient.get(var, 0.0) - rec.dual * partial

        if rec.equality:
            # Equality multipliers have unrestricted sign and no
            # complementarity condition.
            continue

        # Non-negative slacks on the two bound sides
        slack_lb = body_value - rec.lower if rec.lower is not None else None
        slack_ub = rec.upper - body_value if rec.upper is not None else None
        lower_active = slack_lb is not None and slack_lb <= primal_tol
        upper_active = slack_ub is not None and slack_ub <= primal_tol
        signed_dual = sense * rec.dual

        # Dual feasibility: the active side determines the required sign.
        if lower_active and not upper_active:
            rec.dual_sign_residual = max(0.0, -signed_dual)
        elif upper_active and not lower_active:
            rec.dual_sign_residual = max(0.0, signed_dual)
        # both active (degenerate equality-like range) or neither active:
        # sign is unconstrained; a non-zero dual on an inactive constraint
        # is captured by the complementarity residual below.

        # Complementarity: the reported dual belongs to the side whose sign
        # it carries; the other side has an implicit zero multiplier.
        if slack_lb is not None and signed_dual > 0:
            rec.complementarity_lb_residual = abs(rec.dual) * max(slack_lb, 0.0)
        if slack_ub is not None and signed_dual < 0:
            rec.complementarity_ub_residual = abs(rec.dual) * max(slack_ub, 0.0)

    # ------------------------------------------------------------------
    # Variables: bound feasibility, reduced-cost signs, stationarity and
    # bound complementarity
    # ------------------------------------------------------------------
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        rec = VariableResidual(var)
        results.variable_residuals.append(rec)

        if var.has_lb():
            rec.lower = value(var.lower, exception=False)
        if var.has_ub():
            rec.upper = value(var.upper, exception=False)

        if rec.value is None:
            rec.errors.append('variable has no assigned value')
            results.errors.append(f'{var.name}: missing variable value')
            continue

        # Primal bound feasibility
        if rec.lower is not None:
            rec.primal_lb_residual = max(0.0, rec.lower - rec.value)
        if rec.upper is not None:
            rec.primal_ub_residual = max(0.0, rec.value - rec.upper)

        if rec.fixed:
            # Fixed variables are parameters of the KKT system: they carry
            # no stationarity or reduced-cost condition.
            continue

        r_j = reduced_gradient.get(var, 0.0)
        rec.reduced_gradient = r_j

        if rc_suffix is not None and var in rc_suffix:
            rec.reduced_cost = rc_suffix[var]

        slack_lb = rec.value - rec.lower if rec.lower is not None else None
        slack_ub = rec.upper - rec.value if rec.upper is not None else None
        lower_active = slack_lb is not None and slack_lb <= primal_tol
        upper_active = slack_ub is not None and slack_ub <= primal_tol
        interior = not lower_active and not upper_active

        if rec.reduced_cost is None:
            # No reduced cost available: fold the implied bound-multiplier
            # conditions into the stationarity residual.
            if interior:
                rec.stationarity_residual = abs(r_j)
            elif lower_active and not upper_active:
                rec.stationarity_residual = max(0.0, -sense * r_j)
            elif upper_active and not lower_active:
                rec.stationarity_residual = max(0.0, sense * r_j)
            continue

        # Stationarity: reduced gradient must equal the reduced cost.
        rec.stationarity_residual = abs(r_j - rec.reduced_cost)
        signed_rc = sense * rec.reduced_cost

        # Dual feasibility of the bound multipliers.
        if interior:
            # Both bound multipliers must vanish at an interior point.
            rec.interior_multiplier_residual = abs(rec.reduced_cost)
        elif lower_active and not upper_active:
            rec.dual_sign_residual = max(0.0, -signed_rc)
        elif upper_active and not lower_active:
            rec.dual_sign_residual = max(0.0, signed_rc)

        # Bound complementarity.
        if slack_lb is not None and signed_rc > 0:
            rec.complementarity_lb_residual = abs(rec.reduced_cost) * max(slack_lb, 0.0)
        if slack_ub is not None and signed_rc < 0:
            rec.complementarity_ub_residual = abs(rec.reduced_cost) * max(slack_ub, 0.0)

    _finalize_results(results)
    return results


def log_optimality_check(model, tol=1e-6, logger=logger, **kwds):
    """Run :func:`check_optimality` and log the residual violations.

    Messages are emitted at the ``INFO`` level. Keyword arguments are passed
    through to :func:`check_optimality`.

    Parameters
    ----------
    model: Block
        The Pyomo model to verify.
    tol: float
        Absolute tolerance.
    logger: logging.Logger
        Logger to output to; defaults to the ``pyomo.util.check_optimality``
        logger.
    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning(
            'log_optimality_check() called with a logger whose effective '
            'level is higher than logging.INFO: no output will be logged '
            'regardless of optimality status'
        )
    results = check_optimality(model, tol=tol, **kwds)
    for category, component, residual, detail in results.violations():
        logger.info(
            '%s %s: %.3g (%s)', category.upper(), component.name, residual, detail
        )
    for msg in results.errors:
        logger.info('ERROR %s', msg)
    logger.info('verdict: %s', 'OPTIMAL' if results.is_optimal else 'NOT OPTIMAL')
    return results


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
class _UnevaluableGradient(Exception):
    pass


def _find_suffix(model, local_name):
    for suffix in model.component_objects(ctype=Suffix, active=True, descend_into=True):
        if suffix.local_name == local_name:
            return suffix
    return None


def _quadratic_repn(expr, component_label, allow_quadratic=True):
    """Return a standard repn, rejecting out-of-scope expressions.

    Objective bodies may be linear or convex quadratic (degree <= 2).
    Constraint bodies must be linear: a convex quadratic program has linear
    constraints, and this checker does not verify convexity of quadratic
    constraint sets.
    """
    repn = generate_standard_repn(expr, quadratic=True)
    if repn.nonlinear_expr is not None:
        raise ValueError(
            f'Optimality checking is only supported for linear and convex '
            f'quadratic models; {component_label} contains a non-quadratic '
            f'expression: {repn.nonlinear_expr}'
        )
    if not allow_quadratic and repn.quadratic_coefs:
        raise ValueError(
            f'Optimality checking is only supported for models with linear '
            f'constraints; {component_label} contains a quadratic '
            f'expression: {expr}'
        )
    return repn


def _numeric_gradient(repn, component_label):
    """Evaluate the gradient of a linear/quadratic repn at the current values.

    Returns a :class:`ComponentMap` mapping each free variable to its numeric
    partial derivative. Fixed variables are folded into the repn constants by
    ``generate_standard_repn`` and therefore do not appear.
    """
    gradient = ComponentMap()
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        coef_value = value(coef, exception=False)
        if coef_value is None:
            raise _UnevaluableGradient(
                f'linear coefficient of {var.name} in {component_label} '
                'could not be evaluated'
            )
        gradient[var] = gradient.get(var, 0.0) + float(coef_value)
    for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
        coef_value = value(coef, exception=False)
        if coef_value is None:
            raise _UnevaluableGradient(
                f'quadratic coefficient in {component_label} could not be ' 'evaluated'
            )
        if v1 is v2:
            if v1.value is None:
                gradient[v1] = None
            else:
                gradient[v1] = gradient.get(v1, 0.0) + (
                    2.0 * float(coef_value) * float(v1.value)
                )
        else:
            if v2.value is None:
                gradient[v1] = None
            else:
                gradient[v1] = gradient.get(v1, 0.0) + (
                    float(coef_value) * float(v2.value)
                )
            if v1.value is None:
                gradient[v2] = None
            else:
                gradient[v2] = gradient.get(v2, 0.0) + (
                    float(coef_value) * float(v1.value)
                )
    return gradient


def _check_objective_convexity(repn, sense, check_convexity):
    if not check_convexity or not repn.quadratic_coefs:
        return 'convex' if not repn.quadratic_coefs else 'not_checked'
    try:
        import numpy as np
    except ImportError:
        logger.info(
            'NumPy is not available; objective convexity eigenvalue check '
            'was skipped.'
        )
        return 'not_checked'

    ordered_vars = []
    index = ComponentMap()
    for v1, v2 in repn.quadratic_vars:
        for v in (v1, v2):
            if v not in index:
                index[v] = len(ordered_vars)
                ordered_vars.append(v)
    hessian = np.zeros((len(ordered_vars), len(ordered_vars)))
    for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
        i, j = index[v1], index[v2]
        hessian[i, j] += float(coef)
        if i != j:
            hessian[j, i] += float(coef)
        else:
            # repn stores the coefficient of v**2 once; the Hessian entry
            # is twice that coefficient.
            hessian[i, i] += float(coef)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if sense == 1:
        return 'convex' if eigenvalues.min() >= -1e-9 else 'non_convex'
    return 'convex' if eigenvalues.max() <= 1e-9 else 'non_convex'


def _finalize_results(results):
    max_primal = 0.0
    max_dual = 0.0
    max_stationarity = 0.0
    max_complementarity = 0.0

    for rec in results.constraint_residuals:
        max_primal = max(max_primal, rec.primal_residual)
        max_dual = max(max_dual, rec.dual_sign_residual)
        max_complementarity = max(max_complementarity, rec.complementarity_residual)
    for rec in results.variable_residuals:
        max_primal = max(max_primal, rec.primal_residual)
        max_dual = max(max_dual, rec.dual_sign_residual)
        max_stationarity = max(max_stationarity, rec.stationarity_residual)
        max_complementarity = max(max_complementarity, rec.complementarity_residual)

    results.max_primal_residual = max_primal
    results.max_dual_residual = max_dual
    results.max_stationarity_residual = max_stationarity
    results.max_complementarity_residual = max_complementarity

    tol = results.tolerances
    results.is_optimal = (
        not results.errors
        and results.convexity != 'non_convex'
        and max_primal <= tol[PRIMAL]
        and max_dual <= tol[DUAL]
        and max_stationarity <= tol[STATIONARITY]
        and max_complementarity <= tol[COMPLEMENTARITY]
    )
