# -*- coding: utf-8 -*-
# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Tests for KKT optimality verification of LP and convex QP models."""

import logging
from io import StringIO

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    Objective,
    RangeSet,
    Suffix,
    Var,
    maximize,
    minimize,
    value,
)
from pyomo.util.check_optimality import (
    PRIMAL,
    STATIONARITY,
    check_optimality,
    log_optimality_check,
)

try:
    from pyomo.contrib.appsi.solvers.highs import Highs

    _highs_available = bool(Highs().available())
except ImportError:
    _highs_available = False

try:
    import numpy  # noqa: F401

    _numpy_available = True
except ImportError:
    _numpy_available = False


def build_qp(sense=minimize):
    """Convex QP with a hand-derived KKT point.

    Minimization form::

        min 0.5*x**2 + 0.5*y**2 + x - y
        s.t. x + y >= 1              (c_ge, lower-only)
             0 <= x - y <= 2         (c_rng, ranged, lower side active)
             0 <= x, y <= 3

    KKT point: x = y = 0.5 with duals c_ge = 0.5, c_rng = 1.0 and zero
    reduced costs. The maximization form negates the objective and every
    multiplier (duals become -0.5 and -1.0).
    """
    m = ConcreteModel()
    m.x = Var(bounds=(0, 3), initialize=0.5)
    m.y = Var(bounds=(0, 3), initialize=0.5)
    expr = 0.5 * m.x**2 + 0.5 * m.y**2 + m.x - m.y
    if sense == minimize:
        m.obj = Objective(expr=expr, sense=minimize)
        d_ge, d_rng = 0.5, 1.0
    else:
        m.obj = Objective(expr=-expr, sense=maximize)
        d_ge, d_rng = -0.5, -1.0
    m.c_ge = Constraint(expr=m.x + m.y >= 1)
    m.c_rng = Constraint(expr=(0, m.x - m.y, 2))
    m.dual = Suffix(direction=Suffix.IMPORT)
    m.rc = Suffix(direction=Suffix.IMPORT)
    m.dual[m.c_ge] = d_ge
    m.dual[m.c_rng] = d_rng
    m.rc[m.x] = 0.0
    m.rc[m.y] = 0.0
    return m


def build_lp(sense=minimize):
    """LP with a hand-derived KKT point.

    Minimization form::

        min 3*x + y
        s.t. x + y >= 2   (c1)
             x, y >= 0

    KKT point: x = 0 (at lower bound), y = 2 with dual c1 = 1, rc_x = 2
    and rc_y = 0.
    """
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10), initialize=0.0)
    m.y = Var(bounds=(0, 10), initialize=2.0)
    m.obj = Objective(expr=3 * m.x + m.y, sense=sense)
    m.c1 = Constraint(expr=m.x + m.y >= 2)
    m.dual = Suffix(direction=Suffix.IMPORT)
    m.rc = Suffix(direction=Suffix.IMPORT)
    if sense == minimize:
        m.dual[m.c1] = 1.0
        m.rc[m.x] = 2.0
        m.rc[m.y] = 0.0
    return m


class TestCheckOptimality(unittest.TestCase):
    """Analytic KKT tests that do not require a solver."""

    # ------------------------------------------------------------------
    # Happy path: LP and convex QP, minimization and maximization
    # ------------------------------------------------------------------
    def test_qp_min_optimal(self):
        res = check_optimality(build_qp(minimize))
        self.assertTrue(res.is_optimal)
        if _numpy_available:
            self.assertEqual(res.convexity, 'convex')
        else:
            self.assertEqual(res.convexity, 'not_checked')
        self.assertAlmostEqual(res.max_primal_residual, 0.0)
        self.assertAlmostEqual(res.max_dual_residual, 0.0)
        self.assertAlmostEqual(res.max_stationarity_residual, 0.0)
        self.assertAlmostEqual(res.max_complementarity_residual, 0.0)
        self.assertEqual(len(res.violations()), 0)

    def test_qp_max_optimal_reversed_signs(self):
        res = check_optimality(build_qp(maximize))
        self.assertTrue(res.is_optimal)
        self.assertEqual(res.objective_sense, -1)
        if _numpy_available:
            self.assertEqual(res.convexity, 'convex')
        self.assertEqual(len(res.violations()), 0)

    def test_lp_min_optimal(self):
        res = check_optimality(build_lp(minimize))
        self.assertTrue(res.is_optimal)
        self.assertEqual(res.convexity, 'convex')
        rec_x = next(r for r in res.variable_residuals if r.name == 'x')
        self.assertAlmostEqual(rec_x.reduced_gradient, 2.0)
        self.assertAlmostEqual(rec_x.reduced_cost, 2.0)

    def test_stationarity_identity_qp(self):
        # r_j = grad f - sum dual * grad body must equal the loaded rc.
        res = check_optimality(build_qp())
        for rec in res.variable_residuals:
            self.assertAlmostEqual(rec.reduced_gradient, rec.reduced_cost, places=12)

    # ------------------------------------------------------------------
    # Ranged (two-sided) constraints
    # ------------------------------------------------------------------
    def test_ranged_lower_active_sign(self):
        m = build_qp()  # x - y = 0, lower side active; min => dual >= 0
        m.dual[m.c_rng] = -1.0  # wrong sign
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.dual_sign_residual, 1.0)
        self.assertTrue(rec.ranged)

    def test_ranged_upper_active_sign_min_and_max(self):
        # x - y = 2: upper side active.
        m = build_qp()
        m.x.set_value(2.5)
        m.y.set_value(0.5)
        m.dual[m.c_rng] = -0.7  # min, upper side => dual <= 0: correct
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.0)

        m.dual[m.c_rng] = 0.7  # wrong sign for minimization
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.7)
        self.assertFalse(res.is_optimal)

        # Maximization flips the required upper-side sign to >= 0.
        m = build_qp(maximize)
        m.x.set_value(2.5)
        m.y.set_value(0.5)
        m.dual[m.c_rng] = 0.7
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.0)

    def test_ranged_inactive_nonzero_dual_is_complementarity_violation(self):
        m = build_qp()
        m.x.set_value(1.0)
        m.y.set_value(0.2)  # x - y = 0.8, strictly inside [0, 2]
        m.dual[m.c_rng] = 0.7
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.0)
        self.assertAlmostEqual(rec.complementarity_lb_residual, 0.7 * 0.8)

    # ------------------------------------------------------------------
    # One-sided constraints and equalities
    # ------------------------------------------------------------------
    def test_one_sided_dual_sign_violations(self):
        m = build_lp()
        m.dual[m.c1] = -1.0  # min, lower-only => must be >= 0
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c1')
        self.assertAlmostEqual(rec.dual_sign_residual, 1.0)
        self.assertFalse(res.is_optimal)

        # Upper-only constraint, minimization => dual <= 0. The constraint
        # is active at the QP point (body = 1.5), so the sign is checked.
        m = build_qp()
        m.c_le = Constraint(expr=m.x + 2 * m.y <= 1.5)
        m.dual[m.c_le] = 0.4
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_le')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.4)

    def test_maximization_one_sided_signs_reversed(self):
        m = build_qp(maximize)
        m.dual[m.c_ge] = 0.5  # max, lower-only => must be <= 0
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_ge')
        self.assertAlmostEqual(rec.dual_sign_residual, 0.5)
        self.assertFalse(res.is_optimal)

    def test_equality_dual_has_unrestricted_sign(self):
        m = build_qp()
        m.c_eq = Constraint(expr=m.x + 2 * m.y == 1.5)
        for trial in (-3.2, 0.0, 3.2):
            m.dual[m.c_eq] = trial
            res = check_optimality(m)
            rec = next(r for r in res.constraint_residuals if r.name == 'c_eq')
            self.assertTrue(rec.equality)
            self.assertAlmostEqual(rec.dual_sign_residual, 0.0)
            self.assertAlmostEqual(rec.complementarity_residual, 0.0)

    def test_constraint_complementarity_residual(self):
        m = build_lp()
        m.x.set_value(1.0)
        m.y.set_value(2.0)  # c1 slack = 1 while dual stays 1.0
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c1')
        self.assertAlmostEqual(rec.complementarity_lb_residual, 1.0)
        self.assertFalse(res.is_optimal)

    # ------------------------------------------------------------------
    # Primal feasibility
    # ------------------------------------------------------------------
    def test_primal_constraint_violations(self):
        m = build_qp()
        m.x.set_value(0.1)
        m.y.set_value(0.1)  # x + y = 0.2 < 1
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_ge')
        self.assertAlmostEqual(rec.primal_lb_residual, 0.8)
        self.assertAlmostEqual(res.max_primal_residual, 0.8)
        self.assertFalse(res.is_optimal)

        # Upper-side violation of the ranged constraint.
        m = build_qp()
        m.x.set_value(3.0)
        m.y.set_value(0.0)  # x - y = 3 > 2
        res = check_optimality(m)
        rec = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertAlmostEqual(rec.primal_ub_residual, 1.0)

    def test_primal_variable_bound_violations(self):
        m = build_qp()
        m.x.set_value(-0.25)
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'x')
        self.assertAlmostEqual(rec.primal_lb_residual, 0.25)
        self.assertFalse(res.is_optimal)

        m = build_qp()
        m.y.set_value(3.5)
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'y')
        self.assertAlmostEqual(rec.primal_ub_residual, 0.5)

    # ------------------------------------------------------------------
    # Reduced costs: stationarity, signs and bound complementarity
    # ------------------------------------------------------------------
    def test_stationarity_violation_via_rc_mismatch(self):
        m = build_qp()
        m.rc[m.x] = 0.3
        res = check_optimality(m)
        self.assertAlmostEqual(res.max_stationarity_residual, 0.3)
        self.assertFalse(res.is_optimal)
        cats = {v[0] for v in res.violations()}
        self.assertIn(STATIONARITY, cats)

    def test_rc_sign_at_bounds_min_and_max(self):
        # Minimization: rc at a lower bound must be >= 0.
        m = build_lp()
        m.rc[m.x] = -2.0
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'x')
        self.assertAlmostEqual(rec.dual_sign_residual, 2.0)

        # Maximization: rc at a lower bound must be <= 0.
        m = ConcreteModel()
        m.z = Var(bounds=(0, 5), initialize=0.0)
        m.obj = Objective(expr=m.z, sense=maximize)
        m.c = Constraint(expr=m.z <= 5)
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)
        m.dual[m.c] = 1.0
        m.rc[m.z] = 1.0  # wrong: max at lower bound requires rc <= 0
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'z')
        self.assertAlmostEqual(rec.dual_sign_residual, 1.0)
        self.assertFalse(res.is_optimal)

    def test_interior_reduced_cost_must_vanish(self):
        m = build_lp()  # y = 2 is interior; give it a non-zero rc
        m.rc[m.y] = 0.9
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'y')
        # Scale-free interior penalty and the literal KKT product
        # |rc| * slack = 0.9 * 2.
        self.assertAlmostEqual(rec.interior_multiplier_residual, 0.9)
        self.assertAlmostEqual(rec.complementarity_lb_residual, 1.8)
        self.assertAlmostEqual(rec.complementarity_residual, 1.8)
        self.assertFalse(res.is_optimal)

    def test_variable_bound_complementarity_product(self):
        m = build_lp()
        m.x.set_value(0.5)
        m.y.set_value(1.5)  # c1 stays active; rc_x = 2 with slack 0.5
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'x')
        self.assertAlmostEqual(rec.complementarity_lb_residual, 1.0)
        self.assertAlmostEqual(rec.stationarity_residual, 0.0)
        self.assertFalse(res.is_optimal)

    def test_fixed_variable_skips_stationarity(self):
        m = build_qp()
        m.x.fix(0.5)
        m.rc[m.x] = 99.0  # must be ignored for fixed variables
        res = check_optimality(m)
        rec = next(r for r in res.variable_residuals if r.name == 'x')
        self.assertTrue(rec.fixed)
        self.assertAlmostEqual(rec.stationarity_residual, 0.0)
        self.assertAlmostEqual(rec.dual_sign_residual, 0.0)
        self.assertTrue(res.is_optimal)

    def test_missing_rc_suffix_uses_implied_multipliers(self):
        m = build_qp()
        m.del_component(m.rc)
        res = check_optimality(m)
        self.assertFalse(res.reduced_costs_available)
        self.assertTrue(res.is_optimal)

        # An implied multiplier violation still surfaces as a stationarity
        # residual when no rc suffix is present.
        m = build_lp()
        m.del_component(m.rc)
        m.dual[m.c1] = 0.25  # corrupts the reduced gradient
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)
        self.assertGreater(res.max_stationarity_residual, 0.0)

    # ------------------------------------------------------------------
    # Missing data and input validation
    # ------------------------------------------------------------------
    def test_missing_dual_key_recorded_as_error(self):
        m = build_qp()
        m.dual.clear()
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)
        self.assertTrue(any('missing dual' in e for e in res.errors))
        recs = [r for r in res.constraint_residuals if r.errors]
        self.assertEqual(len(recs), 2)

    def test_missing_variable_value_recorded_as_error(self):
        m = build_qp()
        m.x.value = None
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)
        self.assertTrue(any('x' in e for e in res.errors))

    def test_missing_dual_suffix_raises(self):
        m = build_qp()
        m.del_component(m.dual)
        with self.assertRaisesRegex(ValueError, 'dual'):
            check_optimality(m)

    def test_objective_count_validation(self):
        m = build_qp()
        m.obj.deactivate()
        with self.assertRaisesRegex(ValueError, 'active objective'):
            check_optimality(m)
        m.obj2 = Objective(expr=m.x + m.y)
        m.obj.activate()
        with self.assertRaisesRegex(ValueError, 'active objective'):
            check_optimality(m)

    def test_nonlinear_and_quadratic_constraints_rejected(self):
        m = build_qp()
        m.del_component(m.c_ge)
        m.c_ge = Constraint(expr=m.x * m.y >= 1)
        with self.assertRaisesRegex(ValueError, 'linear constraints'):
            check_optimality(m)

        m = build_qp()
        m.del_component(m.c_ge)
        m.c_ge = Constraint(expr=m.x**3 >= 1)
        with self.assertRaisesRegex(ValueError, 'non-quadratic'):
            check_optimality(m)

    @unittest.skipUnless(_numpy_available, 'NumPy is not available')
    def test_non_convex_objective_flagged(self):
        m = build_qp()
        m.del_component(m.obj)
        m.obj = Objective(expr=-m.x**2 + m.y)
        res = check_optimality(m)
        self.assertEqual(res.convexity, 'non_convex')
        self.assertFalse(res.is_optimal)

        # A concave objective is not valid for maximization either.
        m = build_qp(maximize)
        m.del_component(m.obj)
        m.obj = Objective(expr=m.x**2 + m.y, sense=maximize)
        res = check_optimality(m)
        self.assertEqual(res.convexity, 'non_convex')
        self.assertFalse(res.is_optimal)

    def test_convexity_check_can_be_disabled(self):
        m = build_qp()
        m.del_component(m.obj)
        m.obj = Objective(expr=-m.x**2 + m.y)
        res = check_optimality(m, check_convexity=False)
        self.assertNotEqual(res.convexity, 'non_convex')

    @unittest.skipUnless(_numpy_available, 'NumPy is not available')
    def test_cross_term_hessian_convexity(self):
        m = build_qp()
        m.del_component(m.obj)
        # 0.5*(x + y)**2 has Hessian [[1, 1], [1, 1]]: PSD (eigs 2, 0).
        m.obj = Objective(expr=0.5 * (m.x + m.y) ** 2)
        res = check_optimality(m)
        self.assertEqual(res.convexity, 'convex')

        m.del_component(m.obj)
        # x*y has Hessian [[0, 1], [1, 0]] with eigenvalues +/-1.
        m.obj = Objective(expr=m.x * m.y)
        res = check_optimality(m)
        self.assertEqual(res.convexity, 'non_convex')
        self.assertFalse(res.is_optimal)

    def test_linear_model_reports_convex_without_eigenvalues(self):
        m = build_lp()
        res = check_optimality(m)
        self.assertEqual(res.convexity, 'convex')

    # ------------------------------------------------------------------
    # Structural behavior
    # ------------------------------------------------------------------
    def test_tolerance_overrides(self):
        m = build_qp()
        m.rc[m.x] = 5e-7  # below the default 1e-6 tolerance
        self.assertTrue(check_optimality(m).is_optimal)
        res = check_optimality(m, stationarity_tol=1e-8)
        self.assertFalse(res.is_optimal)
        self.assertAlmostEqual(res.max_stationarity_residual, 5e-7)

    def test_violations_filtering_and_order(self):
        m = build_qp()
        m.dual[m.c_ge] = -0.5  # dual sign violation
        m.x.set_value(-0.25)  # primal violation
        res = check_optimality(m)
        primal_only = res.violations(categories=(PRIMAL,))
        self.assertTrue(all(cat == PRIMAL for cat, _, _, _ in primal_only))
        self.assertGreater(len(primal_only), 0)
        all_violations = res.violations()
        # Constraints are reported before variables; violations() yields
        # (category, component, residual, detail) tuples.
        self.assertEqual(type(all_violations[0][1]).__name__, 'ScalarConstraint')

    def test_write_report(self):
        m = build_qp()
        m.dual[m.c_ge] = -0.5
        stream = StringIO()
        res = check_optimality(m)
        res.write_report(ostream=stream)
        text = stream.getvalue()
        self.assertIn('NOT OPTIMAL', text)
        self.assertIn('c_ge', text)
        self.assertIn('max dual residual', text)

    def test_log_helper(self):
        m = build_qp()
        m.dual[m.c_ge] = -0.5
        with LoggingIntercept(level=logging.INFO) as buf:
            res = log_optimality_check(m)
        text = buf.getvalue()
        self.assertIn('DUAL', text)
        self.assertIn('NOT OPTIMAL', text)
        self.assertFalse(res.is_optimal)

    def test_model_is_not_modified(self):
        m = build_qp()

        def snapshot():
            return {
                'x': m.x.value,
                'y': m.y.value,
                'dual': sorted((c.name, v) for c, v in m.dual.items()),
                'rc': sorted((c.name, v) for c, v in m.rc.items()),
                'n_constraints': len(
                    list(m.component_data_objects(Constraint, descend_into=True))
                ),
                'n_vars': len(list(m.component_data_objects(Var, descend_into=True))),
                'objective_active': m.obj.active,
                'c_ge_active': m.c_ge.active,
            }

        before = snapshot()
        check_optimality(m)
        self.assertEqual(before, snapshot())

    def test_subblock_traversal(self):
        # min x, 0 <= x <= 1, x <= 1: optimum x = 0; the upper constraint
        # is slack (dual 0) and the lower-bound reduced cost equals 1.
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var(bounds=(0, 1), initialize=0.0)
        m.b.obj = Objective(expr=m.b.x)
        m.b.c = Constraint(expr=m.b.x <= 1.0)
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)
        m.dual[m.b.c] = 0.0
        m.rc[m.b.x] = 1.0
        res = check_optimality(m)
        names = {r.name for r in res.variable_residuals}
        self.assertIn('b.x', names)
        self.assertTrue(res.is_optimal)

    def test_indexed_components(self):
        m = ConcreteModel()
        m.idx = RangeSet(1, 2)
        m.x = Var(m.idx, bounds=(0, 5), initialize=1.0)
        m.obj = Objective(expr=sum(m.x[i] for i in m.idx))
        m.c = Constraint(m.idx, rule=lambda b, i: m.x[i] >= 1.0)
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)
        for i in m.idx:
            m.dual[m.c[i]] = 1.0
            m.rc[m.x[i]] = 0.0
        res = check_optimality(m)
        self.assertEqual(len(res.constraint_residuals), 2)
        self.assertEqual(len(res.variable_residuals), 2)
        self.assertTrue(res.is_optimal)


@unittest.skipUnless(_highs_available, 'HiGHS solver is not available')
class TestCheckOptimalityHiGHS(unittest.TestCase):
    """End-to-end checks against a real LP solver (Pyomo suffix values)."""

    def _solve(self, sense):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5), initialize=0)
        m.y = Var(bounds=(0, None), initialize=0)
        m.z = Var(bounds=(None, 4), initialize=0)
        m.w = Var(bounds=(-10, 10), initialize=0)
        m.obj = Objective(expr=3 * m.x + 2 * m.y - m.z + m.w, sense=sense)
        m.c_ge = Constraint(expr=m.x + m.y >= 2)
        m.c_le = Constraint(expr=m.z + m.w <= 3)
        m.c_eq = Constraint(expr=m.x - m.z == 1)
        m.c_rng = Constraint(expr=(-1, m.y + m.z, 1))
        opt = Highs()
        opt.solve(m)
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)
        m.dual.update(opt.get_duals())
        m.rc.update(opt.get_reduced_costs())
        return m

    def test_lp_min_end_to_end(self):
        m = self._solve(minimize)
        res = check_optimality(m)
        self.assertTrue(res.is_optimal, res.violations())
        # Ranged constraint ends at its upper bound under minimization:
        # the dual must carry the upper-side (non-positive) sign.
        rng = next(r for r in res.constraint_residuals if r.name == 'c_rng')
        self.assertTrue(rng.ranged)
        self.assertLessEqual(value(rng.dual), 1e-9)

    def test_lp_max_end_to_end(self):
        m = self._solve(maximize)
        res = check_optimality(m)
        self.assertTrue(res.is_optimal, res.violations())
        self.assertEqual(res.objective_sense, -1)
        # Every loaded reduced cost must satisfy the stationarity identity.
        for rec in res.variable_residuals:
            self.assertAlmostEqual(rec.stationarity_residual, 0.0, places=9)

    def test_solver_solution_with_injected_perturbation(self):
        m = self._solve(minimize)
        some_var = next(iter(m.component_data_objects(Var)))
        some_var.set_value(some_var.value + 0.5)
        res = check_optimality(m)
        self.assertFalse(res.is_optimal)


if __name__ == '__main__':
    unittest.main()
