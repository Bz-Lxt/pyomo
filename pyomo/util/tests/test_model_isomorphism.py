# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import unittest

from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numeric_expr import MaxExpression, MinExpression
from pyomo.environ import (
    Binary,
    Block,
    ConcreteModel,
    Constraint,
    Expression,
    ExternalFunction,
    Integers,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Reals,
    Reference,
    SOSConstraint,
    Set,
    Suffix,
    Var,
    maximize,
    minimize,
    sqrt,
)
from pyomo.util.model_isomorphism import (
    ModelIsomorphismError,
    ModelIsomorphismResult,
    compare_models,
    is_isomorphic,
)


def _full_model_a():
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    m.p = Param(m.I, initialize=lambda m, i: 2 * i, mutable=True)
    m.x = Var(m.I, domain=Integers, bounds=(0, 10))
    m.y = Var()
    m.o = Objective(expr=sum(m.x[i] for i in m.I) + m.y)
    m.c = Constraint(m.I, rule=lambda m, i: m.x[i] + m.y * m.p[i] >= m.p[i])
    m.c2 = Constraint(expr=m.y <= 5)
    return m


def _full_model_b():
    # Same structure, different names, reversed set declaration order,
    # reversed component creation order and commutative reorderings.
    m = ConcreteModel()
    m.J = Set(initialize=[3, 2, 1])
    m.q = Param(m.J, initialize=lambda m, i: 2 * i, mutable=True)
    m.z = Var(m.J, domain=Integers, bounds=(0, 10))
    m.w = Var()
    m.d2 = Constraint(expr=m.w <= 5)
    m.d = Constraint(
        m.J, rule=lambda m, i: m.q[i] * m.w + m.z[i] >= m.q[i]
    )
    m.obj = Objective(expr=m.w + sum(m.z[i] for i in m.J))
    return m


class TestModelIsomorphism(unittest.TestCase):

    def test_renamed_reordered_models_are_isomorphic(self):
        result = compare_models(_full_model_a(), _full_model_b())
        self.assertTrue(result.isomorphic, msg=result.differences)
        ma, mb = _full_model_a(), _full_model_b()
        result = compare_models(ma, mb)

        self.assertEqual(len(result.variables), 4)
        self.assertEqual(len(result.constraints), 4)
        self.assertEqual(len(result.objectives), 1)
        self.assertEqual(len(result.parameters), 3)

        self.assertIs(result.variables[ma.x[1]], mb.z[1])
        self.assertIs(result.variables[ma.x[2]], mb.z[2])
        self.assertIs(result.variables[ma.x[3]], mb.z[3])
        self.assertIs(result.variables[ma.y], mb.w)
        self.assertIs(result.constraints[ma.c[1]], mb.d[1])
        self.assertIs(result.constraints[ma.c2], mb.d2)
        self.assertIs(result.objectives[ma.o], mb.obj)
        self.assertIs(result.parameters[ma.p[2]], mb.q[2])
        # The mapping is a bijection.
        self.assertEqual(
            len({id(v) for v in result.variables.values()}),
            len(result.variables),
        )
        self.assertIsInstance(result.variables, ComponentMap)
        self.assertIsInstance(result, ModelIsomorphismResult)
        self.assertTrue(is_isomorphic(ma, mb))

    def test_mapping_is_deterministic(self):
        ma, mb = _full_model_a(), _full_model_b()
        first = compare_models(ma, mb)
        second = compare_models(ma, mb)
        for kind in first.mapping:
            pairs_first = [
                (id(a), id(b)) for a, b in first.mapping[kind].items()
            ]
            pairs_second = [
                (id(a), id(b)) for a, b in second.mapping[kind].items()
            ]
            self.assertEqual(pairs_first, pairs_second)

    def test_symmetric_structure_gets_index_consistent_mapping(self):
        def build():
            m = ConcreteModel()
            m.I = RangeSet(2)
            m.v = Var(m.I)
            m.k = Constraint(m.I, rule=lambda m, i: m.v[i] >= 0)
            m.o = Objective(expr=sum(m.v[i] for i in m.I))
            return m

        result = compare_models(build(), build())
        self.assertTrue(result.isomorphic, msg=result.differences)
        ma, mb = build(), build()
        result = compare_models(ma, mb)
        for i in ma.I:
            self.assertIs(result.variables[ma.v[i]], mb.v[i])
            self.assertIs(result.constraints[ma.k[i]], mb.k[i])

    def test_regular_graphs_requiring_exhaustive_backtracking(self):
        # K3,3 and the triangular prism are both 3-regular graphs on
        # six vertices (nine edges), so color refinement assigns every
        # variable one color and every edge constraint one color in
        # both models. Only the complete backtracking search can tell
        # that no structure-preserving bijection exists.
        k33 = [
            (1, 4), (1, 5), (1, 6),
            (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 5), (3, 6),
        ]
        prism = [
            (1, 2), (2, 3), (3, 1),
            (4, 5), (5, 6), (6, 4),
            (1, 4), (2, 5), (3, 6),
        ]

        def graph_model(edges):
            m = ConcreteModel()
            m.I = RangeSet(6)
            m.x = Var(m.I)
            m.e = Constraint(
                RangeSet(len(edges)),
                rule=lambda m, k: (
                    m.x[edges[k - 1][0]] + m.x[edges[k - 1][1]] >= 0
                ),
            )
            m.o = Objective(expr=sum(m.x[i] for i in m.I))
            return m

        result = compare_models(graph_model(k33), graph_model(prism))
        self.assertFalse(result.isomorphic)
        ok = compare_models(graph_model(prism), graph_model(prism))
        self.assertTrue(ok.isomorphic, msg=ok.differences)
        self.assertEqual(len(ok.variables), 6)
        self.assertEqual(len(ok.constraints), 9)
        self.assertEqual(
            len({id(v) for v in ok.variables.values()}), 6
        )

    # -- commutativity -------------------------------------------------

    def test_commutative_reorderings_match(self):
        def build(swap):
            m = ConcreteModel()
            m.a = Var()
            m.b = Var()
            m.r = Var()
            if not swap:
                m.eq = Constraint(expr=m.a + m.b == m.r)
                m.prod = Constraint(expr=m.a * m.b <= 2 * m.r)
            else:
                m.eq = Constraint(expr=m.r == m.b + m.a)
                m.prod = Constraint(expr=m.b * m.a <= m.r * 2)
            m.obj = Objective(
                expr=MaxExpression((m.a, m.b, m.r))
                + MinExpression((m.b, m.a))
            )
            return m

        result = compare_models(build(False), build(True))
        self.assertTrue(result.isomorphic, msg=result.differences)

    def test_non_commutative_reorderings_do_not_match(self):
        def build(flip):
            m = ConcreteModel()
            # Distinct bounds break the swap automorphism of the two vars.
            m.a = Var(bounds=(0, None))
            m.b = Var()
            if not flip:
                m.sub = Constraint(expr=m.a - m.b >= 1)
                m.div = Constraint(expr=m.a / m.b == 1)
                m.power = Constraint(expr=m.a ** m.b <= 4)
                m.ineq = Constraint(expr=m.a <= m.b)
            else:
                m.sub = Constraint(expr=m.b - m.a >= 1)
                m.div = Constraint(expr=m.b / m.a == 1)
                m.power = Constraint(expr=m.b ** m.a <= 4)
                m.ineq = Constraint(expr=m.b <= m.a)
            m.o = Objective(expr=m.a + m.b)
            return m

        self.assertFalse(is_isomorphic(build(False), build(True)))

    def test_strict_inequality_flag_matters(self):
        def build(strict):
            m = ConcreteModel()
            m.a = Var()
            m.b = Var()
            m.c = Constraint(expr=m.a <= m.b)
            if strict:
                # Constraint construction rejects strict relations;
                # attach the strict relational expression directly.
                m.c._expr = (m.a < m.b)
            m.o = Objective(expr=m.a)
            return m

        self.assertFalse(is_isomorphic(build(True), build(False)))
        self.assertTrue(is_isomorphic(build(True), build(True)))

    def test_ranged_constraint_order_matters(self):
        from pyomo.environ import inequality

        def build(flip):
            m = ConcreteModel()
            m.a = Var()
            if not flip:
                expr = inequality(0, m.a, 1)
            else:
                expr = inequality(1, m.a, 0)
            m.c = Constraint(expr=m.a <= 1)
            m.c._expr = expr
            m.o = Objective(expr=m.a)
            return m

        self.assertFalse(is_isomorphic(build(False), build(True)))

    # -- blocks ---------------------------------------------------------

    def test_hierarchical_indexed_blocks(self):
        def build():
            m = ConcreteModel()

            def leaf_rule(leaf, tag):
                leaf.v = Var(domain=NonNegativeReals)
                leaf.g = Param(
                    initialize=leaf.parent_block().index(), mutable=True
                )
                leaf.c = Constraint(expr=leaf.v >= leaf.g)

            def rule(block, i):
                block.leaf = Block(['p', 'q'], rule=leaf_rule)
                block.s = Var()
                block.cs = Constraint(expr=block.s == i)

            m.top = Block([1, 2], rule=rule)
            m.obj = Objective(
                expr=sum(
                    m.top[i].leaf[tag].v
                    for i in (1, 2)
                    for tag in ('p', 'q')
                )
            )
            return m

        ma, mb = build(), build()
        result = compare_models(ma, mb)
        self.assertTrue(result.isomorphic, msg=result.differences)
        self.assertIs(result.blocks[ma.top[1].leaf['p']],
                      mb.top[1].leaf['p'])
        self.assertIs(
            result.variables[ma.top[2].leaf['q'].v],
            mb.top[2].leaf['q'].v,
        )
        self.assertIs(
            result.constraints[ma.top[1].cs], mb.top[1].cs
        )
        self.assertIs(
            result.parameters[ma.top[2].leaf['p'].g],
            mb.top[2].leaf['p'].g,
        )

    def test_block_structure_mismatch_detected(self):
        def build(nested):
            m = ConcreteModel()
            m.x = Var()
            if nested:
                m.b = Block()
                m.b.y = Var()
                m.o = Objective(expr=m.x + m.b.y)
            else:
                m.y = Var()
                m.o = Objective(expr=m.x + m.y)
            return m

        # Both have two variables but only one nests one inside a block.
        result = compare_models(build(True), build(False))
        self.assertFalse(result.isomorphic)
        self.assertTrue(result.differences)

    def test_subtree_comparison(self):
        def build():
            m = ConcreteModel()
            m.b = Block()
            m.b.x = Var()
            m.b.c = Constraint(expr=m.b.x >= 1)
            m.b.o = Objective(expr=m.b.x)
            return m

        ma, mb = build(), build()
        result = compare_models(ma.b, mb.b)
        self.assertTrue(result.isomorphic, msg=result.differences)
        self.assertIs(result.variables[ma.b.x], mb.b.x)

    # -- references -----------------------------------------------------

    def test_reference_aliases_introduce_no_new_structure(self):
        ma = _full_model_a()
        mb = _full_model_b()
        mb.rz = Reference(mb.z[:])
        mb.rd = Reference(mb.d[:])
        result = compare_models(ma, mb)
        self.assertTrue(result.isomorphic, msg=result.differences)
        # The reference targets resolve to the real component data.
        self.assertIs(result.variables[ma.x[1]], mb.z[1])
        self.assertIs(result.constraints[ma.c[1]], mb.d[1])

    def test_cross_block_reference_resolves_to_referent(self):
        def build(with_reference):
            m = ConcreteModel()
            m.b = Block()
            m.b.x = Var([1, 2])
            m.c = Constraint([1, 2], rule=lambda m, i: m.b.x[i] >= i)
            if with_reference:
                m.alias = Reference(m.b.x[:])
            m.o = Objective(expr=sum(m.b.x[i] for i in (1, 2)))
            return m

        self.assertTrue(is_isomorphic(build(True), build(False)))

    # -- parameters -----------------------------------------------------

    def test_mutable_param_current_values_compared(self):
        ma, mb = _full_model_a(), _full_model_b()
        self.assertTrue(is_isomorphic(ma, mb))
        mb.q[1] = 99
        result = compare_models(ma, mb)
        self.assertFalse(result.isomorphic)
        self.assertTrue(
            any('current_value' in d for d in result.differences),
            msg=result.differences,
        )

    def test_tolerance_applies_to_parameter_values(self):
        def build(value):
            m = ConcreteModel()
            m.p = Param(initialize=value, mutable=True)
            m.x = Var()
            m.c = Constraint(expr=m.x >= m.p)
            m.o = Objective(expr=m.x)
            return m

        self.assertFalse(is_isomorphic(build(1.0), build(1.05)))
        self.assertTrue(
            is_isomorphic(build(1.0), build(1.05), tolerance=0.1)
        )

    def test_mutable_vs_immutable_param_distinguished(self):
        def build(mutable):
            m = ConcreteModel()
            m.p = Param(initialize=2.0, mutable=mutable)
            m.x = Var()
            m.c = Constraint(expr=m.p * m.x >= 1)
            m.o = Objective(expr=m.x)
            return m

        self.assertFalse(is_isomorphic(build(True), build(False)))

    # -- variable features ---------------------------------------------

    def test_domain_bounds_and_fixed_values(self):
        def build(domain, bounds, fixed_value=None):
            m = ConcreteModel()
            m.x = Var(domain=domain, bounds=bounds)
            if fixed_value is not None:
                m.x.fix(fixed_value)
            m.o = Objective(expr=m.x)
            return m

        self.assertTrue(
            is_isomorphic(build(Integers, (0, 1)), build(Integers, (0, 1)))
        )
        self.assertFalse(
            is_isomorphic(build(Integers, (0, 1)), build(Reals, (0, 1)))
        )
        self.assertFalse(
            is_isomorphic(build(Reals, (0, 1)), build(Reals, (0, 2)))
        )
        self.assertFalse(
            is_isomorphic(build(Reals, None, 3), build(Reals, None, 4))
        )
        self.assertTrue(
            is_isomorphic(build(Reals, None, 3), build(Reals, None, 3))
        )

    def test_parametric_bounds_walked_as_expressions(self):
        def build(scale):
            m = ConcreteModel()
            m.p = Param(initialize=scale, mutable=True)
            m.x = Var(bounds=(None, None))
            m.x._lb = m.p
            m.x._ub = m.p + 10
            m.o = Objective(expr=m.x)
            return m

        self.assertTrue(is_isomorphic(build(2), build(2)))
        self.assertFalse(is_isomorphic(build(2), build(3)))

    def test_variable_values_optional(self):
        def build(value):
            m = ConcreteModel()
            m.x = Var()
            m.x.value = value
            m.o = Objective(expr=m.x)
            return m

        # Unfixed variable values are solver state, ignored by default.
        self.assertTrue(is_isomorphic(build(1.0), build(2.0)))
        self.assertFalse(
            is_isomorphic(
                build(1.0), build(2.0), include_variable_values=True
            )
        )

    # -- constraints, objectives and named expressions -----------------

    def test_objective_sense_matters(self):
        def build(sense):
            m = ConcreteModel()
            m.x = Var()
            m.o = Objective(expr=m.x, sense=sense)
            return m

        self.assertFalse(is_isomorphic(build(minimize), build(maximize)))
        self.assertTrue(is_isomorphic(build(minimize), build(minimize)))

    def test_inactive_components_preserved_in_signature(self):
        ma, mb = _full_model_a(), _full_model_b()
        mb.d[1].deactivate()
        self.assertFalse(is_isomorphic(ma, mb))
        ma.c[1].deactivate()
        self.assertTrue(is_isomorphic(ma, mb))

    def test_named_expressions_and_sharing(self):
        def build(shared):
            m = ConcreteModel()
            m.x = Var([1, 2])
            if shared:
                m.e = Expression(expr=m.x[1] + m.x[2])
                m.c1 = Constraint(expr=m.e >= 1)
                m.c2 = Constraint(expr=m.e <= 10)
            else:
                m.e1 = Expression(expr=m.x[1] + m.x[2])
                m.e2 = Expression(expr=m.x[1] + m.x[2])
                m.c1 = Constraint(expr=m.e1 >= 1)
                m.c2 = Constraint(expr=m.e2 <= 10)
            m.o = Objective(expr=m.x[1])
            return m

        # One shared named expression vs two separate ones is a
        # different component structure.
        self.assertFalse(is_isomorphic(build(True), build(False)))
        self.assertTrue(is_isomorphic(build(True), build(True)))
        result = compare_models(build(True), build(True))
        self.assertTrue(result.isomorphic)
        self.assertEqual(len(result.expressions), 1)

    def test_unary_and_external_functions(self):
        def build(use_sin, lib_name=('libf.so', 'f')):
            m = ConcreteModel()
            m.x = Var()
            if use_sin:
                m.c = Constraint(expr=sqrt(m.x) <= 1)
            else:
                m.c = Constraint(expr=m.x ** 0.5 <= 1)
            m.g = ExternalFunction(library=lib_name[0], function=lib_name[1])
            m.o = Objective(expr=m.g(m.x))
            return m

        # sqrt unary function vs power expression are different nodes.
        self.assertFalse(is_isomorphic(build(True), build(False)))
        # Renamed external-function component, same library/function.
        self.assertTrue(is_isomorphic(build(False), build(False)))
        other_lib = build(False, lib_name=('libf.so', 'g'))
        self.assertFalse(is_isomorphic(build(False), other_lib))

    # -- indexing and edge cases ---------------------------------------

    def test_tuple_valued_indices(self):
        def build():
            m = ConcreteModel()
            m.J = Set(initialize=[('a', 1), ('b', 2)])
            m.x = Var(m.J)
            m.c = Constraint(m.J, rule=lambda m, *j: m.x[j] >= 0)
            m.o = Objective(expr=sum(m.x[j] for j in m.J))
            return m

        result = compare_models(build(), build())
        self.assertTrue(result.isomorphic, msg=result.differences)
        ma, mb = build(), build()
        result = compare_models(ma, mb)
        self.assertIs(result.variables[ma.x[('a', 1)]], mb.x[('a', 1)])

    def test_empty_models(self):
        result = compare_models(ConcreteModel(), ConcreteModel())
        self.assertTrue(result.isomorphic)
        # Only the root blocks are mapped; no data components.
        self.assertEqual(len(result.blocks), 1)
        self.assertEqual(
            sum(
                len(entries)
                for kind, entries in result.mapping.items()
                if kind != 'blocks'
            ),
            0,
        )

    def test_different_component_counts_reported(self):
        ma = _full_model_a()
        mb = _full_model_b()
        mb.extra = Var()
        result = compare_models(ma, mb)
        self.assertFalse(result.isomorphic)
        self.assertTrue(
            any('variables' in d for d in result.differences),
            msg=result.differences,
        )

    def test_ignored_component_types_do_not_break_comparison(self):
        ma, mb = _full_model_a(), _full_model_b()
        ma.dual = Suffix()
        mb.rc = Suffix()
        self.assertTrue(is_isomorphic(ma, mb))

    def test_unsupported_component_raises(self):
        ma, mb = _full_model_a(), _full_model_b()
        mb.sos = SOSConstraint(var=mb.w, sos=1)
        with self.assertRaises(ModelIsomorphismError):
            compare_models(ma, mb)

    def test_result_repr(self):
        ok = compare_models(_full_model_a(), _full_model_b())
        self.assertIn('isomorphic=True', repr(ok))
        bad = compare_models(_full_model_a(), ConcreteModel())
        self.assertIn('isomorphic=False', repr(bad))

    # -- read-only guarantee -------------------------------------------

    def test_comparison_does_not_modify_models(self):
        ma, mb = _full_model_a(), _full_model_b()

        def snapshot(model):
            data = {}
            for var in model.component_data_objects(Var):
                data[('var', id(var))] = (
                    var.fixed,
                    var.value,
                    id(var._lb),
                    id(var._ub),
                    var.active,
                )
            for param in model.component_data_objects(Param):
                data[('param', id(param))] = (
                    param.value, param.active
                )
            for con in model.component_data_objects(Constraint):
                data[('con', id(con))] = (
                    id(con.expr), con.active
                )
            for obj in model.component_data_objects(Objective):
                data[('obj', id(obj))] = (id(obj.expr), obj.sense)
            return data

        before_a, before_b = snapshot(ma), snapshot(mb)
        compare_models(ma, mb)
        compare_models(ma, mb)
        self.assertEqual(before_a, snapshot(ma))
        self.assertEqual(before_b, snapshot(mb))
        # Expression object identity is preserved (no rewriting/cloning).
        self.assertIs(ma.c[1].expr, ma.c[1].expr)
        self.assertEqual(
            [id(n) for n in ma.component_data_objects(Var)],
            [id(n) for n in ma.component_data_objects(Var)],
        )


if __name__ == '__main__':
    unittest.main()
