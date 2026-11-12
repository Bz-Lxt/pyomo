# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Structural isomorphism comparison for (concrete) Pyomo models.

This module determines whether two Pyomo models are *structurally
isomorphic*: they contain the same optimization structure even if
components have different names and were created in a different order.
When the models are isomorphic, a deterministic, one-to-one mapping of
the variable, constraint, objective, parameter, named-expression and
block data objects between the two models is returned.

The comparison deliberately ignores

* component names (local names attached to blocks), and
* component creation / declaration order,

while faithfully preserving

* the hierarchical :class:`Block` tree,
* the full algebraic structure of every constraint, objective,
  named expression and variable bound,
* the commutativity of addition, multiplication, equality, ``max`` and
  ``min`` (so ``a + b`` matches ``b + a``),
* :func:`Reference` aliases (a reference does not introduce new
  structural entities; it resolves to the referenced component data),
* the *current values* of mutable :class:`Param` objects, and
* variable domains, bounds, fixed status and fixed values, constraint
  sense/ranges and objective sense.

The comparison is strictly read-only: neither model is modified,
solved, constructed or evaluated in any way.

"""

import sys

from pyomo.common.collections import ComponentMap
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.objective import Objective
from pyomo.core.base.param import Param
from pyomo.core.base.var import Var
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

__all__ = [
    'ModelIsomorphismResult',
    'ModelIsomorphismError',
    'compare_models',
    'is_isomorphic',
]

#: Component types that participate in the structural graph.
_DATA_CTYPES = (Var, Param, Constraint, Objective, Expression)

#: Component types that do not carry optimization structure and are
#: therefore ignored (index structure is still captured through the
#: index sets of the indexed components that use them).
_IGNORED_CTYPE_NAMES = frozenset(
    (
        'Suffix',
        'Set',
        'RangeSet',
        'BuildAction',
        'BuildCheck',
        'ExternalFunction',
    )
)

_KIND_RANK = {
    'block': 0,
    'variable': 1,
    'parameter': 2,
    'expression': 3,
    'constraint': 4,
    'objective': 5,
}

#: Expression node labels whose arguments form an unordered multiset.
_COMMUTATIVE_LABELS = frozenset(
    (
        'SumExpression',
        'ProductExpression',
        'EqualityExpression',
        'MaxExpression',
        'MinExpression',
    )
)

#: Upper bound on color-refinement rounds. Refinement is only a pruning
#: preprocessor: the backtracking matcher is complete on its own, so
#: capping refinement bounds runtime on long chain-like structures
#: without affecting correctness.
_MAX_REFINEMENT_ROUNDS = 12


class ModelIsomorphismError(ValueError):
    """Raised when structural comparison cannot be performed honestly.

    This is raised for models containing component types or index keys
    that :func:`compare_models` does not know how to normalize, *never*
    as a substitute for reporting that two models are not isomorphic.
    """


# -------------------------------------------------------------------------------------
# Deterministic, name-free normalization of plain data
# -------------------------------------------------------------------------------------


def _normalize_scalar(value):
    """Normalize a native (non-component) index or set value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_scalar(v) for v in value)
    raise ModelIsomorphismError(
        f"Cannot normalize index/set value of type {type(value).__name__}: "
        f"{value!r}. Only numeric, string, boolean, None and tuple values "
        "are supported as structural indices."
    )


def _normalize_key(index):
    """Return a canonical, hashable tuple for a component data index."""
    if index is None:
        return ()
    if isinstance(index, tuple):
        return tuple(_normalize_scalar(v) for v in index)
    return (_normalize_scalar(index),)


def _constant_token(value, tolerance):
    """Build a structural token for a native constant leaf.

    When ``tolerance`` is provided, numeric values are quantized onto a
    grid of width ``tolerance`` so that every downstream comparison
    (local signatures, color refinement, expression matching) shares a
    single, transitive notion of numeric equality.
    """
    if value is None:
        return ('k', 'none', None)
    if isinstance(value, bool):
        return ('k', 'bool', value)
    if isinstance(value, str):
        return ('k', 'str', value)
    if isinstance(value, (int, float)) or type(value) in nonpyomo_leaf_types:
        # int and float are merged into a single numeric label so that
        # literal 2 and literal 2.0 compare equal.
        if tolerance is not None and isinstance(value, (int, float)):
            value = round(value / tolerance) * tolerance
        return ('k', 'num', value)
    return ('k', type(value).__name__, value)


def _sort_key(token):
    """Return a totally ordered, deterministic key for expression tokens."""
    if isinstance(token, tuple):
        return tuple(_sort_key(item) for item in token)
    if isinstance(token, str):
        return ('s', token)
    if isinstance(token, bool):
        return ('b', token)
    if isinstance(token, (int, float)):
        return ('n', float(token))
    if token is None:
        return ('z',)
    return ('r', repr(token))


def tokens_equal(left, right, tolerance=None):
    """Recursively compare expression tokens, optionally with tolerance.

    This performs *positional* comparison. It is appropriate for fully
    mapped tokens (final verification) and color refinement, where
    commutative children have already been canonically sorted. Use
    :class:`_TokenMatcher` while a partial component mapping makes the
    relative order of unmapped leaves insignificant.
    """
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not (isinstance(left, tuple) and isinstance(right, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(
            tokens_equal(a, b, tolerance) for a, b in zip(left, right)
        )
    if left == right:
        return True
    if tolerance is not None and _is_numeric_pair(left, right):
        return abs(float(left) - float(right)) <= tolerance
    return False


class _TokenMatcher:
    """Token equality under a partial component mapping.

    Each token is reduced to a hashable *canonical form* in which

    * mapped (``'m'``) and unmapped (``'u'``) component leaves are
      indistinguishable whenever their kind and refined color agree, and
    * the children of a commutative operator become an unordered
      multiset (a sorted tuple of child canonical forms).

    Two tokens therefore match exactly when their canonical forms are
    equal. This turns a commutative child comparison into linear
    bucket counting over hashable forms instead of a quadratic
    bipartite-matching search. Canonical forms are memoized per token
    object for the lifetime of a matcher.
    """

    def __init__(self):
        self._canon_cache = {}

    def match(self, left, right):
        return self.canonical(left) == self.canonical(right)

    def canonical(self, token):
        if not isinstance(token, tuple):
            return ('atom', token)
        cached = self._canon_cache.get(id(token))
        if cached is not None:
            return cached
        tag = token[0]
        if tag in ('m', 'u'):
            # ('m'|'u', kind, color): mapped and unmapped leaves with
            # the same kind and refined color are interchangeable.
            form = ('L', token[1], token[2])
        elif tag == 'id':
            form = ('I', token[1], token[2])
        elif tag == 'o':
            # ('o', label, supplemental, children)
            label, supplemental, children = token[1], token[2], token[3]
            if label in _COMMUTATIVE_LABELS:
                child_forms = tuple(
                    sorted(
                        (self.canonical(child) for child in children),
                        key=_sort_key,
                    )
                )
            else:
                child_forms = tuple(
                    self.canonical(child) for child in children
                )
            form = ('O', label, supplemental, child_forms)
        else:
            # Constant ('k', ...) and ('nil',) tokens are already
            # canonical; numeric constants were quantized onto the
            # tolerance grid when the token was built.
            form = token
        self._canon_cache[id(token)] = form
        return form


def _is_numeric_pair(left, right):
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    )


# -------------------------------------------------------------------------------------
# Expression canonicalization
# -------------------------------------------------------------------------------------


def _operator_label(node):
    """Normalize the class name of an expression node.

    NPV (constant-folding) variants, the linear sum/monomial helpers and
    trivial relational wrappers are collapsed onto their structural
    equivalents so that equivalent algebra always compares equal.
    """
    name = type(node).__name__
    if name.startswith('NPV_'):
        name = name[4:]
    if name == 'LinearExpression':
        return 'SumExpression'
    if name == 'MonomialTermExpression':
        return 'ProductExpression'
    if name == 'TrivialRelationalExpression':
        return 'InequalityExpression'
    return name


class _ExpressionTokenizer(StreamBasedExpressionVisitor):
    """Build canonical, hashable token tuples from expression trees.

    Parameters
    ----------
    index : _ModelIndex
        The index of the model owning the expression.
    leaf_gid : callable
        Function returning the current color (group id) of an unmapped
        component leaf.
    leaf_mapped_token : callable or None
        Function returning the token for a component leaf that is
        already matched through the partial bijection, or ``None`` to
        treat every leaf as unmapped.
    tolerance : float or None
        Numeric tolerance for native constant leaves.
    """

    def __init__(self, index, leaf_gid, leaf_mapped_token, tolerance):
        super().__init__()
        self._index = index
        self._leaf_gid = leaf_gid
        self._leaf_mapped_token = leaf_mapped_token
        self._tolerance = tolerance

    def initializeWalker(self, expr):
        if expr is None:
            return False, ('nil',)
        return True, None

    def enterNode(self, node):
        node_type = type(node)
        if node_type in nonpyomo_leaf_types or not hasattr(
            node, 'is_expression_type'
        ):
            return (), []
        if (
            node.is_variable_type()
            or node.is_parameter_type()
            or node.is_named_expression_type()
        ):
            return (), []
        return node.args, []

    def exitNode(self, node, data):
        node_type = type(node)
        if node_type in nonpyomo_leaf_types or not hasattr(
            node, 'is_expression_type'
        ):
            return _constant_token(node, self._tolerance)
        if node.is_variable_type():
            return self._component_leaf(node, 'variable')
        if node.is_parameter_type():
            return self._component_leaf(node, 'parameter')
        if node.is_named_expression_type():
            return self._component_leaf(node, 'expression')
        return self._operator_token(node, data)

    def _component_leaf(self, node, kind):
        if self._leaf_mapped_token is not None:
            mapped = self._leaf_mapped_token(node)
            if mapped is not None:
                return ('m', kind, mapped)
        leaf_node = self._index.node_by_object[node]
        return ('u', kind, self._leaf_gid(leaf_node))

    def _operator_token(self, node, child_tokens):
        label = _operator_label(node)
        supplemental = None
        if label == 'InequalityExpression':
            supplemental = ('strict', bool(node.strict))
        elif label == 'RangedExpression':
            supplemental = ('strict', tuple(bool(flag) for flag in node.strict))
        elif label in ('UnaryFunctionExpression', 'AbsExpression'):
            supplemental = ('fcn', node.getname())
        elif label == 'ExternalFunctionExpression':
            fcn = node._fcn
            supplemental = ('ext', getattr(fcn, '_library', None), getattr(
                fcn, '_function', None
            ))
        if label in _COMMUTATIVE_LABELS:
            child_tokens = tuple(sorted(child_tokens, key=_sort_key))
        else:
            child_tokens = tuple(child_tokens)
        return ('o', label, supplemental, child_tokens)


# -------------------------------------------------------------------------------------
# Per-model structural index
# -------------------------------------------------------------------------------------


class _Node:
    """A structural graph node (a component data object or block data)."""

    __slots__ = (
        'obj',
        'kind',
        'key',
        'block',
        'container',
        'index_set_signature',
        'order_key',
        'sort_key',
        'base_signature',
        'expression_specs',
        'gid',
    )

    def __init__(self, obj, kind, key):
        self.obj = obj
        self.kind = kind
        self.key = key
        self.block = None
        self.container = None
        self.index_set_signature = ()
        self.order_key = None
        self.sort_key = None
        self.base_signature = None
        self.expression_specs = ()
        self.gid = None


class _ModelIndex:
    def __init__(self, model, tolerance, include_variable_values):
        self.model = model
        self.tolerance = tolerance
        self.include_variable_values = include_variable_values
        self.nodes = []
        self.node_by_object = ComponentMap()
        self.unsupported_ctypes = set()
        self._block_nodes = ComponentMap()
        self._container_nodes = {}
        self._container_signatures = {}

        self._collect_blocks()
        self._collect_components()
        self._finalize_relationships()
        self._build_base_signatures()

    # -- collection -----------------------------------------------------

    def _collect_blocks(self):
        for block in self.model.block_data_objects(active=None, descend_into=True):
            node = _Node(block, 'block', _normalize_key(block.index()))
            node.block = block.parent_block()
            node.container = block.parent_component()
            self._block_nodes[block] = node
            self.nodes.append(node)
            self.node_by_object[block] = node

    def _collect_components(self):
        for block in self._block_nodes:
            for component in block.component_objects(
                active=None, descend_into=False
            ):
                # Reference containers reuse the referent's component
                # data objects, which are already collected through the
                # real container; the reference itself introduces no
                # structural entity.
                if hasattr(component, 'referent'):
                    continue
                if component.ctype is Block:
                    continue
                ctype_name = component.ctype.__name__
                if ctype_name in _IGNORED_CTYPE_NAMES:
                    continue
                if not isinstance(component, _DATA_CTYPES):
                    self.unsupported_ctypes.add(ctype_name)
                    continue
                kind = self._kind_name(component)
                data_nodes = []
                for data in component.values():
                    node = _Node(data, kind, _normalize_key(data.index()))
                    node.block = block
                    node.container = component
                    node.expression_specs = self._expression_specs(kind, data)
                    data_nodes.append(node)
                    self.nodes.append(node)
                    self.node_by_object[data] = node
                self._container_nodes[id(component)] = data_nodes

    @staticmethod
    def _kind_name(component):
        if component.ctype is Var:
            return 'variable'
        if component.ctype is Param:
            return 'parameter'
        if component.ctype is Constraint:
            return 'constraint'
        if component.ctype is Objective:
            return 'objective'
        return 'expression'

    @staticmethod
    def _expression_specs(kind, data):
        if kind == 'constraint':
            expr = data.expr
            return (('constraint', expr),) if expr is not None else ()
        if kind == 'objective':
            expr = data.expr
            return (('objective', expr),) if expr is not None else ()
        if kind == 'expression':
            expr = data.expr
            return (('named_expression', expr),) if expr is not None else ()
        if kind == 'variable':
            specs = []
            if data._lb is not None:
                specs.append(('lb', data._lb))
            if data._ub is not None:
                specs.append(('ub', data._ub))
            return tuple(specs)
        return ()

    # -- relationship finalization -------------------------------------

    def _finalize_relationships(self):
        block_paths = {}

        def path_of(block_data):
            if block_data is None:
                return ()
            if block_data in block_paths:
                return block_paths[block_data]
            parent = block_data.parent_block()
            path = path_of(parent) + _normalize_key(block_data.index())
            block_paths[block_data] = path
            return path

        for node in self.nodes:
            if node.kind == 'block':
                node.block = self._block_nodes.get(node.block)
            else:
                node.block = self._block_nodes[node.block]
            node.index_set_signature = self._signature_for(node.container)

        for node in self.nodes:
            block_path = path_of(node.block.obj) if node.block is not None else ()
            node.order_key = (
                _KIND_RANK[node.kind],
                block_path,
                type(node.container).__name__,
                node.key,
            )
            node.sort_key = _sort_key(node.order_key)
        self.nodes.sort(key=lambda n: n.sort_key)

    def _signature_for(self, container):
        container_id = id(container)
        signature = self._container_signatures.get(container_id)
        if signature is None:
            signature = self._container_signature(container)
            self._container_signatures[container_id] = signature
        return signature

    @staticmethod
    def _container_signature(container):
        keys = sorted(
            (_normalize_key(data.index()) for data in container.values()),
            key=_sort_key,
        )
        return tuple(keys)

    # -- local, mapping-independent signatures -------------------------

    def _build_base_signatures(self):
        for node in self.nodes:
            node.base_signature = self._base_signature(node)

    def _base_signature(self, node):
        kind = node.kind
        data = node.obj
        if kind == 'block':
            return ('block', node.key, bool(data.active))
        if kind == 'variable':
            value_token = None
            if data.fixed or self.include_variable_values:
                value_token = _constant_token(data.value, self.tolerance)
            return (
                'variable',
                node.key,
                bool(data.active),
                self._domain_token(data),
                bool(data.fixed),
                value_token,
                data._lb is not None,
                data._ub is not None,
            )
        if kind == 'parameter':
            return (
                'parameter',
                node.key,
                bool(data.active),
                not data.is_constant(),
                _constant_token(data.value, self.tolerance),
            )
        if kind == 'constraint':
            return ('constraint', node.key, bool(data.active))
        if kind == 'objective':
            return ('objective', node.key, bool(data.active), str(data.sense.name))
        # named expression
        return ('expression', node.key, bool(data.active))

    @staticmethod
    def _domain_token(var):
        domain = var.domain
        parent_block = getattr(domain, 'parent_block', None)
        if parent_block is None or parent_block() is None:
            # Global, registered domains (Reals, Integers, Binary, ...)
            # carry stable, model-independent names.
            return ('global_domain', getattr(domain, 'name', type(domain).__name__))
        try:
            contents = tuple(
                sorted(
                    (_normalize_scalar(value) for value in domain),
                    key=_sort_key,
                )
            )
        except ModelIsomorphismError:
            raise
        except Exception:
            contents = ('<uniterable-domain>',)
        return (
            'local_domain',
            getattr(domain, 'dimen', None),
            contents,
            tuple(domain.bounds()),
        )


# -------------------------------------------------------------------------------------
# Joint color refinement (Weisfeiler-Lehman style)
# -------------------------------------------------------------------------------------


class _FrozenMultiset:
    """Hashable multiset over colors, shared by every member of a
    container during one refinement round.

    The (cached) hash makes embedding the multiset in a refinement
    signature O(1) per node instead of rehashing a per-node color
    tuple; equality is the ordinary equality of the sorted
    ``(color, count)`` pairs.
    """

    __slots__ = ('pairs', '_hash')

    def __init__(self, pairs):
        self.pairs = pairs
        self._hash = hash(pairs)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, _FrozenMultiset)
            and self.pairs == other.pairs
        )

    def __repr__(self):
        return f"_FrozenMultiset({self.pairs!r})"


class _ColorRefinement:
    def __init__(self, index_a, index_b, tolerance):
        self.index_a = index_a
        self.index_b = index_b
        self.tolerance = tolerance
        self.gids_a = {}
        self.gids_b = {}
        self._round = -1
        self._token_cache_a = {}
        self._token_cache_b = {}
        # Per-round cache mapping container id to the frozen multiset
        # of its members' current colors.
        self._container_multisets = {}

    def run(self):
        signature_to_gid = {}
        for index, gid_map in (
            (self.index_a, self.gids_a),
            (self.index_b, self.gids_b),
        ):
            for node in index.nodes:
                gid = signature_to_gid.setdefault(
                    node.base_signature, len(signature_to_gid)
                )
                gid_map[node] = gid

        previous_partition = None
        while True:
            self._round += 1
            self._token_cache_a = {}
            self._token_cache_b = {}
            self._container_multisets = {}
            shared_signatures = {}
            new_gids_a = {}
            new_gids_b = {}
            for index, gid_map, new_map, cache in (
                (self.index_a, self.gids_a, new_gids_a, self._token_cache_a),
                (self.index_b, self.gids_b, new_gids_b, self._token_cache_b),
            ):
                for node in index.nodes:
                    signature = self._refined_signature(
                        index, node, gid_map, cache
                    )
                    gid = shared_signatures.setdefault(
                        signature, len(shared_signatures)
                    )
                    new_map[node] = gid
            partition = self._partition(new_gids_a, new_gids_b)
            if partition == previous_partition:
                self.gids_a = new_gids_a
                self.gids_b = new_gids_b
                return
            previous_partition = partition
            self.gids_a = new_gids_a
            self.gids_b = new_gids_b
            if self._round + 1 >= _MAX_REFINEMENT_ROUNDS:
                # Coarser colors are safe: they only enlarge candidate
                # sets; the backtracking matcher remains complete.
                return

    def _refined_signature(self, index, node, gid_map, token_cache):
        if node.kind == 'block':
            structure = self._block_structure(index, node, gid_map)
        else:
            structure = self._data_structure(index, node, gid_map, token_cache)
        return (node.base_signature, structure)

    def _data_structure(self, index, node, gid_map, token_cache):
        parent_gid = gid_map[node.block] if node.block is not None else -1
        sibling_gids = self._sibling_signature(index, node, gid_map)
        expression_tokens = tuple(
            (
                label,
                self._cached_token(
                    index, node, label, expr, gid_map, token_cache
                ),
            )
            for label, expr in node.expression_specs
        )
        return (
            ('parent', parent_gid),
            ('container', type(node.container).__name__, node.index_set_signature),
            ('siblings', sibling_gids),
            ('expressions', expression_tokens),
        )

    def _block_structure(self, index, node, gid_map):
        block_data = node.obj
        parent_gid = gid_map[node.block] if node.block is not None else -1
        child_blocks = tuple(
            sorted(
                gid_map[index.node_by_object[child]]
                for component in block_data.component_objects(
                    Block, active=None, descend_into=False
                )
                if not hasattr(component, 'referent')
                for child in component.values()
            )
        )
        containers = []
        for component in block_data.component_objects(
            active=None, descend_into=False
        ):
            if hasattr(component, 'referent'):
                continue
            if component.ctype is Block:
                continue
            if (
                component.ctype.__name__ in _IGNORED_CTYPE_NAMES
                or not isinstance(component, _DATA_CTYPES)
            ):
                continue
            data_gids = tuple(
                sorted(
                    gid_map[index.node_by_object[data]]
                    for data in component.values()
                )
            )
            containers.append(
                (
                    type(component).__name__,
                    index._signature_for(component),
                    data_gids,
                )
            )
        containers.sort(key=_sort_key)
        sibling_gids = self._sibling_signature(index, node, gid_map)
        return (
            ('parent', parent_gid),
            ('siblings', sibling_gids),
            ('child_blocks', child_blocks),
            ('containers', tuple(containers)),
        )

    def _sibling_signature(self, index, node, gid_map):
        """Sibling color structure within a component container.

        The container's full color multiset is built once per
        refinement round (as a shared, hashable object); pairing it
        with the node's own color identifies which occurrence is the
        node itself and is at least as discriminating as the explicit
        sibling multiset (it can additionally separate nodes whose
        containers differ), while costing O(1) per node. A finer
        refinement only prunes more candidates: the backtracking
        matcher remains complete, so correctness is unaffected. Block
        containers are not tracked here: their sibling relationships
        are enforced by the block hierarchy and the container
        bijection in the backtracking matcher.
        """
        container_id = id(node.container)
        members = index._container_nodes.get(container_id)
        if not members:
            return ()
        multiset = self._container_multisets.get(container_id)
        if multiset is None:
            counts = {}
            for member in members:
                gid = gid_map[member]
                counts[gid] = counts.get(gid, 0) + 1
            multiset = _FrozenMultiset(tuple(sorted(counts.items())))
            self._container_multisets[container_id] = multiset
        return (multiset, gid_map[node])

    def _cached_token(self, index, node, label, expr, gid_map, token_cache):
        key = (id(node), label)
        if key not in token_cache:
            tokenizer = _ExpressionTokenizer(
                index,
                leaf_gid=lambda leaf: gid_map[leaf],
                leaf_mapped_token=None,
                tolerance=self.tolerance,
            )
            token_cache[key] = tokenizer.walk_expression(expr)
        return token_cache[key]

    @staticmethod
    def _partition(gids_a, gids_b):
        def groups(gid_map):
            grouped = {}
            for node, gid in gid_map.items():
                grouped.setdefault(gid, []).append(id(node))
            return tuple(
                (gid, tuple(sorted(ids)))
                for gid, ids in sorted(grouped.items())
            )

        return groups(gids_a), groups(gids_b)


# -------------------------------------------------------------------------------------
# Deterministic backtracking bijection search
# -------------------------------------------------------------------------------------


class _BacktrackingMatcher:
    def __init__(self, index_a, index_b, gids_a, gids_b, tolerance):
        self.index_a = index_a
        self.index_b = index_b
        self.gids_a = gids_a
        self.gids_b = gids_b
        self.tolerance = tolerance

        self._b_by_gid = {}
        for node in index_b.nodes:
            self._b_by_gid.setdefault(gids_b[node], []).append(node)
        for candidates in self._b_by_gid.values():
            candidates.sort(key=lambda n: n.sort_key)

        self.map_ab = ComponentMap()
        self._used_b = set()
        # Order-preserving doubly-linked list of unmatched A-side
        # nodes (in deterministic sort order). Removing a matched node
        # and relinking it on backtrack lets every MRV scan visit only
        # unmatched nodes instead of re-scanning the matched prefix.
        self._node_by_id = {id(node): node for node in index_a.nodes}
        self._prev_link = {}
        self._next_link = {}
        previous_id = None
        for node in index_a.nodes:
            node_id = id(node)
            self._prev_link[node_id] = previous_id
            self._next_link[node_id] = None
            if previous_id is not None:
                self._next_link[previous_id] = node_id
            previous_id = node_id
        self._head_id = id(index_a.nodes[0]) if index_a.nodes else None
        self._removed_links = {}
        self._container_map = {}
        self._container_refcount = {}
        self._used_containers_b = set()
        self._b_token_cache = {}
        self.failure_reason = None

    def match(self):
        if not self._recurse():
            if self.failure_reason is None:
                self.failure_reason = (
                    "no consistent one-to-one component mapping exists"
                )
            return False
        return self._final_verification()

    # -- recursion ------------------------------------------------------

    def _recurse(self):
        node_a, candidates = self._select_node()
        if node_a is None:
            return True
        if not candidates:
            self.failure_reason = self._no_candidate_reason(node_a)
            return False
        for node_b in candidates:
            self._assign(node_a, node_b)
            if self._recurse():
                return True
            self._unassign(node_a, node_b)
        return False

    def _select_node(self):
        best = None
        best_candidates = None
        node_id = self._head_id
        while node_id is not None:
            node_a = self._node_by_id[node_id]
            candidates = self._candidates(node_a)
            if best is None or len(candidates) < len(best_candidates) or (
                len(candidates) == len(best_candidates)
                and node_a.sort_key < best.sort_key
            ):
                best = node_a
                best_candidates = candidates
                if len(candidates) <= 1:
                    break
            node_id = self._next_link[node_id]
        return best, best_candidates

    def _candidates(self, node_a):
        candidates = []
        for node_b in self._b_by_gid.get(self.gids_a[node_a], ()):
            if id(node_b) in self._used_b:
                continue
            if node_b.kind != node_a.kind:
                continue
            if node_b.key != node_a.key:
                continue
            if not self._block_compatible(node_a, node_b):
                continue
            if not self._container_compatible(node_a, node_b):
                continue
            if not self._expressions_compatible(node_a, node_b):
                continue
            candidates.append(node_b)
        return candidates

    def _block_compatible(self, node_a, node_b):
        block_a = node_a.block
        block_b = node_b.block
        if block_a is None:
            return block_b is None
        if block_b is None:
            return False
        mapped_block = self.map_ab.get(block_a)
        if mapped_block is not None:
            return mapped_block is block_b
        return id(block_b) not in self._used_b

    def _container_compatible(self, node_a, node_b):
        container_a = node_a.container
        container_b = node_b.container
        mapped = self._container_map.get(id(container_a))
        if mapped is not None:
            return mapped is container_b
        return id(container_b) not in self._used_containers_b

    def _expressions_compatible(self, node_a, node_b):
        if len(node_a.expression_specs) != len(node_b.expression_specs):
            return False
        tokenizer_b = self._tokenizer(self.index_b, self.gids_b, mapped=False)
        tokenizer_a = self._tokenizer(self.index_a, self.gids_a, mapped=True)
        matcher = _TokenMatcher()
        b_cache = self._b_token_cache.setdefault(id(node_b), {})
        for (label_a, expr_a), (label_b, expr_b) in zip(
            node_a.expression_specs, node_b.expression_specs
        ):
            if label_a != label_b:
                return False
            token_a = tokenizer_a.walk_expression(expr_a)
            token_b = b_cache.get(label_b)
            if token_b is None:
                token_b = tokenizer_b.walk_expression(expr_b)
                b_cache[label_b] = token_b
            if not matcher.match(token_a, token_b):
                return False
        return True

    def _tokenizer(self, index, gid_map, mapped):
        if mapped:
            def leaf_mapped(node):
                target = self.map_ab.get(index.node_by_object[node])
                if target is None:
                    return None
                return self.gids_b[target]

            return _ExpressionTokenizer(
                index,
                leaf_gid=lambda leaf: gid_map[leaf],
                leaf_mapped_token=leaf_mapped,
                tolerance=self.tolerance,
            )
        return _ExpressionTokenizer(
            index,
            leaf_gid=lambda leaf: gid_map[leaf],
            leaf_mapped_token=None,
            tolerance=self.tolerance,
        )

    # -- assignment bookkeeping ----------------------------------------

    def _assign(self, node_a, node_b):
        self.map_ab[node_a] = node_b
        self._used_b.add(id(node_b))
        self._remove_unmatched(node_a)
        container_a = node_a.container
        if id(container_a) not in self._container_map:
            self._container_map[id(container_a)] = node_b.container
            self._used_containers_b.add(id(node_b.container))
            self._container_refcount[id(container_a)] = 0
        self._container_refcount[id(container_a)] += 1

    def _unassign(self, node_a, node_b):
        del self.map_ab[node_a]
        self._used_b.discard(id(node_b))
        self._restore_unmatched(node_a)
        container_a = node_a.container
        self._container_refcount[id(container_a)] -= 1
        if self._container_refcount[id(container_a)] == 0:
            del self._container_map[id(container_a)]
            del self._container_refcount[id(container_a)]
            self._used_containers_b.discard(id(node_b.container))

    def _remove_unmatched(self, node_a):
        node_id = id(node_a)
        previous_id = self._prev_link[node_id]
        next_id = self._next_link[node_id]
        self._removed_links[node_id] = (previous_id, next_id)
        if previous_id is None:
            self._head_id = next_id
        else:
            self._next_link[previous_id] = next_id
        if next_id is not None:
            self._prev_link[next_id] = previous_id

    def _restore_unmatched(self, node_a):
        node_id = id(node_a)
        previous_id, next_id = self._removed_links.pop(node_id)
        self._prev_link[node_id] = previous_id
        self._next_link[node_id] = next_id
        if previous_id is None:
            self._head_id = node_id
        else:
            self._next_link[previous_id] = node_id
        if next_id is not None:
            self._prev_link[next_id] = node_id

    # -- final, identity-level verification -----------------------------

    def _final_verification(self):
        if len(self.map_ab) != len(self.index_a.nodes):
            self.failure_reason = "incomplete component mapping after search"
            return False
        if len(self._used_b) != len(self.index_b.nodes):
            self.failure_reason = "mapping is not surjective on the second model"
            return False
        for node_a in self.index_a.nodes:
            node_b = self.map_ab[node_a]
            if not self._verify_node(node_a, node_b):
                self.failure_reason = (
                    f"final verification failed for {node_a.kind} data "
                    f"with structural index {node_a.key}"
                )
                return False
        return True

    def _verify_node(self, node_a, node_b):
        if node_a.base_signature != node_b.base_signature:
            return False
        if len(node_a.expression_specs) != len(node_b.expression_specs):
            return False
        tokenizer_a = self._identity_tokenizer(self.index_a, target=False)
        tokenizer_b = self._identity_tokenizer(self.index_b, target=True)
        for (label_a, expr_a), (label_b, expr_b) in zip(
            node_a.expression_specs, node_b.expression_specs
        ):
            if label_a != label_b:
                return False
            token_a = tokenizer_a.walk_expression(expr_a)
            token_b = tokenizer_b.walk_expression(expr_b)
            if not tokens_equal(token_a, token_b, self.tolerance):
                return False
        return True

    def _identity_tokenizer(self, index, target):
        if target:
            def leaf_token(node):
                leaf = index.node_by_object[node]
                return ('id', leaf.kind, leaf.order_key)

            return _ExpressionTokenizer(
                index,
                leaf_gid=lambda leaf: leaf.order_key,
                leaf_mapped_token=leaf_token,
                tolerance=self.tolerance,
            )

        def mapped_token(node):
            leaf_a = self.index_a.node_by_object[node]
            leaf_b = self.map_ab[leaf_a]
            return ('id', leaf_b.kind, leaf_b.order_key)

        return _ExpressionTokenizer(
            index,
            leaf_gid=lambda leaf: leaf.order_key,
            leaf_mapped_token=mapped_token,
            tolerance=self.tolerance,
        )

    def _no_candidate_reason(self, node_a):
        detail = f"{node_a.kind} data with structural index {node_a.key}"
        if node_a.block is not None and node_a.block not in self.map_ab:
            return f"no candidate found for {detail}: owning block is unmatched"
        return f"no candidate found for {detail}"


# -------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------


class ModelIsomorphismResult:
    """Result of a structural isomorphism comparison.

    Attributes
    ----------
    isomorphic : bool
        ``True`` if the two models are structurally isomorphic.
    mapping : dict
        Maps kind (``'variables'``, ``'constraints'``, ``'objectives'``,
        ``'parameters'``, ``'expressions'``, ``'blocks'``) to a
        :class:`~pyomo.common.collections.ComponentMap` holding the
        deterministic one-to-one correspondence from data objects in
        the first model to the matched data objects in the second
        model. Empty maps when the models are not isomorphic.
    differences : tuple of str
        Human-readable explanations of why the models are not
        isomorphic. Empty when :attr:`isomorphic` is ``True``.
    """

    _PLURAL = {
        'block': 'blocks',
        'variable': 'variables',
        'parameter': 'parameters',
        'expression': 'expressions',
        'constraint': 'constraints',
        'objective': 'objectives',
    }

    def __init__(self, isomorphic, index_a=None, index_b=None, matcher=None,
                 differences=()):
        self.isomorphic = isomorphic
        self.differences = tuple(differences)
        self.mapping = {
            name: ComponentMap()
            for name in self._PLURAL.values()
        }
        if isomorphic and index_a is not None and matcher is not None:
            for node_a in index_a.nodes:
                node_b = matcher.map_ab[node_a]
                self.mapping[self._PLURAL[node_a.kind]][node_a.obj] = node_b.obj

    @property
    def variables(self):
        """ComponentMap of variable data objects (model 1 -> model 2)."""
        return self.mapping['variables']

    @property
    def constraints(self):
        """ComponentMap of constraint data objects (model 1 -> model 2)."""
        return self.mapping['constraints']

    @property
    def objectives(self):
        """ComponentMap of objective data objects (model 1 -> model 2)."""
        return self.mapping['objectives']

    @property
    def parameters(self):
        """ComponentMap parameter data objects (model 1 -> model 2)."""
        return self.mapping['parameters']

    @property
    def expressions(self):
        """ComponentMap of named expression data (model 1 -> model 2)."""
        return self.mapping['expressions']

    @property
    def blocks(self):
        """ComponentMap of block data objects (model 1 -> model 2)."""
        return self.mapping['blocks']

    def __bool__(self):
        return self.isomorphic

    def __repr__(self):
        if self.isomorphic:
            sizes = ', '.join(
                f"{name}={len(entries)}"
                for name, entries in self.mapping.items()
            )
            return f"ModelIsomorphismResult(isomorphic=True, {sizes})"
        return (
            "ModelIsomorphismResult(isomorphic=False, "
            f"differences={list(self.differences)!r})"
        )


def compare_models(model1, model2, *, tolerance=None,
                   include_variable_values=False):
    """Compare two Pyomo models for structural isomorphism.

    The comparison ignores component names and creation order while
    preserving the block hierarchy, algebraic expression structure
    (including commutative reorderings), :func:`Reference` aliases and
    mutable parameter current values.  The models are never modified.

    Parameters
    ----------
    model1, model2 : ConcreteModel or BlockData
        The models (or block subtrees) to compare.
    tolerance : float, optional
        Absolute tolerance applied to native numeric constants and
        parameter values. If ``None`` (the default), numeric values must
        compare exactly (with ordinary Python equality).
    include_variable_values : bool, optional
        If ``True``, the current (typically solver-assigned) values of
        all variables are part of the structural signature. Fixed
        variable values are always compared regardless of this flag.

    Returns
    -------
    ModelIsomorphismResult
        A result object carrying the isomorphism verdict, the
        deterministic component mappings and any differences found.

    Raises
    ------
    ModelIsomorphismError
        If the models use component types or index values that cannot
        be normalized structurally.
    """
    index_a = _ModelIndex(model1, tolerance, include_variable_values)
    index_b = _ModelIndex(model2, tolerance, include_variable_values)

    # The backtracking search and token comparisons recurse once per
    # component; temporarily raise the interpreter recursion limit so
    # that large models can be handled without leaking this setting.
    required_limit = 2000 + 4 * (
        len(index_a.nodes) + len(index_b.nodes)
    )
    previous_limit = sys.getrecursionlimit()
    if required_limit > previous_limit:
        sys.setrecursionlimit(min(required_limit, 1_000_000))
    try:
        return _run_comparison(index_a, index_b, tolerance)
    finally:
        if required_limit > previous_limit:
            sys.setrecursionlimit(previous_limit)


def _run_comparison(index_a, index_b, tolerance):
    unsupported = sorted(
        index_a.unsupported_ctypes | index_b.unsupported_ctypes
    )
    if unsupported:
        raise ModelIsomorphismError(
            "Structural comparison does not support component type(s): "
            f"{', '.join(unsupported)}. Supported component types are "
            "Block, Var, Param, Constraint, Objective and Expression "
            "(Reference aliases are resolved automatically; Suffix, Set, "
            "RangeSet, BuildAction, BuildCheck and ExternalFunction "
            "components are ignored)."
        )

    differences = []
    if not _kind_counts_match(index_a, index_b, differences):
        return ModelIsomorphismResult(False, differences=differences)

    if not _base_signatures_match(index_a, index_b, differences):
        return ModelIsomorphismResult(False, differences=differences)

    refinement = _ColorRefinement(index_a, index_b, tolerance)
    refinement.run()

    if not _refined_partitions_match(
        refinement.gids_a, refinement.gids_b, differences
    ):
        return ModelIsomorphismResult(False, differences=differences)

    matcher = _BacktrackingMatcher(
        index_a, index_b, refinement.gids_a, refinement.gids_b, tolerance
    )
    if not matcher.match():
        differences.append(matcher.failure_reason)
        return ModelIsomorphismResult(False, differences=differences)

    return ModelIsomorphismResult(
        True, index_a=index_a, index_b=index_b, matcher=matcher
    )


def is_isomorphic(model1, model2, **kwds):
    """Return ``True`` if the two models are structurally isomorphic.

    See :func:`compare_models` for the supported keyword arguments.
    """
    return compare_models(model1, model2, **kwds).isomorphic


# -------------------------------------------------------------------------------------
# Diagnostics helpers
# -------------------------------------------------------------------------------------


def _kind_counts_match(index_a, index_b, differences):
    counts_a = {}
    counts_b = {}
    for node in index_a.nodes:
        counts_a[node.kind] = counts_a.get(node.kind, 0) + 1
    for node in index_b.nodes:
        counts_b[node.kind] = counts_b.get(node.kind, 0) + 1
    ok = True
    for kind in sorted(set(counts_a) | set(counts_b),
                       key=lambda k: _KIND_RANK[k]):
        if counts_a.get(kind, 0) != counts_b.get(kind, 0):
            ok = False
            differences.append(
                f"different number of {ModelIsomorphismResult._PLURAL[kind]}: "
                f"{counts_a.get(kind, 0)} vs {counts_b.get(kind, 0)}"
            )
    return ok


def _base_signatures_match(index_a, index_b, differences):
    """Compare the mapping-independent (round-0) component signatures."""
    groups_a = {}
    groups_b = {}
    for node in index_a.nodes:
        groups_a.setdefault(node.base_signature, []).append(node)
    for node in index_b.nodes:
        groups_b.setdefault(node.base_signature, []).append(node)
    ok = True
    for signature in sorted(
        set(groups_a) | set(groups_b), key=lambda s: _sort_key(s)
    ):
        count_a = len(groups_a.get(signature, ()))
        count_b = len(groups_b.get(signature, ()))
        if count_a == count_b:
            continue
        ok = False
        example = (groups_a.get(signature) or groups_b.get(signature))[0]
        differences.append(
            f"{_describe_base_signature(example)} occurs {count_a} time(s) "
            f"in model 1 but {count_b} time(s) in model 2"
        )
    return ok


def _refined_partitions_match(gids_a, gids_b, differences):
    """Compare the stabilized color partitions of the two models."""
    sizes_a = {}
    sizes_b = {}
    for gid in gids_a.values():
        sizes_a[gid] = sizes_a.get(gid, 0) + 1
    for gid in gids_b.values():
        sizes_b[gid] = sizes_b.get(gid, 0) + 1
    mismatched = sum(
        1
        for gid in set(sizes_a) | set(sizes_b)
        if sizes_a.get(gid, 0) != sizes_b.get(gid, 0)
    )
    if mismatched:
        differences.append(
            f"{mismatched} component group(s) differ in their expression "
            "structure, block placement or component connectivity, so no "
            "structure-preserving bijection exists"
        )
        return False
    return True


def _describe_base_signature(node):
    sig = node.base_signature
    kind = sig[0]
    index = node.key
    if kind == 'block':
        detail = f"active={sig[2]}"
    elif kind == 'variable':
        _, _, active, domain, fixed, value_token, has_lb, has_ub = sig
        detail = (
            f"active={active}, domain={domain[0]}:{domain[1]}, "
            f"fixed={fixed}, fixed_value={value_token}, "
            f"has_lower_bound={has_lb}, has_upper_bound={has_ub}"
        )
    elif kind == 'parameter':
        _, _, active, mutable, value_token = sig
        detail = (
            f"active={active}, mutable={mutable}, "
            f"current_value={_token_value_repr(value_token)}"
        )
    elif kind == 'constraint':
        detail = f"active={sig[2]}"
    elif kind == 'objective':
        detail = f"active={sig[2]}, sense={sig[3]}"
    else:
        detail = f"active={sig[2]}"
    return f"{kind} data with structural index {index} ({detail})"


def _token_value_repr(value_token):
    if value_token is None:
        return None
    return value_token[-1]
