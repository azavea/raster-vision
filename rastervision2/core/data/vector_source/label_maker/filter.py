# Copied from https://github.com/developmentseed/label-maker/blob/master/label_maker/filter.py
# flake8: noqa

# pylint: disable=eval-used,too-many-return-statements
"""Create a feature filtering function from a Mapbox GL Filter."""

# Python port of https://github.com/mapbox/mapbox-gl-js/blob/c9900db279db776f493ce8b6749966cedc2d6b8a/src/style-spec/feature_filter/index.js


def create_filter(filt):
    """Create a feature filtering function from a Mapbox GL Filter.

    Given a filter expressed as nested lists, return a new function
    that evaluates whether a given feature (with a .properties or .tags property)
    passes its test. More information:
    - https://www.mapbox.com/mapbox-gl-js/style-spec/#other-filter
    - https://github.com/mapbox/mapbox-gl-js/tree/master/src/style-spec/feature_filter

    Parameters
    ------------
    filt: list
        Mapbox GL filter

    Returns
    --------
    func: function
        A function which evaluates whether a GeoJSON feature meets the input filter criteria
    """

    def func(f):
        """evaluates whether a given feature passes its filter"""
        p = f.get('properties', {}) if f else {}  # pylint: disable=unused-variable
        return eval(_compile(filt))

    return func


def _compile(filt):
    """Return a string represented the compiled filter function"""
    if not filt:
        return 'True'
    op = filt[0]
    if len(filt) == 1:
        return 'False' if op == 'any' else 'True'
    if op in ['==', '!=', '<', '>', '<=', '>=']:
        return _compile_comparison_op(filt[1], filt[2], op)
    elif op == 'any':
        return _compile_logical_op(filt[1:], ' or ')
    elif op == 'all':
        return _compile_logical_op(filt[1:], ' and ')
    elif op == 'none':
        return _compile_negation(_compile_logical_op(filt[1:], ' or '))
    elif op == 'in':
        return _compile_in_op(filt[1], filt[2:])
    elif op == '!in':
        return _compile_negation(_compile_in_op(filt[1], filt[2:]))
    elif op == 'has':
        return _compile_has_op(filt[1])
    elif op == '!has':
        return _compile_negation(_compile_has_op(filt[1]))
    return 'True'


def _compile_property_reference(prop):
    """Find the correct reference on the input feature"""
    if prop == '$type':
        return 'f.get("geometry").get("type")'
    elif prop == '$id':
        return 'f.get("id")'
    return 'p.get("{}")'.format(prop)


def _compile_comparison_op(prop, value, op):
    """Combine two values with a comparison operator"""
    left = _compile_property_reference(prop)
    right = _stringify(value)
    return left + op + right


def _compile_logical_op(expressions, op):
    """Join multiple logical expressions"""
    return op.join(map(_compile, expressions))


def _compile_in_op(prop, values):
    """Test if a property is within a list of values"""
    return '{} in {}'.format(_compile_property_reference(prop), values)


def _compile_has_op(prop):
    """Test if a property exists on a feature"""
    return '"id" in f' if prop == '$id' else '{} in p'.format(_stringify(prop))


def _compile_negation(expression):
    """Negate the input expression"""
    return 'not ({})'.format(expression)


def _stringify(s):
    """Convert input to string, wrap with quotes if already a string"""
    return '"{}"'.format(s) if isinstance(s, str) else str(s)
