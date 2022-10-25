{{ name | escape | underline}}

.. automodule:: {{ fullname }}


{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:

{% for module in modules %}
{% if not module.endswith('examples') %}
{% set module_short_name = module.split('.')|last %}
   {{ module_short_name }}  
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
