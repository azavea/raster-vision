{{ name | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for attr in attributes %}
      {{ attr }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: class.rst
      :nosignatures:
   {% for class in classes %}
      {{ class }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block configs %}
   {% if not modules %}
   {% set configs = [] %}
   {% for m in members %}
      {% if (m.endswith('Config') and m != 'RVConfig') or m.endswith('Options') %}
         {% set _ = configs.append(m) %}
      {% endif %}
   {%- endfor %}
   {% if configs %}
   .. rubric:: {{ _('Configs') }}

   .. autosummary::
      :toctree:
      :template: config.rst
   {% for cfg in configs %}
      {{ cfg }}
   {%- endfor %}
   {% endif %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% set function_to_show = [] %}
   {% for f in functions %}
      {% if not f.endswith('config_upgrader') %}
         {% set _ = function_to_show.append(f) %}
      {% endif %}
   {%- endfor %}
   {% if function_to_show %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :template: function.rst
   {% for function in function_to_show %}
      {{ function }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for exception in exceptions %}
      {{ exception }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: {{ _('Modules') }}

.. autosummary::
   :toctree:
   :template: module.rst
   :nosignatures:
   :recursive:
{% for module in modules %}
{% if not module.endswith('examples') %}
{% set module_short_name = module.split('.')|last %}
   {{ module_short_name }}  
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
