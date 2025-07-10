{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members:
   :exclude-members: __dict__, __weakref__, __slots__, __init__, __deepcopy__, __hash__

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:
   {% for item in methods %}
      {{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}