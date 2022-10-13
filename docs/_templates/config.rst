{{ objname | escape | underline}}

.. note:: All Configs are derived from :class:`rastervision.pipeline.config.Config`, which itself is a `pydantic Model <https://pydantic-docs.helpmanual.io/usage/models/>`__.

.. currentmodule:: {{ module }}

.. autopydantic_model:: {{ objname }}
   :model-show-json:
   :model-show-json-error-strategy: warn
   :model-show-config-summary:
   :model-show-validator-summary:
   :model-show-field-summary:
   :undoc-members:
   :inherited-members: BaseModel
