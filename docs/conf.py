# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

from typing import TYPE_CHECKING, List
import sys
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from sphinx.application import Sphinx


# https://read-the-docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['pyproj', 'h5py', 'osgeo']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Allow Jinja templates in non-template .rst files -------------------------


def rstjinja(app: 'Sphinx', docname: str, source: List[str]) -> None:
    """Allow use of jinja templating in all doc pages.

    Adapted from:
    https://www.ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/
    """
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return

    src = source[0]
    rendered = app.builder.templates.render_string(src,
                                                   app.config.html_context)
    source[0] = rendered


def setup(app: 'Sphinx') -> None:
    """Register event handler for ``source-read`` event.

    See: https://www.sphinx-doc.org/en/master/extdev/appapi.html
    """
    app.connect('source-read', rstjinja)


# -- Path setup ---------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information ------------------------------------------------------

project = u'Raster Vision'
copyright = u'2018, Azavea'
author = u'Azavea'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'0.20'
# The full version, including alpha/beta/rc tags
release = u'0.20-dev'

# -- Extension configuration --------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # support Google-style docstrings
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    'sphinx.ext.napoleon',
    # mardown support
    'myst_parser',
    # allow linking to python docs; see intersphinx_mapping below
    'sphinx.ext.intersphinx',
    # better rendering of pydantic Configs
    'sphinxcontrib.autodoc_pydantic',
    # for linking to source files from docs
    'sphinx.ext.viewcode',
    # for rendering examples in docstrings
    'sphinx.ext.doctest',
    # jupyter notebooks
    'nbsphinx',
    # jupyter notebooks in a gallery
    'sphinx_gallery.load_style',
    # add a copy button to code blocks
    'sphinx_copybutton',
    # search-as-you-type
    'sphinx_search.extension',
]

#########################
# autodoc, autosummary
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
#########################
# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

autosummary_generate = True
autosummary_ignore_module_all = False

autodoc_typehints = 'both'
autodoc_class_signature = 'separated'
autodoc_member_order = 'groupwise'
autodoc_mock_imports = ['torch', 'torchvision', 'pycocotools', 'geopandas']
#########################

#########################
# nbsphinx options
#########################
nbsphinx_execute = 'never'
sphinx_gallery_conf = {
    'line_numbers': True,
}
# external thumnails
nbsphinx_thumbnails = {
    # The _images dir is under build/html. This looks brittle but using the
    # more natural img/tensorboard.png path does not work.
    'tutorials/train': '_images/tensorboard.png',
}
nbsphinx_prolog = r"""
{% set docpath = env.doc2path(env.docname, base=False) %}
{% set docname = docpath.split('/')|last %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note:: This page was generated from `{{ docname }} <https://github.com/azavea/raster-vision/blob/master/docs/{{ docpath }}>`__.
""" # noqa
#########################

#########################
# intersphinx
#########################

# connect docs in other projects
intersphinx_mapping = {
    'python': (
        'https://docs.python.org/3',
        'https://docs.python.org/3/objects.inv',
    ),
    'rasterio': (
        'https://rasterio.readthedocs.io/en/stable/',
        'https://rasterio.readthedocs.io/en/stable/objects.inv',
    ),
    'shapely': (
        'https://shapely.readthedocs.io/en/stable/',
        'https://shapely.readthedocs.io/en/stable/objects.inv',
    ),
    'matplotlib': (
        'https://matplotlib.org/stable/',
        'https://matplotlib.org/stable/objects.inv',
    ),
    'geopandas': (
        'https://geopandas.org/en/stable/',
        'https://geopandas.org/en/stable/objects.inv',
    ),
    'numpy': (
        'https://numpy.org/doc/stable/',
        'https://numpy.org/doc/stable/objects.inv',
    ),
    'pytorch': (
        'https://pytorch.org/docs/stable/',
        'https://pytorch.org/docs/stable/objects.inv',
    ),
}

#########################

#########################
# sphinx_copybutton
# https://sphinx-copybutton.readthedocs.io/en/latest/index.html
#########################

copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: |> '
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_line_continuation_character = '\\'

#########################

# -- General configuration ----------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
root_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#
# today = ''
#
# Else, today_fmt is used as the format for a strftime call.
#
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# These patterns also affect html_static_path and html_extra_path
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 'README.md', '**.ipynb_checkpoints'
]

# The reST default role (used for this markup: `text`) to use for all
# documents.
#
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
#
# To see all availabel values:
# >>> from pygments.styles import get_all_styles
# >>> styles = list(get_all_styles())
#
# pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'
# html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# Furo theme options: https://pradyunsg.me/furo/customisation/
html_theme_options = {
    'sidebar_hide_name': True,
    'top_of_page_button': None,
    'navigation_with_keys': True,
}

# A dictionary of values to pass into the template engine’s context for all
# pages. Single values can also be put in this dictionary using the -A
# command-line option of sphinx-build.
#
# yapf: disable
html_context = dict(
    version=version,
    release=release,
    s3_model_zoo=f'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-{version}', # noqa
)
# yapf: enable

# Add any paths that contain custom themes here, relative to this directory.
#
# html_theme_path = []

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
html_title = f'{project} v{release} documentation'

# A shorter title for the navigation bar.  Default is the same as html_title.
#
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'img/raster-vision-logo.png'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'img/raster-vision-icon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# A list of CSS files. The entry must be a filename string or a tuple
# containing the filename string and the attributes dictionary. The filename
# must be relative to the html_static_path, or a full URI with scheme like
# https://example.org/style.css. The attributes is used for attributes of
# <link> tag. It defaults to an empty list.
html_css_files = ['custom.css']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#
# html_extra_path = []

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
#
# html_last_updated_fmt = None

# Custom sidebar templates, maps document names to template names.
#
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#
# html_additional_pages = {}

# If false, no module index is generated.
#
# html_domain_indices = True

# If false, no index is generated.
#
# html_use_index = True

# If true, the index is split into individual pages for each letter.
#
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr', 'zh'
#
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# 'ja' uses this config value.
# 'zh' user can custom change `jieba` dictionary path.
#
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'RasterVisiondoc'

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (root_doc, 'RasterVision.tex', 'Raster Vision Documentation', 'Azavea',
     'manual'),
]

# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, 'RasterVisoin-{}.tex', html_title, [author], 'manual')]

# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (root_doc, 'RasterVision', 'Raster Vision Documentation', author,
     'RasterVision', 'One line description of project.', 'Miscellaneous'),
]
