# Raster Vision Documentation

This directory contains the Sphinx documentation for Raster Vision.

## Developing

All commands are issued in this directory.

### Install dependencies

You will need to install the required packages:

```sh
> pip install -r requirements.txt
```

### Build

To generate html, run:

```sh
> make html
```

To run a live local server that updates with changes, run:

```sh
> make livehtml
```

### Getting started

- [Familiarize yourself with Sphinx](https://www.sphinx-doc.org/en/master/).
- Read through the configuration options in [`conf.py`](./conf.py).
- [Familiarize yourself with reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
	- [rst Cheatsheet](https://bashtage.github.io/sphinx-material/rst-cheatsheet/rst-cheatsheet.html)
	- This is used in `.rst` files as well as python docstrings.
- The API reference is automatically generated and uses the `.rst` templates in [`_templates/`](./_templates/) to present different views for different kinds of objects (plugins, modules, classes, configs, functions).
- Documentation pages can be in either `.rst` or `.md` files.
- IPython notebooks are automatically rendered to HTML via the `nbsphinx` extension and can be linked to from other pages like any other `.rst` or `.md` page. See [`npsphinx` documentation](https://nbsphinx.readthedocs.io/en/latest/index.html).
- You can use jinja templates in `.rst` files. By default, Sphinx only allows their use in templates, but the `rstjinja()` hook defined in [`conf.py`](./conf.py) allows us to use it in any `.rst` file.

### Notes

- In [`quickstart.rst`](./quickstart.rst), vector labels will not show up on the map when running locally. This is fine. They will still show correctly up on RTD.
- You can specify a thumbnail for a notebook (which is shown in the gallery) in the following ways:
	- To use the output of a cell in that notebook, add a `nbsphinx-thumbnail` tag to that cell's metadata. If the cell has multiple image outputs, [see instructions here](https://nbsphinx.readthedocs.io/en/latest/gallery/multiple-outputs.html). **Note:** this ONLY works with outputs of *code* cells.
	- To use an arbitrary image, add it to the [`img/`](./img/) directory and then add its path to the `nbsphinx_thumbnails` `dict` in [`conf.py`](./conf.py).
