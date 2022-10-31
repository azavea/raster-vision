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
- IPython notebooks are automatically rendered to HTML via the `nbsphinx` extension and can be linked to from other pages like any other `.rst` or `.md` page.

### Notes

- In [`quickstart.rst`](./quickstart.rst), vector labels will not show up on the map when running locally. This is fine. They will show up on RTD.
