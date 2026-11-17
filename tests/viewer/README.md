# Viewer tests

These tests exercise the optional Qt/PyVista viewer. The standard CI workflow
excludes this directory; backend mesh, dataset and compression tests remain
in the main test suite.

Install the viewer dependencies in your development environment, then run:

```console
python tests/viewer/run_test.py -q
```

The runner accepts pytest arguments and finds the viewer tests independently
of the current working directory. You can also use `pytest tests/viewer`.
The GUI fixtures use Qt's offscreen platform and require a Qt binding even
when no visible windows are shown.
