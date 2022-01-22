# libamtrack3d_dosimetry

This code requires python 3.8 or newer.

Create a local Python virtual enviroment in hidden `.env` directory:

```
python3 -m venv .env
```

Activate venv:

```
source .env/bin/activate
```

Install requirements

```
python -m pip install -r requirements.txt
```

To run demonstration code:

```
python demo.py
```

To see the analysis code take a look at `model.ipynb` jupyter notebook.

When your work is finished deactivate venv:

```
deactivate
```

Simulation parameters are defined in `settings.py`, necessary libamtrack wrappers are in `helper.py`.

