# Simulated Bifurcation

This includes a script running [simulated_bifurcation_algorithm](https://github.com/bqth29/simulated-bifurcation-algorithm).

## Python dependencies

```sh
cd PythonScript
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu      
pip install tdqm numpy "simulated-bifurcation==2.0.0"
pip freeze > requirements.txt
deactivate
```