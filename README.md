# SBM-Migration

### Protocol

1. Intall MATLAB engine for python

```
cd "mablab_root/extern/engines/python"
python setup.py install
```

2. Start MATLAB engine

```
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
```

3. Test with MATLAB function

```
x_matlab = matlab.double(x_python.tolist())

result_python = function(x_python)
result_matlab = eng.function(x_matlab)

np.testing.assert_array_almost_equal(result_python, result_matlab)
```
