rm -r build/
rm -r cppenv.egg-info/
rm -r dist/
rm cppenv/env_wrap.cpp
rm cppenv/env.py
pip uninstall cppenv -y

python setup.py build
python setup.py install