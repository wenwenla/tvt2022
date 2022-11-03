rm -r build/
rm -r cppenv.egg-info/
rm -r dist/
rm cppenv/env_wrap.cpp
rm cppenv/env.py
pip uninstall cppenv -y

python3 setup.py build
python3 setup.py install