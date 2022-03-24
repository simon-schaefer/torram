# Torram Tests

### Testing setup
```
pip3 install pytest
pip3 install coverage
```

### Execute tests
```
pytest tests/

# running tests with coverge report
coverage run --source torram -m pytest tests/
coverage report -i --sort=Cover     # terminal coverage report
coverage html                       # html coverage report
```
