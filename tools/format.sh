for f in **/*.py; do docformatter --in-place $f; done
isort --skip-gitignore .
yapf **/*.py --in-place --recursive
