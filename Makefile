.PHONY: install extract extract-cls clean

install:
	pip install -e .

extract:
	python scripts/extract_features.py --patches

extract-cls:
	python scripts/extract_features.py

clean:
	rm -rf features/ figures/*.png figures/*.pdf
