# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

convert-markdown-to-rst:
	@echo "Converting Markdown files to reStructuredText..."
	@mkdir -p examples
	@find ../examples -name '*.md' -type f -exec sh -c ' \
		mkdir -p "examples/$$(dirname {})"; \
		pandoc "{}" -o "examples/$$(dirname {})/$$(basename -s .md {}).rst"; \
		sed -i ":a; /:alt:/ { N; s/\n   / /; ta }" "examples/$$(dirname {})/$$(basename -s .md {}).rst"; \
		sed -i "s|\(\(\.\.\/\)\+\)docs/img|\1img|g" "examples/$$(dirname {})/$$(basename -s .md {}).rst" \
	' \;

html: convert-markdown-to-rst
	@echo "Running custom commands..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# package docs for CMS upload
package_docs:
	apt-get update && apt-get install -y zip unzip
	rm -rf deeplearning/
	mkdir deeplearning
	mkdir deeplearning/physicsnemo
	mkdir deeplearning/physicsnemo/physicsnemo-core
	cp -r _build/html/* deeplearning/physicsnemo/physicsnemo-core/
	zip -r deeplearning.zip deeplearning
