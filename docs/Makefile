# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

EXAMPLE_SRCDIR := ../examples
EXAMPLE_BUILDDIR := examples

# Find all Markdown files in ../examples
MD_FILES := $(shell find $(EXAMPLE_SRCDIR) -name '*.md')

# Compute the corresponding .rst output files in examples/
RST_FILES := $(patsubst $(EXAMPLE_SRCDIR)/%.md,$(EXAMPLE_BUILDDIR)/%.rst,$(MD_FILES))


# Make sure the output dir exists
prepare-examples-dir/:
	@mkdir -p $(EXAMPLE_BUILDDIR)

# Rule to convert each example md to rst:
$(EXAMPLE_BUILDDIR)/%.rst: $(EXAMPLE_SRCDIR)/%.md | prepare-examples-dir
	@mkdir -p $(dir $@)
	@pandoc $< -o $@ --wrap=none --lua-filter=rewrite_examples_image_paths.lua

clean-examples:
	@rm -r $(EXAMPLE_BUILDDIR)


convert-markdown-to-rst: $(RST_FILES)
	@echo "All Markdown files have been converted to reStructuredText and post-processed."

.SECONDARY: $(RST_FILES)


.PHONY: convert-markdown-to-rst prepare-examples-dir clean-convert-markdown


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

.PHONY: clean

clean: clean-examples
	@echo "All generated files have been cleaned."


# Catch-all for Sphinx, but ignore file-like targets
%: Makefile
	@if echo "$@" | grep -q '\.'; then \
		:; \
	else \
		echo "Running $@..."; \
		$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); \
	fi
