---
name: Build
about: A part of the frontend API from DESIGN.md, built by /build
title: "area: what to build"
labels: build, ready
---
Spec: `DESIGN.md`, {sections}
Prototype: {files in the prototype to move from, or none}
After: none (or issues whose task must be merged first: #n, #m)

Task: {the types and functions to build}

Built by `/build` (see `tools/prompt_build.md`): an implementer opens a pull request, a reviewer
checks it against DESIGN.md, then the gate merges it. The last comment ends with `Result: done`.
