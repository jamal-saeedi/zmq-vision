# Releasing ZMQ Vision

The package name is `zmq-vision`; the import namespace remains `zmq_inference`. A PyPI name lookup is not a reservation or proof of ownership.

1. Change `__version__` in `src/zmq_inference/__init__.py` and add changelog/release notes.
2. Run tests and the package checks in `CONTRIBUTING.md`. Exercise optional backends on a supported ML Python version.
3. Build distributions with `python -m build`, then `python -m twine check dist/*`.
4. Install the wheel in an empty virtual environment and run `zmq-vision demo` outside the checkout. Optionally install each ML extra and repeat the backend tests.
5. Commit and push to `main`, then wait for CI to succeed. Create and push an annotated `vVERSION` tag. The tag workflow verifies the package version, runs core tests and distribution checks, then creates a GitHub release with the wheel and source archive. Include `docs/releases/VERSION.md` before tagging.

## PyPI publication

Configure PyPI Trusted Publishing for this repository's `publish.yml` workflow and the GitHub `pypi` environment. The environment's reviewer settings apply to both publication paths. See [PyPI's Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

After an explicit publication decision, add a separate paragraph containing `Publish-To-PyPI: yes` to the annotated tag message. The **Publish to PyPI** workflow starts when the corresponding GitHub release workflow succeeds. For an existing release, dispatch **Publish to PyPI** manually with the tag. Both paths download the distributions from GitHub, verify their metadata against the tag, and publish with an OIDC identity.

For a first release, ensure your PyPI account can claim the distribution name. Never reuse a token exposed in chat or commit history. Publishing a version to PyPI is irreversible; fixes require another version.
