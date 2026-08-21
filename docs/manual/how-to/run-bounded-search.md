# Run bounded symbolic search

Use this guide to run PTM's finite GNU Prolog searches from the CLI or terminal
workbench.

## Prepare GNU Prolog

Install GNU Prolog as described in [Install PTM](install.md). Make `gprolog`
available on `PATH`, set `PTM_GPROLOG`, or supply the executable through the
CLI option shown by `ptm search KIND --help`.

Run `ptm help bounded-search` for current examples and requirements. The live
parser owns command names, arguments, defaults, and metavariables.

## Run a custom request

Create a `ptm.search.request.v1` JSON document using the
[search-contract reference](../reference/search-contracts.md), then pass it to
the matching search kind. The command kind and document kind must agree.

Decision-tree and repair results that fit the fixed five-binding Logic ABI can
be exported to a new `.ptm` file. PTM refuses to overwrite an existing output.

Successful commands exit with status 0. A valid search with no exact solution
emits `no_solution` and exits with status 3. Invalid requests and runtime
failures exit with status 2. Ctrl+C terminates and reaps the active GNU Prolog
process before the CLI exits.

## Use the workbench

Launch `ptm tui`, open Search, choose a kind to load its bounded example, and
edit the JSON and deadline. The view reports the candidate ceiling before
launch; repair results populate a counterexample table. Use contextual help or
the [generated control reference](../reference/help-topics.md) for bindings.

On Windows, PTM starts `gprolog.exe` headlessly with redirected streams,
`LINEDIT=gui=no`, and `CREATE_NO_WINDOW`. This child-only behavior does not
modify the user's environment or interactive GNU Prolog sessions.
