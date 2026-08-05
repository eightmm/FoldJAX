# OpenDDE example jobs

Runnable inputs in OpenDDE's **native** dialect (the Protenix-family
list-of-jobs JSON), recovered from the standalone `opendde_jax` repository when
it was vendored. `tiny.json` is a minimal protein job, `tiny_with_ccd.json` adds
a CCD ligand, `tiny_modified.json` a modified residue, and `2lwu_cyclic.json` a
cyclic peptide with the covalent bonds that closes it.

The suite does not read these — `test_predict_cli.py` and
`test_verify_inputs_cli.py` write their own `tiny.json` into a temporary
directory, which is what keeps them independent of this file. These are kept as
worked examples of the native dialect, which is otherwise only described by the
translator in `foldjax.input`.

For the model-neutral schema that FoldJAX translates *into* this one, see
`examples/` at the repository root.
