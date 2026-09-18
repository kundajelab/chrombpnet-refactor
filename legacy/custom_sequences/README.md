# Custom-sequence notebooks (archived)

These two notebooks and their `example.fa` input predate the PyTorch port. They
load Keras/TensorFlow `.h5` models, which the production API deliberately does
not accept (see `README.md`, "Python API"), and
`custom_sequence_interpret.ipynb` calls `shap_utils.shuffle_several_times`,
which never existed in the module it imports. They were unreachable from the
CLI and broken as written, so they were moved here with the rest of the
historical assets rather than left under `chrombpnet/evaluation/`.

`chrombpnet/evaluation/interpret/shap_utils.py` was deleted alongside them: no
Python module imported it, these notebooks were its only referrer, and its one
live function duplicated `interpret.ProfileWrapper.forward`.

For contribution scoring on custom sequences, use `chrombpnet contribs_bw`, or
call `chrombpnet.evaluation.interpret.interpret.interpret` directly.
