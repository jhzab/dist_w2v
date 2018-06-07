# Asynchronous Training of Word Embeddings for large text corpora

## How to use

In the `code` directory is the script `execute_epochs.sh`. It
handles/sets:
* calling hadoop streaming
* number of VCOREs per reducer
* number of reducers
* name of the job
* number of epochs

Additionally, the files `mapper.py` and `reducer.py` need to be modified.

In `mapper.py` one has to set the number of reducers/sampling rate and if
required the sub sampling of the actual data.

In the `reducer.py` the active hadoop namenode has to be set, as well as
the path for the models. That path needs to exist (and be word writeable).
