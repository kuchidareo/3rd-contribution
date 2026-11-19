# Source provenance

This folder is tracked as ordinary files by the parent `3rd-contribution` repository.
It is not a submodule or a separate Git working tree.

Original source: https://github.com/HazyResearch/ncm
Fork used for this copy: https://github.com/kuchidareo/ncm
Imported revision: c70274273f489abc73832b2b98584a8483b6b203

Local baseline changes are tracked by the parent repository. The original nested
Git metadata is preserved locally at `_local_git_backups/ncm.git` relative to the
parent root; that backup is ignored and must not be committed.

The Kaggle notebooks must clone the parent repository and use
`261119-NCM-replicating/ncm` as their execution directory. Set their `REPO_URL` to
the parent remote URL and `REPO_REF` to the branch or committed revision you push.
