# Licence provenance corrections (v0.5)

Earlier generated v0.1-v0.4 documents described the oarriaga repository as GPL3.
That statement was wrong. Primary-source inspection of its pinned revision
`b861d21b0e76ca5514cdeb5b56a689b7318584f4/LICENSE` gives **MIT**, copyright
2017 Octavio. Archived source URL:
https://raw.githubusercontent.com/oarriaga/face_classification/b861d21b0e76ca5514cdeb5b56a689b7318584f4/LICENSE

The repository's MIT terms do not independently establish rights to the source
training images or remove any FER dataset obligations. No image redistribution.
Keep the upstream notice with redistributed software. Historical reports remain
in Git to preserve the record; this correction supersedes their licence claims.

ViT checkpoint card: Apache2 at revision
`747cf16692eea925b54b7b543cf436848128b68d`; base model google/vit-base-patch16-224-in21k.
Training-image rights are distinct from checkpoint/source licence.

CREMA-D official licence: ODbL1.0 for database, DbCL1.0 for contents; the downloaded
official text is in `eval_results/v0.5/cremad/LICENSE.txt`. The maintainers ask
users to register use via the public form linked in their README. No private
credentials or restricted authentication were used; no form was submitted on the
user's behalf. Preserve attribution and review share-alike terms if redistributing
an adapted database. We retain video locally and publish derived metrics only.

The user-provided FER ZIP has no bundled licence, source URL or subject IDs.
Its structure/counts match common FER2013 exports, but prior assertions of a
fully verified Kaggle licence chain were too strong. Local evaluation is not a
claim of commercial, redistribution or broad-consent rights.
