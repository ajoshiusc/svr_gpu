import os

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SVORT_URL_DICT = {
    "v1": "https://zenodo.org/record/7486938/files/checkpoint.pt?download=1",
    "v2": "https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1",
}
MONAIFBS_URL = "https://zenodo.org/record/4282679/files/models.tar.gz?download=1"
IQA2D_URL = "https://zenodo.org/record/7368570/files/pytorch.ckpt?download=1"
IQA3D_URL = (
    "https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/"
    "weights_resnet_sw2_k3.hdf5"
)
