# AFID

**A Unified Open Framework for Automated Fingermark Identification, Quality Assessment and Feature Extraction**

[![arXiv](https://img.shields.io/badge/arXiv-2609.07439-b31b1b.svg)](https://arxiv.org/abs/2609.07439)

AFID is an open framework for processing friction ridge images. It performs fingermark recognition, quality assessment and feature extraction with a single shared encoder, trained only on publicly available data.

- **Recognition.** A fixed-length representation learned for identity discrimination, with essentially no preprocessing beyond resizing and padding.
- **Quality assessment.** A quality module on the same frozen backbone predicts the recognition utility of a fingermark.
- **Feature extraction.** Lightweight decoders recover minutiae, ridge orientation and segmentation.

For details and results, see the paper on [arXiv](https://arxiv.org/abs/2609.07439). The paper is currently under review.

## Code and models

🚧 **Coming soon.** The code, pretrained models, annotations and other resources will be released in this repository.

## Related work

AFID continues our earlier work on automated fingermark quality assessment in [OpenAFQA](https://github.com/timoblak/OpenAFQA), which is now archived.

## Citation

If you find this work useful, please cite:

```bibtex
@article{oblak2026afid,
  title   = {AFID: A Unified Open Framework for Automated Fingermark Identification, Quality Assessment and Feature Extraction},
  author  = {Oblak, Tim and Haraksim, Rudolf and Peer, Peter},
  journal = {arXiv preprint arXiv:2609.07439},
  year    = {2026}
}
```
