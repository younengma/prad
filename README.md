#  PrAd: Prompt Adaptive Tuning for Decoder-only Language Models

This is the official implementation for  [PrAd: Prompt Adaptive Tuning for Decoder-only Language Models](https://aclanthology.org/2025.findings-emnlp.254.pdf) (Findings EMNLP 2025).



We used code from the following git repositories:

- [kernel-adapters](https://github.com/ychen-stat-ml/kernel-adapters)
- [VGLM](https://github.com/zlinao/VGLM)

We didn't include the evaluation code in our repository for simplicity.  You can use the following resource for evaluation: 

- [e2e_evaluation](https://github.com/younengma/e2e_metrics)
- [webnlg_evaluation](https://github.com/ychen-stat-ml/kernel-adapters)
- Metrics for others tasks, e.g.  BLEU (MT), Rouge (XSUM),  Accuracy (SST2, MNLI) ,  you can simply use the `evaluate` package.

If you find our method helpful, feel free to cite our article:

```
@inproceedings{ma-etal-2025-prad,
    title = "{P}r{A}d: Prompt Adaptive Tuning for Decoder-only Language Models",
    author = "Ma, Youneng  and
      He, Junyi  and
      Fei, Haojun",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.254/",
    doi = "10.18653/v1/2025.findings-emnlp.254",
    pages = "4729--4743",
    ISBN = "979-8-89176-335-7"
}
```



