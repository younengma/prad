## training
### command for training the model: 
`python train.py configs/<task>/<config_file>  <seed>  <learning_rate>`.  Here `seed` and `learning_rate` are optional, if provided it will overwrite the values in the config file.

### explanation of config file

The meaning of different arguments can be seen in `arguments.py`. Here we mention several arguments that needs special attention:
- apply_c_masks:  if True, we apply a prompt mask only to the input so that the adpaters only work on the prompt not the decoded tokens. In this way, we turn a conventional adapter into PrAd-Adapter. 
- "train_file", "validation_file", "test_file" can be omitted as each task have a decicated data file configured in the file `data_utils.py`. You only need to provide the task name.
- full_tune: if Set True, full finetuning the model, adapter configurations will be ignored.



