## training
### Command for training the model: 
`python train.py configs/<task>/<config_file>  <seed>  <learning_rate>`.  Here `seed` and `learning_rate` are optional, if provided it will overwrite the values in the config file.  e.g. `python train.py configs/mt/adapter.json 1234 2e-4`

### Explanation  of  config file

The meanings of the various arguments are defined in arguments.py. Below are a few key arguments that require special attention:
**apply_c_masks**:
When set to True, this applies a prompt mask only to the input tokens, ensuring that the adapters (or LoRA modules) operate exclusively on the prompt and not on the generated (decoded) tokens. This effectively transforms a standard adapter into a PrAd-Adapter.
In the configuration files:
Files with a "0" in their name (e.g., adapter0.json, lora0.json) have apply_c_masks = false, indicating they use the conventional adapter/LoRA.
Files like adapter.json and lora.json have `apply_c_masks = true`,  indicating they implement PrAd-Adapter and PrAd-LoRA, respectively.

**train_file, validation_file, test_file**:
These can be omitted. Each task has its corresponding data files pre-configured in `data_utils.py`. You only need to specify the task name, and the correct data paths will be loaded automatically.

**full_tune**:
If set to True, the entire model will be fine-tuned end-to-end, and any adapter-related configurations (including LoRA or PrAd settings) will be ignored.

### data process

The dataset included in the 'data' folder contains only 100 examples. If you plan to run the full experiments, you’ll need to download the complete dataset from the official website and convert it into the same format as the provided samples.







