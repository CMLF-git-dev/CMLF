## Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaption Fusion

0. ### **Dependencies**

   Install packages from `requirements.txt`.  


1. ### **Load Data using qlib**
	```linux
	$ cd ./load_data
	```

	#### Download daily data:

	```python
	$ python load_dataset.py
	```
	* Change parameter `market` to get data from different dataset: `csi300`, `csi800`, `NASDAQ` etc.

	  ##### Data Sample - SH600000 in CSI300

	  ![](https://ftp.bmp.ovh/imgs/2021/02/28e2e1b545cf8ffc.png)
	
	  features dimensions = 6 * 20 + 1 = 121

	#### Download high-frequency data:
	
	```python
	$ python high_freq_resample.py
	```
	
	* Change parameter `N` to get data from different frequencies: `15min`, `30min`, `120min` etc.
	
	  ##### Data Sample - SH600000 in CSI300
	
        ![](https://ftp.bmp.ovh/imgs/2021/02/21213511c92c4c44.png)
	
	  features dimensions = 16 * 6 * 20 + 1 = 1921

2. ### **Framework**
* Pre-training Stage: Contrastive Mechanisms：`./framework/models/contrastive_all_2_encoder.py`
* Adaptive Multi-granularity Feature Fusion: `./framework/models/contrastive_all_2_stage.py`
3. ### **Run**
  ```linux
  $ cd ./framework
  ```

  #### Train `Pre-train` model:

  ```python
  $ python main_contrast.py with config/contrast_all_2_encoder.json model_name=contrastive_all_2_encoder
  ```

  * Add `hyper-param` = {`values`} after `with` or change them in `config/main_model.json`
  * Prediction results of each model are saved as `pred_{model_name}.pkl` in `./out/`.

  #### Train `Adaptive Multi-granularity Feature Fusion` model:

  ```python
  $ python main_contrast_2_stage.py with config/contrast_all_2_stage.json model_name=contrastive_all_2_stage
  ```


  #### Run `Market Trading Simulation`:
  * Prerequisites:   
  	* Server with qlib
  	* Prediction results 
  ```linux
  $ cd ./framework
  ```
  ```python
  $ python trade_sim.py
  ```
4. ### **Records**
	Records for each experiment are saved in `./framework/my_runs/`.  
	Each record file includes: 
	> config.json
	* contains the parameter settings and data path.

	> cout.txt
	* contains the name of dataset, detailed model output, and experiment results.

	> pred_{model_name}_{seed}.pkl
  >
	>  * contains the  `score` (model prediction) and `label`
	
	> run.json
	
	* contains the hash ids of every script used in the experiment. And the source code can be found in `./framework/my_runs/source/`.
