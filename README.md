## Step 0 - populate data folder
`data` folder: contains raw and intermediate data before the processing step 

contains namely: 
```
['fashion-mnist.csv',
 'g2-2-10.csv',
 'g2-2-30.csv',
 'g2-2-50.csv',
 'g2-2-70.csv',
 'g2-2-90.csv',
 'g2-4-10.csv',
 'g2-4-30.csv',
 'g2-4-50.csv',
 'g2-4-70.csv',
 'g2-4-90.csv',
 'g2-8-10.csv',
 'g2-8-30.csv',
 'g2-8-50.csv',
 'g2-8-70.csv',
 'g2-8-90.csv',
 'human-activity-recognition.csv',
 'iris.csv',
 'mnist.csv',
 's1.csv',
 's2.csv',
 's3.csv',
 's4.csv',
 'seeds.csv',
 'u1.csv',
 'wine.csv']
```

## Step 1 - process data
run: `./run_process_data.sh data processed_data`

`processed_data` folder contains csv files with train, validate and test sets

An example csv:
```csv
0.8922738167609834,-0.18212939353359578,6,train
1.5311921584329236,0.14579143051733984,2,train
0.1283434622373828,-1.748731625837468,11,train
0.22713400413211715,1.0114971822631194,9,validate
0.5013168600835375,1.2436928115592418,14,validate
0.27739929888557735,-0.5895817913985576,12,validate
0.26006817718634706,1.8199333381184486,15,test
-1.0240102584403796,1.3313346823820678,9,test
-1.3874756138659605,-0.6516498740556852,7,test
1.307247351146994,-1.36262410266844,5,test
```


## Step 2 - plot processed data
run: `./run_visualize_processed_data.sh processed_data/ plots_processed_data`

`plots_processed_data` contains `.png` files, each representing a plot of a dataset with true cluster labels

## Step 3 - apply gaussian and knock-out datasets shifts
To apply gaussian noise,
run: `./run_apply_dataset_shift_gau.sh processed_data/ processed_data+dataset_shift_gau 0.5
`
To apply knock-out,
run: `./run_apply_dataset_shift_knock.sh processed_data/ processed_data+dataset_shift_knock half`

## Step 4 - plot gaussian and knock-out datasets shifts

run: `./run_visualize_dataset_shifts_gau.sh processed_data+dataset_shift_gau/ plots_dataset_shift`

run: `./run_visualize_dataset_shifts_knock.sh processed_data+dataset_shift_knock/ plots_dataset_shift`

## Step 5 - copy all datasets to an one folder

To copy all contents from `processed_data` to `processed_data+dataset_shift`, so that we can use the `processed_data+dataset_shift` folder to group all datasets by their names
run: `cp -r processed_data/* processed_data+dataset_shift/` run: `cp -r processed_data+dataset_shift_gau/* processed_data+dataset_shift/` run: `cp -r processed_data+dataset_shift_knock/* processed_data+dataset_shift/` 


## Step 6 - group all data by name
move data from `processed_data+dataset_shift` to `grouped_processed_data+dataset_shift` as groups
run: `python group_processed_data.py processed_data+dataset_shift/ grouped_processed_data+dataset_shift`


## Step 7 - perform clustering
run bitch:  `./run_clustering_birch.sh grouped_processed_data+dataset_shift/ predicted_birch`
run dbstream:  `./run_clustering_dbstream.sh grouped_processed_data+dataset_shift/ predicted_dbstream`
run streamKMeans:  `./run_clustering_stream_kmeans.sh grouped_processed_data+dataset_shift/ predicted_stream_kmeans`


## Step 8 - compute CVIs
run: `./run_compute_CVIs+running_time.sh ./predicted_birch/ predicted_birch+CVIs predicted_birch+time`
run: `./run_compute_CVIs+running_time.sh ./predicted_dbstream/ predicted_dbstream+CVIs predicted_dbstream+time`
run: `./run_compute_CVIs+running_time.sh ./predicted_stream_kmeans/ predicted_stream_kmeans+CVIs predicted_stream_kmeans+time`

## Step 9 - compute correlations
