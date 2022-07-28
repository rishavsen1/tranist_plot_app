# Plots for transit parameters

1. Step 1
Store the APC parquet data in the [data](https://github.com/rishavsen1/tranist_plot_app/tree/master/data) folder

2. Step 2  
* Open [apc_preprocess.py](https://github.com/rishavsen1/tranist_plot_app/blob/master/apc_preprocess.py) and enter the path to APC parquet in `src/config.py`.  
```
MTA_PARQUET_PREPROCESS = "cleaned-wego-daily.apc.parquet"
CARTA_PARQUET_PREPROCESS = "carta-apc.parquet"
```  
Run the file. This will genereate the preprocessed APC files as a parquet.  
Update the locations of the processed parquet files in `src/config.py`  
```
MTA_PARQUET = "cleaned-wego-daily.apc.out.parquet"
CARTA_PARQUET = "carta_apc_out.parquet"
```

3. Step 3  
* Run [Transit heatmap.py](https://github.com/rishavsen1/tranist_plot_app/blob/master/Transit heatmap.py) by typing: 
```streamlit run Transit heatmap.py```
* This will generate a link to the streamlit app and open up in the browser.

4. Step 4  
To have a feel of how the graph may look click on 'Dummy graph'. <br>
To plot from the APC dataset, choose:
    1. Choose Aggregation time (in minutes) and the dates to filter by on the left side menu
    2. Click the 'Plot graph from dataset' button  

![home_page](https://github.com/rishavsen1/tranist_plot_app/blob/master/example/transit_app_1.png) 

The plot produced is similar to this

![heatmap](https://github.com/rishavsen1/tranist_plot_app/blob/master/example/transit_app_2.png)

5. Step 5  
* Click on the Route-wise statistics to know about other parameters. 
* This page shows shows the Max Occupancy, Baordings, Delays, and Headway aggregated data.

![route page](https://github.com/rishavsen1/tranist_plot_app/blob/master/example/transit_app_3.png)

To plot from the APC dataset, choose:
  1. Choose the bus route from the dropdown menu
  2. Choose Aggregation time (in minutes) and the dates to filter by on the left side menu
  3. Click the 'Plot graphs' button

The plots are produced

![route plots](https://github.com/rishavsen1/tranist_plot_app/blob/master/example/transit_app_4.png)

