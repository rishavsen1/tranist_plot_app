**Plots for transit parameters**

Step 1 <br>Open (apc_preprocess.py)[https://github.com/rishavsen1/tranist_plot_app/blob/master/apc_preprocess.py] and enter the path to APC parquet format data on line 83. Run the file. This will genereate the preprocessed APC files as a parquet.
<br>

Step 2<br>
Run (main_page.py)[https://github.com/rishavsen1/tranist_plot_app/blob/master/main_page.py] by typing: 
```streamlit run main_page.py```
This will generate a link to the streamlit app and open up in the browser.

Step 3<br>
Choose the aggregation time (in minutes) and the dates to filter by on the left side menu [!home_page](https://github.com/rishavsen1/tranist_plot_app/blob/master/example/transit_app_1.png)
