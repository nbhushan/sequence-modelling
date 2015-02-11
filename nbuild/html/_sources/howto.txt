How to..? 
===============================================

How to Collect data?
-------------------------

1) Connect to EEBOX 'mandoegi'
2) GOTO C:\PowerMeasurement
3) Open PhidgetTest1.10.exe
4) Follow the following sequence.
    First connect to the database using sql credentials.
    SERVER : VIGNEMALE
    DataBase : SimegyModelGenerator
  
  a) Enter 220 in the "Select Voltage Rating" box
  b) Enter 0.9 in the "Enter Power Factor" box
  c) Enter <device_name> in the "Device ID" box.
  d) Click "Add Device"
  e) Select the <device_name> from the "Device ID" dropdown box.
  f) Click open Phidget.
  g) The phidget is now recording power and is storing it in the database.

How to retrieve data?
-------------------------
SERVER : VIGNEMALE
DataBase : SimegyModelGenerator
Table: Phidget Data.

Table attributes:
DeviceID  ; DateTimestamp  ;    PowerRating
nVarchar  ; YYYY-mm-dd hh:mm:ss  ;  numeric


SQL QUERY to retrieve data and then save the results as a .csv file.

  SELECT *
  FROM [SimegyModelGenerator].[dbo].[PhidgetData]
  WHERE DeviceID = '<device_name>' 
  AND DateTimeStamp 
  BETWEEN '<from_date>' AND '<to_date>'
  ORDER By DateTimeStamp


This feature is automatically implemented in the C# application developed using stored procedures.
However, this is not present in Python as yet as there are issues with the ODBC package in Linux. 


How to View data?
---------------------
Choose from several options available.
  1) Use "PhidgetTest1.10.exe" and view the data. This is generated using Microsoft Chart controls.
  2) Use "PowerModelAnalyzer.exe". The chart is displayed using Microsoft d3 which supports dynamic zooming.
  3) Use a python script as shown in exp_debug.py and view the data using Matplotlib. 

How to fit a HMM to the data?
--------------------------------
1) Using PowerModelAnalyze.exe (*Requires accord.net*).
  a) Connect to the database.
  b) Select date range to retrieve data.
  c) Enter number of states and fit the HMM.
  d) View the Viterbi sequence or check the logs for the model parameters.

2) Using Python code - hmm.py.
  Refer hmmtest.py. 
  The module does the following.
    1) Define an initial gaussian emmission object
    2) Generate observations from the emmission object
    3) Define intial A , where pi = A[-1]
    4) Train the model and find params which best fit the observations
    5) Visualize the state sequence.
        
How to fit a QDHMM to the data?
--------------------------------        
Refer qdhmmtest.py. 
