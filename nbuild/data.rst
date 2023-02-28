Data 
===============================================

Real Power 
------------
In electrical systems, power is defined as the rate of flow of energy at a given point in the circuit. In a circuit with a purely resistive load the current is in phase with the voltage and hence the power equations become similar to that of an equivalent DC circuit with zero energy being returned back to the source. 
In circuits with reactive loads as well as resistive loads, the power will alternate between positive and negative instantaneous values of energy. But in comparison to a purely reactive load where the net energy dissipation is zero, the energy absorbed will be higher than the energy returned to the source. To reflect the true energy consumption of a device, it is hence necessary to capture the true energy being absorbed. This energy is denoted as real power and is measured in Watts. Most domestic power meters record real power (KWh) and the consumers are charged for the real power consumed. We chose to model the energy consumption of the device based on it's real power consumption.


Data Collection / Phidgets
---------------------------
Phidgets are a viable alternative to traditional sensors and transducers. They are low cost and can be easily controlled by USB from a host computer.
The i-snail-VC 10 sensor was used to detect the AC current consumed by the printer.
http://www.phidgets.com/products.php?product_id=3500_0
The sensor is connected to the Phidget Interface kit (PIK). The PIK is then connected to the computer using USB. An application was developed to control the sensor and store the power data recorded in a database. 
http://www.phidgets.com/products.php?product_id=1018
There are two ways in which we can control the sensor to log data. Depending on the application requirements, we can chose to poll the sensor every predefined time interval and retrieve the value, or we can follow a event driven approach. This is possible by changing the sensitivity threshold of the analog sensor. Sensitivity can be defined here as a minimum difference in value needed to register a change. Whenever the sensor generates a new value, it will be compared to the sensitivity. If the difference between the previous data point and the new data point is less than the sensitivity value, no event will be generated.
Alternatively, one can change the data rate setting for an analog sensor. This corresponds to the fastest rate at which  events will be fired. The data rate is superseded by sensitivity, which should be set to 0 if a constant data rate is required. Data Rate is in milliseconds and corresponds to the amount of time between events. A sampling rate of 1 s chosen. In the application developed using the Phidget API's, we set the sensitivity to be zero and since the sensor does not support an absolute sampling rate of 1s, we sample data at a constant rate of 992ms (any multiple of 16ms). Newer versions of phidgets support sampling rates in multiples of 8ms, thus it would be possible to sample a reading every second.
To enable remote monitoring of the power load, the sensor was connected to an interface kit, which was then connected to an EEBOX using USB. The host application was run on the EEBOX and stored the sensor reading in a database. 
Note that new version of the PIK support data rates which are multiples of 8ms. 