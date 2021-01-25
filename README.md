# User and Entity Behaviour Analytics

Scripts that apply machine learning techniques to Linux log files in an attempt to discern what is
going on.

## Tools

### CLI style comparisons

[bashhistory.py](ueba/parse/bashhistory.py) uses a Variational Auto Encoder neural net using Keras to try to spot the 
differences in style between `.bash_history` files. An explanation can be found 
[here](http://javaagile.blogspot.com/2019/09/neural-nets-and-anomaly-detection.html).

### De-noising /var/logs/syslog

[syslog.py](ueba/parse/syslog.py) uses Fourier transforms to remove the most regular messages in a `syslog` file.
Note that a worrying message that coincides with a regular signal will not be highlighted.

## Caveats

This code is work in progress. Feeback welcome.  