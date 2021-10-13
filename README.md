# RenewML_Energy
Survey of machine learning methods to predict renewable energy grid load and output.

<h4>Data Module</h4>

<ul>

<li><code>Data()</code></li>

<ul>Base object for data loading and analyzing.</ul>

<li><code>Data.get_data(datafile: str, powerfile: str=None, dropna: bool=True, rescale_power: bool=True)</code></li>
<ul>Method to retrieve and format data from CSV.
<li>datafile (str): location of 4y historical weather data file.</li>
<li>powerfile (str): location of power output file from solar stations.</li>
<li>dropna (bool, default: true): Whether to drop NaN values from loaded CSV.</li>
<li>rescale_power (bool, default: true): Whether to rescale power from W to kW.</li>
</ul>

<li><code>Data.transform(lag: _list_ or _str_, )</code></li>



</ul>

<h4>VAR model</h4>
<ul>
<code></code>
</ul>