# RenewML_Energy
Survey of machine learning methods to predict renewable energy grid load and output.

<h4>Data Module</h4>

<ul>

<li><code>Data()</code></li>

<ul>Base object for data loading and analyzing.</ul>

<li><code>Data.get_data(datafile: <i>str</i>, powerfile: <i>str</i>=None, dropna: <i>bool</i>=True, rescale_power: <i>bool</i>=True)</code></li>
<ul>Method to retrieve and format data from CSV.
<li>datafile (str): location of 4y historical weather data file.</li>
<li>powerfile (str): location of power output file from solar stations.</li>
<li>rescale_power (bool, default: true): Whether to rescale power from W to kW.</li>
</ul>

<li><code>Data.transform(lag: <i>list</i> or <i>str</i>, resample: <i>str</i>=None, scaler: <i>str</i>=None, copy: <i>bool</i>=True)</code></li>
<ul>
Make input data stationary
        <li> lag: Input list of choice from {15minutes|minute|day|week|month|season|year} Input choice of {day|week|month|season|year} for differencing or list of multiple for multi-order differencing.</li>
        <li> scaler: Use scikit-learns scaler to standardize or normalize data. choose from [ minmax | standard ] </li>
        <li> resample: Numpy frequency string. Frequency to resample data to. Default frequency is inferred from powerfile.</li>
</ul>

</ul>

<h4>VAR model</h4>
<ul>
<code></code>
</ul>